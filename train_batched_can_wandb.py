import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from generate_av_integration_data import AVIntegrationDataset
from train_batched_can import (
    BatchedContinuousAttractor,
    bump_stability_loss,
    plot_can_kernels,
)
from train_ring_attractor import (
    bump_amplitude_loss,
    create_initial_bump,
    decode_angle_from_argmax,
    decode_angle_from_population_vector,
    cosine_similarity_loss,
)


def _load_sweep_configs(config_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not config_path:
        return None
    path = Path(config_path)
    with path.open() as f:
        if path.suffix in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "PyYAML is required to read YAML configs. Install with `pip install pyyaml`."
                ) from exc
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Sweep config must be a dict or a list of dicts.")


def _default_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "num_neurons": args.num_neurons,
        "seq_len": args.seq_len,
        "training_steps": args.training_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "tau": args.tau,
        "dt": args.dt,
        "smoothing_width": args.smoothing_width,
        "smoothing_strength": args.smoothing_strength,
        "max_av": args.max_av,
        "fast_mode": not args.slow_mode,
        "initialization": args.initialization,
        "n_act_choice": args.n_act_choice,
        "init_noise_std": args.init_noise_std,
        "weight_lr_scale": args.weight_lr_scale,
        "random_weight_std": args.random_weight_std,
        "train_ring_gains": not args.fix_ring_gains,
        "stability_weight": args.stability_weight,
        "stability_eps": args.stability_eps,
        "log_interval": args.log_interval,
    }


def _spectral_radius(W: torch.Tensor) -> float:
    eigvals = torch.linalg.eigvals(W)
    return eigvals.abs().max().item()


def compute_w_eff_radius(
    model: BatchedContinuousAttractor,
    left_scale: float = 0.0,
    right_scale: float = 0.0,
) -> float:
    with torch.no_grad():
        ones = torch.ones_like(model.W_sym)
        sym = model.J_I * ones + model.J_E * model.W_sym
        asym = left_scale * model.W_asym[0] + right_scale * model.W_asym[1]
        W_eff = (sym + model.g_v * asym) / model.num_neurons
        return _spectral_radius(W_eff.detach().cpu())


def train_with_wandb(config: Dict[str, Any], project: str, entity: Optional[str], run_name: Optional[str]):
    auto_name = (
        f"{config['initialization']}"
        f"_lr{config['learning_rate']:.1e}"
        f"_wlr{config['weight_lr_scale']}"
        f"_stab{config['stability_weight']}"
        f"_JE{config.get('init_je', 'auto')}"
        f"_JI{config.get('init_ji', -1.0)}"
        f"_g{config.get('init_gv', 1.0)}"
    )
    run = wandb.init(project=project, entity=entity, config=config, name=run_name or auto_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AVIntegrationDataset(
        num_samples=config["training_steps"] * config["batch_size"],
        seq_len=config["seq_len"],
        zero_padding_start_ratio=0.5,
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        max_av=config["max_av"],
        device=device,
        fast_mode=config["fast_mode"],
    )

    model = BatchedContinuousAttractor(
        num_neurons=config["num_neurons"],
        tau=config["tau"],
        dt=config["dt"],
        smoothing_width=config["smoothing_width"],
        smoothing_strength=config["smoothing_strength"],
        initialization=config["initialization"],
        n_act_choice=config["n_act_choice"],
        init_noise_std=config["init_noise_std"],
        random_weight_std=config["random_weight_std"],
        train_ring_gains=config["train_ring_gains"],
    ).to(device)

    initial_W_sym = model.W_sym.detach().clone()
    initial_W_asym = model.W_asym.detach().clone()
    initial_J_E = model.J_E.detach().item()
    initial_J_I = model.J_I.detach().item()
    initial_g_v = model.g_v.detach().item()

    print(f"[wandb] Using device: {device}")

    if config["weight_lr_scale"] != 1.0:
        weight_params = [p for n, p in model.named_parameters() if "W_sym" in n or "W_asym" in n]
        other_params = [p for n, p in model.named_parameters() if "W_sym" not in n and "W_asym" not in n]
        param_groups = [
            {"params": weight_params, "lr": config["learning_rate"] * config["weight_lr_scale"]},
            {"params": other_params, "lr": config["learning_rate"]},
        ]
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    loss_history = []
    for step in range(config["training_steps"]):
        av_signal, target_angle = dataset.generate_batch(config["batch_size"])
        initial_angle = target_angle[:, 0]
        r_init = create_initial_bump(initial_angle, config["num_neurons"], device=device)
        target_amp = torch.sum(torch.abs(r_init), dim=1, keepdim=True) / config["num_neurons"]

        cosine_activity, bump_activity = model(av_signal, r_init=r_init)

        main_loss = cosine_similarity_loss(cosine_activity, target_angle)
        amp_loss = bump_amplitude_loss(bump_activity, target_amplitude=target_amp)
        stability_loss = bump_stability_loss(bump_activity, av_signal, eps=config["stability_eps"])
        total_loss = main_loss + 0.2 * amp_loss + config["stability_weight"] * stability_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(total_loss.item())

        if (step % config["log_interval"]) == 0:
            right_mean = torch.relu(av_signal).mean().item()
            left_mean = torch.relu(-av_signal).mean().item()
            w_eff_radius = compute_w_eff_radius(model)
            w_eff_rot_radius = compute_w_eff_radius(model, left_mean, right_mean)
            print(
                f"[wandb] Step {step}/{config['training_steps']} "
                f"total={total_loss.item():.4f} angle={main_loss.item():.4f} "
                f"amp={amp_loss.item():.4f} stable={stability_loss.item():.4f} "
                f"J_E={model.J_E.item():.3f} J_I={model.J_I.item():.3f} "
                f"|W_sym|={model.W_sym.norm().item():.2f} "
                f"|W_asym_L|={model.W_asym[0].norm().item():.2f} "
                f"|W_asym_R|={model.W_asym[1].norm().item():.2f} "
                f"rho_maint={w_eff_radius:.3f} rho_rot={w_eff_rot_radius:.3f}"
            )
            metrics = {
                "train/total_loss": total_loss.item(),
                "train/angle_loss": main_loss.item(),
                "train/bump_loss": amp_loss.item(),
                "train/stability_loss": stability_loss.item(),
                "input/right_mean": right_mean,
                "input/left_mean": left_mean,
                "params/J_E": model.J_E.item(),
                "params/J_I": model.J_I.item(),
                "params/g_v": model.g_v.item(),
                "params/W_sym_norm": model.W_sym.norm().item(),
                "params/W_asym_L_norm": model.W_asym[0].norm().item(),
                "params/W_asym_R_norm": model.W_asym[1].norm().item(),
                "params/W_eff_radius": w_eff_radius,
                "params/W_eff_rot_radius": w_eff_rot_radius,
                "step": step,
            }
            wandb.log(metrics, step=step)

    # Kernel drift logging
    sym_delta = torch.norm(model.W_sym.detach() - initial_W_sym).item()
    asym_delta_left = torch.norm(model.W_asym[0].detach() - initial_W_asym[0]).item()
    asym_delta_right = torch.norm(model.W_asym[1].detach() - initial_W_asym[1]).item()
    kernel_metrics = {
        "drift/delta_J_E": model.J_E.item() - initial_J_E,
        "drift/delta_J_I": model.J_I.item() - initial_J_I,
        "drift/delta_g_v": model.g_v.item() - initial_g_v,
        "drift/W_sym": sym_delta,
        "drift/W_asym_left": asym_delta_left,
        "drift/W_asym_right": asym_delta_right,
    }
    print(
        "[wandb] Kernel drift summary: "
        f"dJ_E={kernel_metrics['drift/delta_J_E']:+.4f} "
        f"dJ_I={kernel_metrics['drift/delta_J_I']:+.4f} "
        f"d g_v={kernel_metrics['drift/delta_g_v']:+.4f} "
        f"||ΔW_sym||={kernel_metrics['drift/W_sym']:.3f} "
        f"||ΔW_asym_left||={kernel_metrics['drift/W_asym_left']:.3f} "
        f"||ΔW_asym_right||={kernel_metrics['drift/W_asym_right']:.3f}"
    )
    wandb.log(kernel_metrics, step=config["training_steps"])

    # Plot kernels
    initial_kernels = (
        (
            (initial_J_I * torch.ones_like(initial_W_sym) + initial_J_E * initial_W_sym).cpu().numpy(),
            initial_g_v * initial_W_asym[0].cpu().numpy(),
            initial_g_v * initial_W_asym[1].cpu().numpy(),
        )
    )
    plot_can_kernels(model, title_prefix="Trained", initial_kernels=initial_kernels)
    wandb.log({"plots/kernels": wandb.Image("trained_batched_can_kernels.png")})

    # Evaluation
    evaluate_and_log(model, config, device)

    run.finish()


def evaluate_and_log(model: BatchedContinuousAttractor, config: Dict[str, Any], device: torch.device):
    test_dataset = AVIntegrationDataset(
        num_samples=1,
        seq_len=1200,
        zero_padding_start_ratio=0.3,
        zero_ratios_in_rest=[0.1],
        max_av=config["max_av"],
        device=device,
        fast_mode=config["fast_mode"],
    )
    av_signal_test, target_angle_test = test_dataset.generate_batch(1)
    cosine_activity_test, bump_activity_test = model(av_signal_test)

    initial_angle_test = target_angle_test[:, 0:1]
    offset_test = np.pi - initial_angle_test
    aligned_target_angle_test = (target_angle_test + offset_test) % (2 * np.pi)

    decoded_angle_pv = decode_angle_from_population_vector(cosine_activity_test)
    decoded_angle_argmax = decode_angle_from_argmax(cosine_activity_test)

    rmse = torch.sqrt(
        torch.mean((decoded_angle_pv - aligned_target_angle_test) ** 2)
    ).item()

    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    fig.suptitle("Batched CAN Evaluation")

    im0 = axes[0].imshow(cosine_activity_test[0].detach().cpu().numpy().T, aspect="auto")
    axes[0].set_title("Cosine activity (delta_7 basis)")
    axes[0].set_ylabel("Neuron")

    axes[1].plot(bump_activity_test[0, 0].detach().cpu().numpy(), label="t=0")
    mid_idx = bump_activity_test.shape[1] // 2
    axes[1].plot(
        bump_activity_test[0, mid_idx].detach().cpu().numpy(),
        label=f"t={mid_idx}",
    )
    axes[1].plot(bump_activity_test[0, -1].detach().cpu().numpy(), label="final")
    axes[1].legend()
    axes[1].set_title("Bump activity snapshots")

    axes[2].plot(cosine_activity_test[0, 0].detach().cpu().numpy(), label="t=0")
    axes[2].plot(
        cosine_activity_test[0, mid_idx].detach().cpu().numpy(),
        label=f"t={mid_idx}",
    )
    axes[2].plot(cosine_activity_test[0, -1].detach().cpu().numpy(), label="final")
    axes[2].legend()
    axes[2].set_title("Cosine activity snapshots")

    axes[3].plot(av_signal_test[0].detach().cpu().numpy(), color="black")
    axes[3].set_title("Angular velocity input")
    axes[3].set_ylabel("rad/step")

    axes[4].plot(aligned_target_angle_test[0].cpu().numpy(), label="Target (aligned)")
    axes[4].plot(decoded_angle_pv[0].detach().cpu().numpy(), label="Decoded PV")
    axes[4].plot(decoded_angle_argmax[0].detach().cpu().numpy(), label="Decoded Argmax")
    axes[4].set_title("Angle tracking")
    axes[4].legend()

    axes[5].plot(np.unwrap(aligned_target_angle_test[0].cpu().numpy()), label="Target (unwrapped)")
    axes[5].plot(np.unwrap(decoded_angle_pv[0].detach().cpu().numpy()), label="Decoded PV (unwrapped)")
    axes[5].set_xlabel("Time step")
    axes[5].set_ylabel("Angle (rad)")
    axes[5].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    eval_path = f"trained_batched_can_results_{config['num_neurons']}neurons.png"
    plt.savefig(eval_path)
    wandb.log(
        {
            "eval/rmse": rmse,
            "plots/eval": wandb.Image(eval_path),
        }
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Batched CAN with Weights & Biases logging")
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=None, help="wandb entity/user")
    parser.add_argument("--run_name", default=None, help="Override auto-generated run name (optional)")
    parser.add_argument("--config_path", help="Optional JSON/YAML file with a list of run configs")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between wandb logs")

    parser.add_argument("--num_neurons", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--training_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--smoothing_width", type=int, default=4)
    parser.add_argument("--smoothing_strength", type=float, default=1.0)
    parser.add_argument("--max_av", type=float, default=0.1 * np.pi)
    parser.add_argument(
        "--initialization",
        choices=["canonical", "perfect", "sym+random", "random"],
        default="random",
    )
    parser.add_argument("--n_act_choice", type=int, default=None)
    parser.add_argument("--init_noise_std", type=float, default=0.05)
    parser.add_argument("--weight_lr_scale", type=float, default=1.0)
    parser.add_argument("--random_weight_std", type=float, default=0.5)
    parser.add_argument("--stability_weight", type=float, default=0.1)
    parser.add_argument("--stability_eps", type=float, default=5e-3)
    parser.add_argument("--fix_ring_gains", action="store_true")
    parser.add_argument("--slow_mode", action="store_true")

    args = parser.parse_args()

    base_config = _default_config_from_args(args)
    sweep_configs = _load_sweep_configs(args.config_path)
    configs = sweep_configs or [base_config]

    for idx, cfg in enumerate(configs):
        combined = base_config.copy()
        combined.update(cfg)
        forced_name = cfg.get("run_name")
        name = forced_name or args.run_name  # fallback to auto name inside train_with_wandb
        train_with_wandb(combined, project=args.project, entity=args.entity, run_name=name)


if __name__ == "__main__":
    main()
