import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from generate_av_integration_data import AVIntegrationDataset
from train_ring_attractor import (
    create_initial_bump,
    cosine_similarity_loss,
    bump_amplitude_loss,
    decode_angle_from_population_vector,
    decode_angle_from_argmax,
)


def je_optimal_candidate(num_neurons: int, n_act: Optional[int] = None) -> float:
    """
    Port of CAN.py's je_optimal_candidate.
    Provides a discrete cosine weight that minimizes energy corrugation.
    """
    if n_act is None:
        n_act = max(2, num_neurons - 2)
    ntilde = n_act - (num_neurons / 2.0)
    num = math.sin(2 * math.pi * ntilde / num_neurons)
    den = math.sin(2 * math.pi / num_neurons)
    val = 0.25 + (1.0 / (2.0 * num_neurons)) * (ntilde + (num / den))
    return 1.0 / val


class BatchedContinuousAttractor(nn.Module):
    """
    Batched Euler integration of the CAN dynamics from CAN.py with the same readable
    decomposition used in train_general_ring_no_gain_relu.py:

        W_sym_total = J_I * 1 + J_E * W_sym     (global inh + structured maint. weights)
        W_asym_L/R ≈ Wa_L/R                    (antisymmetric rotation weights for L/R turns)
        τ dh/dt = -h + (1/N)(W_sym_total @ r + g_v * (L*W_asym_L + R*W_asym_R) @ r) + c_ff
        r = ReLU(h),  h ← (1-α)h + α * input
    """

    def __init__(
        self,
        num_neurons: int,
        tau: float = 5.0,
        dt: float = 1.0,
        init_je: Optional[float] = None,
        init_ji: float = -2.0,
        init_c_ff: float = 1.0,
        init_gv: float = 5.0,
        smoothing_width: int = 4,
        smoothing_strength: float = 0.2,
        initialization: str = "canonical",
        n_act_choice: Optional[int] = None,
        init_noise_std: float = 0.05,
        random_weight_std: float = 1.0,
        train_ring_gains: bool = True,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.tau = tau
        self.dt = dt
        self.alpha = dt / tau  # Euler coefficient
        self.smoothing_strength = smoothing_strength
        self.initialization = initialization
        self.init_noise_std = init_noise_std
        self.n_act_choice = n_act_choice
        self.random_weight_std = random_weight_std
        self.train_ring_gains = train_ring_gains

        theta = torch.linspace(0, 2 * math.pi, steps=num_neurons + 1, dtype=torch.float32)[:-1]
        angle_diff = theta.unsqueeze(1) - theta.unsqueeze(0)

        cos_diff = torch.cos(angle_diff)
        sin_diff = torch.sin(angle_diff)
        self.register_buffer("cos_diff", cos_diff)
        self.register_buffer("sin_diff", sin_diff)
        self.register_buffer("W_delta7", torch.cos(angle_diff))

        indices = torch.arange(num_neurons, dtype=torch.float32)
        circ = torch.minimum(
            torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0)),
            torch.tensor(float(num_neurons)) - torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0)),
        )
        smoothing_width = max(1, smoothing_width)
        smooth_mask = (circ <= smoothing_width).float()
        smooth_mask = smooth_mask / (smooth_mask.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("smooth_matrix", smooth_mask)
        self.register_buffer("ones_matrix", torch.ones(num_neurons, num_neurons, dtype=torch.float32))

        # Learnable scalars that mirror CAN.py
        if init_je is None:
            init_je = je_optimal_candidate(num_neurons, n_act_choice)
        self.J_E = nn.Parameter(torch.tensor(init_je, dtype=torch.float32))
        self.J_I = nn.Parameter(torch.tensor(init_ji, dtype=torch.float32))
        self.c_ff = nn.Parameter(torch.tensor(init_c_ff, dtype=torch.float32))
        self.g_v = nn.Parameter(torch.tensor(init_gv, dtype=torch.float32))
        if not self.train_ring_gains:
            self.J_E.requires_grad_(False)
            self.J_I.requires_grad_(False)

        # Learnable structured weight matrices (W_sym sustains the bump, W_asym rotates it)
        self.W_sym = nn.Parameter(self._init_structured_kernel(cos_diff))
        asym_template = cos_diff if initialization == "sym+random" else sin_diff
        self.W_asym = nn.Parameter(self._init_asym_kernels(asym_template))

    def _init_structured_kernel(self, template: torch.Tensor) -> torch.Tensor:
        """Initializes the symmetric maintenance weights (W_sym analogue)."""
        if self.initialization in {"canonical", "perfect", "sym+random"}:
            noise = self.init_noise_std * torch.randn_like(template) / math.sqrt(self.num_neurons)
            kernel = template.clone() + noise
        elif self.initialization == "random":
            kernel = torch.randn_like(template) * self.random_weight_std
        else:
            raise ValueError(f"Unknown initialization '{self.initialization}'")
        return kernel

    def _init_asym_kernels(self, template: torch.Tensor) -> torch.Tensor:
        """Initializes left/right antisymmetric weights (Wa analogue)."""
        if self.initialization in {"canonical", "perfect"}:
            noise = self.init_noise_std * torch.randn(2, *template.shape) / math.sqrt(self.num_neurons)
            left = -template.clone()
            right = template.clone()
            kernel = torch.stack([left, right], dim=0) + noise
        elif self.initialization == "sym+random":
            noise = self.init_noise_std * torch.randn(2, *template.shape) / math.sqrt(self.num_neurons)
            base = template.clone()
            kernel = torch.stack([base, base], dim=0) + noise
        elif self.initialization == "random":
            kernel = torch.randn(
                2, *template.shape, device=template.device, dtype=template.dtype
            ) * self.random_weight_std
        else:
            raise ValueError(f"Unknown initialization '{self.initialization}'")
        return kernel

    def forward(self, av_signal: torch.Tensor, r_init: Optional[torch.Tensor] = None):
        """
        Args:
            av_signal: (batch, seq_len) angular velocities.
            r_init: optional initial bump (batch, num_neurons). If None, defaults to pi-centered bump.
        Returns:
            cosine_activity: (batch, seq_len, num_neurons)
            bump_activity: (batch, seq_len, num_neurons)
        """
        batch_size, seq_len = av_signal.shape

        if av_signal.device != self.W_sym.device:
            av_signal = av_signal.to(self.W_sym.device)

        if r_init is None:
            initial_angle = torch.full((batch_size,), math.pi, device=av_signal.device)
            h = create_initial_bump(initial_angle, self.num_neurons, device=av_signal.device)
        else:
            h = r_init

        sym_matrix = self.J_I * self.ones_matrix + self.J_E * self.W_sym
        base_W_eff = sym_matrix.unsqueeze(0)
        W_asym_left = self.W_asym[0].unsqueeze(0)
        W_asym_right = self.W_asym[1].unsqueeze(0)

        cosine_history = []
        bump_history = []
        for t in range(seq_len):
            rates = torch.relu(h)
            velocity_t = av_signal[:, t].unsqueeze(1)
            left_signal = torch.relu(-velocity_t).unsqueeze(2)
            right_signal = torch.relu(velocity_t).unsqueeze(2)

            W_asym_eff = left_signal * W_asym_left + right_signal * W_asym_right
            # Explicitly form W_eff = J_I*1 + J_E*W_sym + g_v*(L*W_asym_L + R*W_asym_R)
            W_eff = base_W_eff + self.g_v * W_asym_eff

            recurrent_input = (
                torch.bmm(W_eff, rates.unsqueeze(2)).squeeze(2) / self.num_neurons + self.c_ff
            )

            h = h * (1 - self.alpha) + recurrent_input * self.alpha

            if self.smoothing_strength > 0:
                smoothed = torch.matmul(h, self.smooth_matrix)
                h = (1 - self.smoothing_strength) * h + self.smoothing_strength * smoothed

            rates = torch.relu(h)
            bump_history.append(rates)
            cosine_history.append(torch.matmul(rates, self.W_delta7))

        cosine_activity = torch.stack(cosine_history, dim=1)
        bump_activity = torch.stack(bump_history, dim=1)
        return cosine_activity, bump_activity

    @torch.no_grad()
    def export_effective_weights(self):
        """Return numpy copies of the current symmetric and antisymmetric kernels."""
        ones = self.ones_matrix.detach().cpu().numpy()
        w_sym = (
            self.J_I.detach().cpu().item() * ones
            + self.J_E.detach().cpu().item() * self.W_sym.detach().cpu().numpy()
        )
        w_asym_left = self.g_v.detach().cpu().item() * self.W_asym[0].detach().cpu().numpy()
        w_asym_right = self.g_v.detach().cpu().item() * self.W_asym[1].detach().cpu().numpy()
        return w_sym, w_asym_left, w_asym_right, self.smooth_matrix.detach().cpu().numpy()


def plot_can_kernels(model: BatchedContinuousAttractor, title_prefix: str, initial_kernels=None):
    """Visualize the effective kernels used by the batched CAN."""
    w_sym, w_asym_left, w_asym_right, smoothing = model.export_effective_weights()
    if initial_kernels is not None:
        init_w_sym, init_w_left, init_w_right = initial_kernels
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        rows = [("Initial", init_w_sym, init_w_left, init_w_right), ("Trained", w_sym, w_asym_left, w_asym_right)]
        for row_idx, (label, sym, left, right) in enumerate(rows):
            im0 = axes[row_idx, 0].imshow(sym, cmap="viridis")
            axes[row_idx, 0].set_title(f"{label} W_sym")
            fig.colorbar(im0, ax=axes[row_idx, 0])

            im1 = axes[row_idx, 1].imshow(left, cmap="coolwarm")
            axes[row_idx, 1].set_title(f"{label} W_asym (Left turn)")
            fig.colorbar(im1, ax=axes[row_idx, 1])

            im2 = axes[row_idx, 2].imshow(right, cmap="coolwarm")
            axes[row_idx, 2].set_title(f"{label} W_asym (Right turn)")
            fig.colorbar(im2, ax=axes[row_idx, 2])

            im3 = axes[row_idx, 3].imshow(smoothing, cmap="magma")
            axes[row_idx, 3].set_title("Smoothing kernel" if row_idx == 0 else "")
            fig.colorbar(im3, ax=axes[row_idx, 3])
        fig.suptitle(f"{title_prefix} Kernels (Initial vs Trained)")
    else:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f"{title_prefix} Kernels")
        im0 = axes[0].imshow(w_sym, cmap="viridis")
        axes[0].set_title("W_sym (local exc + global inh)")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(w_asym_left, cmap="coolwarm")
        axes[1].set_title("W_asym (Left turn)")
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(w_asym_right, cmap="coolwarm")
        axes[2].set_title("W_asym (Right turn)")
        fig.colorbar(im2, ax=axes[2])

        im3 = axes[3].imshow(smoothing, cmap="magma")
        axes[3].set_title("Smoothing kernel")
        fig.colorbar(im3, ax=axes[3])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"{title_prefix.lower().replace(' ', '_')}_batched_can_kernels.png"
    plt.savefig(filename)
    print(f"Saved kernel visualization to {filename}")
    plt.close(fig)


def run_batched_can_experiment(
    num_neurons: int = 64,
    seq_len: int = 200,
    training_steps: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    tau: float = 5.0,
    dt: float = 1.0,
    smoothing_width: int = 4,
    smoothing_strength: float = 0.0,
    max_av: float = 0.1 * math.pi,
    fast_mode: bool = True,
    initialization: str = "canonical",
    n_act_choice: Optional[int] = None,
    init_noise_std: float = 0.05,
    weight_lr_scale: float = 1.0,
    random_weight_std: float = 1.0,
    train_ring_gains: bool = True,
    stability_weight: float = 0.1,
    stability_eps: float = 5e-3,
):
    """
    Train the batched CAN to reproduce target angles from AVIntegrationDataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training batched CAN with Euler dynamics matching CAN.py")

    dataset = AVIntegrationDataset(
        num_samples=training_steps * batch_size,
        seq_len=seq_len,
        zero_padding_start_ratio=0.3,
        zero_ratios_in_rest=[0.2, 0.5, 0.8],
        max_av=max_av,
        device=device,
        fast_mode=fast_mode,
    )

    model = BatchedContinuousAttractor(
        num_neurons=num_neurons,
        tau=tau,
        dt=dt,
        smoothing_width=smoothing_width,
        smoothing_strength=smoothing_strength,
        initialization=initialization,
        n_act_choice=n_act_choice,
        init_noise_std=init_noise_std,
        random_weight_std=random_weight_std,
        train_ring_gains=train_ring_gains,
    ).to(device)

    initial_W_sym = model.W_sym.detach().clone()
    initial_W_asym = model.W_asym.detach().clone()
    initial_J_E = model.J_E.detach().item()
    initial_J_I = model.J_I.detach().item()
    initial_g_v = model.g_v.detach().item()
    initial_kernels = (
        (
            (initial_J_I * torch.ones_like(initial_W_sym) + initial_J_E * initial_W_sym)
            .cpu()
            .numpy(),
            initial_g_v * initial_W_asym[0].cpu().numpy(),
            initial_g_v * initial_W_asym[1].cpu().numpy(),
        )
    )

    if weight_lr_scale != 1.0:
        scalar_params = []
        weight_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "W_sym" in name or "W_asym" in name:
                weight_params.append(param)
            else:
                scalar_params.append(param)
        optimizer = torch.optim.Adam(
            [
                {"params": weight_params, "lr": learning_rate * weight_lr_scale},
                {"params": scalar_params, "lr": learning_rate},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    bump_loss_history = []

    print("Starting training...")
    for step in range(training_steps):
        av_signal, target_angle = dataset.generate_batch(batch_size)
        initial_angle = target_angle[:, 0]
        r_init = create_initial_bump(initial_angle, num_neurons, device=device)
        target_amplitude = torch.sum(torch.abs(r_init), dim=1, keepdim=True) / num_neurons

        cosine_activity, bump_activity = model(av_signal, r_init=r_init)

        main_loss = cosine_similarity_loss(cosine_activity, target_angle)
        amp_loss = bump_amplitude_loss(bump_activity, target_amplitude=0.6)
        stability_loss = bump_stability_loss(bump_activity, av_signal, eps=stability_eps)
        total_loss = main_loss + 0.2 * amp_loss + stability_weight * stability_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(total_loss.item())
        bump_loss_history.append(amp_loss.item())

        if step % 100 == 0:
            print(
                f"Step {step:04d} | total {total_loss.item():.4f} "
                f"| angle {main_loss.item():.4f} | amp {amp_loss.item():.4f} "
                f"| stable {stability_loss.item():.4f} "
                f"| J_E {model.J_E.item():.3f} J_I {model.J_I.item():.3f} "
                f"| ||W_sym|| {model.W_sym.norm().item():.2f} "
                f"| ||W_asym_L|| {model.W_asym[0].norm().item():.2f} "
                f"||W_asym_R|| {model.W_asym[1].norm().item():.2f} "
                f"| g_v {model.g_v.item():.3f}"
            )

    sym_delta = torch.norm(model.W_sym.detach() - initial_W_sym).item()
    asym_delta_left = torch.norm(model.W_asym[0].detach() - initial_W_asym[0]).item()
    asym_delta_right = torch.norm(model.W_asym[1].detach() - initial_W_asym[1]).item()
    print("Training complete.")
    print(
        "Kernel drift summary:\n"
        f"  ΔJ_E = {model.J_E.item() - initial_J_E:+.4f}, ΔJ_I = {model.J_I.item() - initial_J_I:+.4f}, "
        f"Δg_v = {model.g_v.item() - initial_g_v:+.4f}\n"
        f"  ||ΔW_sym||_F = {sym_delta:.4f}, "
        f"||ΔW_asym_left||_F = {asym_delta_left:.4f}, "
        f"||ΔW_asym_right||_F = {asym_delta_right:.4f}"
    )
    plot_can_kernels(model, title_prefix="Trained", initial_kernels=initial_kernels)

    # Evaluation on a long held-out trajectory
    model.eval()
    test_dataset = AVIntegrationDataset(
        num_samples=1,
        seq_len=1200,
        zero_padding_start_ratio=0.2,
        zero_ratios_in_rest=[0.1],
        max_av=max_av,
        device=device,
        fast_mode=fast_mode,
    )

    av_signal_test, target_angle_test = test_dataset.generate_batch(1)
    cosine_activity_test, bump_activity_test = model(av_signal_test)

    initial_angle_test = target_angle_test[:, 0:1]
    offset_test = math.pi - initial_angle_test
    aligned_target_angle_test = (target_angle_test + offset_test) % (2 * math.pi)

    decoded_angle_pv = decode_angle_from_population_vector(cosine_activity_test)
    decoded_angle_argmax = decode_angle_from_argmax(cosine_activity_test)

    fig, axes = plt.subplots(6, 1, figsize=(12, 16))
    fig.suptitle("Batched CAN Evaluation")

    im0 = axes[0].imshow(cosine_activity_test[0].detach().cpu().numpy().T, aspect="auto")
    axes[0].set_title("Cosine activity (delta_7 basis)")
    axes[0].set_ylabel("Neuron")
    fig.colorbar(im0, ax=axes[0])

    axes[1].plot(bump_activity_test[0, 0].detach().cpu().numpy(), label="t=0")
    mid_idx = bump_activity_test.shape[1] // 2
    axes[1].plot(
        bump_activity_test[0, mid_idx].detach().cpu().numpy(),
        label=f"t={mid_idx}",
    )
    axes[1].plot(bump_activity_test[0, -1].detach().cpu().numpy(), label="final")
    axes[1].set_title("Bump activity snapshots")
    axes[1].legend()

    axes[2].plot(cosine_activity_test[0, 0].detach().cpu().numpy(), label="t=0")
    axes[2].plot(
        cosine_activity_test[0, mid_idx].detach().cpu().numpy(),
        label=f"t={mid_idx}",
    )
    axes[2].plot(cosine_activity_test[0, -1].detach().cpu().numpy(), label="final")
    axes[2].set_title("Cosine activity snapshots")
    axes[2].legend()

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
    eval_path = f"trained_batched_can_results_{num_neurons}neurons.png"
    plt.savefig(eval_path)
    print(f"Saved evaluation plot to {eval_path}")
    plt.close(fig)

    fig_loss, (ax_total, ax_amp) = plt.subplots(1, 2, figsize=(12, 4))
    fig_loss.suptitle("Training curves (Batched CAN)")

    ax_total.plot(loss_history)
    ax_total.set_title("Total loss")
    ax_total.set_xlabel("Step")
    ax_total.set_ylabel("Loss")
    ax_total.set_yscale("log")

    ax_amp.plot(bump_loss_history)
    ax_amp.set_title("Bump amplitude loss")
    ax_amp.set_xlabel("Step")
    ax_amp.set_ylabel("Loss")
    ax_amp.set_yscale("log")

    plt.tight_layout()
    curves_path = f"training_curves_batched_can_{num_neurons}neurons.png"
    plt.savefig(curves_path)
    print(f"Saved training curves to {curves_path}")
    plt.close(fig_loss)

    return model, loss_history


def bump_stability_loss(bump_activity: torch.Tensor, av_signal: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Encourage the bump to remain unchanged whenever the angular velocity input is ~0.
    Looks at consecutive time steps where |av| < eps and penalizes bump deltas.
    """
    if bump_activity.shape[1] < 2:
        return torch.zeros(1, device=bump_activity.device, dtype=bump_activity.dtype)

    silent_mask = (av_signal.abs() < eps).float()
    silent_pairs = silent_mask[:, 1:] * silent_mask[:, :-1]
    if silent_pairs.sum() == 0:
        return torch.zeros(1, device=bump_activity.device, dtype=bump_activity.dtype)

    diffs = bump_activity[:, 1:] - bump_activity[:, :-1]
    sq_norm = diffs.pow(2).sum(dim=2)
    loss = (sq_norm * silent_pairs).sum() / silent_pairs.sum().clamp_min(1.0)
    return loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train batched CAN dynamics on AV integration data")
    parser.add_argument("--num_neurons", type=int, default=64, help="Ring size")
    parser.add_argument("--seq_len", type=int, default=200, help="Sequence length for training batches")
    parser.add_argument("--training_steps", type=int, default=2000, help="Number of optimization steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--tau", type=float, default=5.0, help="Membrane time constant τ")
    parser.add_argument("--dt", type=float, default=1.0, help="Integration step size Δt")
    parser.add_argument("--smoothing_width", type=int, default=4, help="Half-width for smoothing kernel")
    parser.add_argument("--smoothing_strength", type=float, default=0.0, help="Blend factor for smoothing (0 disables)")
    parser.add_argument("--max_av", type=float, default=0.1 * math.pi, help="Max angular velocity magnitude")
    parser.add_argument(
        "--initialization",
        choices=["canonical", "perfect", "sym+random", "random"],
        default="canonical",
        help="Kernel initialization (matches W_sym/W_asym choices).",
    )
    parser.add_argument("--n_act_choice", type=int, default=None, help="N_act used for je_optimal_candidate (default N-2).")
    parser.add_argument("--init_noise_std", type=float, default=0.05, help="Std of noise added to canonical templates.")
    parser.add_argument(
        "--random_weight_std",
        type=float,
        default=1.0,
        help="Std of random weights when using initialization='random'.",
    )
    parser.add_argument(
        "--weight_lr_scale",
        type=float,
        default=1.0,
        help="Multiplier for learning rate applied to W_sym/W_asym parameters.",
    )
    parser.add_argument(
        "--fix_ring_gains",
        action="store_true",
        help="Freeze J_E and J_I (stay at analytic CAN values).",
    )
    parser.add_argument(
        "--stability_weight",
        type=float,
        default=0.1,
        help="Weight for the zero-velocity stability regularizer.",
    )
    parser.add_argument(
        "--stability_eps",
        type=float,
        default=5e-3,
        help="Velocity magnitude threshold for stability loss.",
    )
    parser.add_argument(
        "--slow_mode",
        action="store_true",
        help="Disable ultra-fast dataset generation (useful for debugging on CPU)",
    )

    args = parser.parse_args()
    run_batched_can_experiment(
        num_neurons=args.num_neurons,
        seq_len=args.seq_len,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tau=args.tau,
        dt=args.dt,
        smoothing_width=args.smoothing_width,
        smoothing_strength=args.smoothing_strength,
        max_av=args.max_av,
        fast_mode=not args.slow_mode,
        initialization=args.initialization,
        n_act_choice=args.n_act_choice,
        init_noise_std=args.init_noise_std,
        weight_lr_scale=args.weight_lr_scale,
        random_weight_std=args.random_weight_std,
        train_ring_gains=not args.fix_ring_gains,
        stability_weight=args.stability_weight,
        stability_eps=args.stability_eps,
    )
