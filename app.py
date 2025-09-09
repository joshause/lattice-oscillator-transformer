# --------------------------------------------------------------------------- #
#  Lattice Oscillator Transformer
# --------------------------------------------------------------------------- #

import math, time, random, warnings
from typing import Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
import numpy as np

# Optional visualisation back-end -------------------------------------------------
try:
    import matplotlib.pyplot as plt, seaborn as sns
    from matplotlib.animation import FuncAnimation, PillowWriter 
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = sns = FuncAnimation = None

# Reproducibility helper ----------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Core helpers --------------------------------------------------------------------
def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

def _check_divisible(d_model: int, n_heads: int) -> None:
    if d_model % n_heads:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

# --------------------------------------------------------------------------- #
#  Lattice attention head
# --------------------------------------------------------------------------- #
class LatticeAttentionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        lattice_pos: Tuple[int, int],
        coupling_strength: float = 0.1,
        intrinsic_freq: float = 1.0,
        head_id: int = 0,
    ):
        super().__init__()
        self.d_k, self.lattice_pos, self.head_id = d_k, lattice_pos, head_id
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

        # Phase-oscillator state (persistent buffers)
        self.register_buffer("phase", torch.tensor(0.0))
        self.register_buffer("intrinsic_freq", torch.tensor(float(intrinsic_freq)))
        self.coupling_strength = nn.Parameter(torch.tensor(coupling_strength))
        self.register_buffer("scale", torch.tensor(math.sqrt(float(d_k))))

        # Monitoring
        self.monitor_dynamics = False
        self._phase_history: List[float] = []
        self._coupling_history: List[float] = []
        self._attention_entropy_history: List[float] = []

    # Monitoring API --------------------------------------------------------------
    def enable_monitoring(self):
        self.monitor_dynamics = True
        for lst in (self._phase_history, self._coupling_history, self._attention_entropy_history):
            lst.clear()

    def disable_monitoring(self):
        self.monitor_dynamics = False

    def get_dynamics_history(self) -> Dict[str, List[float]]:
        return dict(phases=self._phase_history.copy(),
                    couplings=self._coupling_history.copy(),
                    attention_entropy=self._attention_entropy_history.copy())

    # Forward ---------------------------------------------------------------------
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        neighbor_phases: Tensor,
        neighbor_weights: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, L, _ = query.shape
        Q = self.W_q(query)  # (B, L, d_k)
        K = self.W_k(key)
        V = self.W_v(value)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, L, L)

        # Kuramoto update
        if neighbor_phases.numel():
            phase_diff = neighbor_phases - self.phase
            coupling_force = (neighbor_weights * torch.sin(phase_diff)).sum()
            new_phase = self.phase + 0.01 * (self.intrinsic_freq + self.coupling_strength * coupling_force)
            if self.monitor_dynamics:
                self._coupling_history.append(coupling_force.item())
        else:
            new_phase = self.phase + 0.01 * self.intrinsic_freq
            if self.monitor_dynamics:
                self._coupling_history.append(0.0)

        with torch.no_grad():
            self.phase.copy_(new_phase.remainder(2 * math.pi))
        if self.monitor_dynamics:
            self._phase_history.append(self.phase.item())

        # Phase bias
        scores = scores + 0.1 * torch.cos(self.phase)

        # Masking
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(B, -1, -1)
            elif mask.dim() == 3 and mask.size(0) == 1:
                mask = mask.expand(B, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        if self.monitor_dynamics:
            with torch.no_grad():
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean().item()
                self._attention_entropy_history.append(entropy)

        out = torch.matmul(attn, V)
        return out, attn.detach()

# --------------------------------------------------------------------------- #
#  Multi-head lattice attention
# --------------------------------------------------------------------------- #
class LatticeMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        lattice_shape: Optional[Tuple[int, int]] = None,
        use_sdpa: bool = True,
    ):
        super().__init__()
        _check_divisible(d_model, n_heads)
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")

        # Build lattice
        if lattice_shape is None:
            side = int(math.sqrt(n_heads))
            lattice_shape = (side, side + 1) if side * side < n_heads else (side, side)
        self.lattice_shape: Tuple[int, int] = lattice_shape

        self.heads = nn.ModuleList()
        self.positions: List[Tuple[int, int]] = []
        for idx in range(n_heads):
            row, col = divmod(idx, lattice_shape[1])
            self.positions.append((row, col))
            freq = 1.0 + 0.1 * idx
            self.heads.append(
                LatticeAttentionHead(d_model, self.d_k, (row, col), intrinsic_freq=freq, head_id=idx)
            )

        self._build_neighbors()
        self.out_proj = nn.Linear(d_model, d_model)

        # Research
        self.monitor_lattice = False
        self._global_sync_history: List[float] = []
        self._attention_maps_history: List[Tensor] = []

    # Research API ----------------------------------------------------------------
    def enable_monitoring(self):
        self.monitor_lattice = True
        self._global_sync_history.clear()
        self._attention_maps_history.clear()
        for h in self.heads:
            h.enable_monitoring()

    def disable_monitoring(self):
        self.monitor_lattice = False
        for h in self.heads:
            h.disable_monitoring()

    # Neighbour topology ----------------------------------------------------------
    def _build_neighbors(self) -> None:
        n = len(self.heads)
        self.register_buffer("neighbor_idx", torch.full((n, 8), -1, dtype=torch.long))
        self.register_buffer("neighbor_w", torch.zeros((n, 8)))

        for i, (r1, c1) in enumerate(self.positions):
            k = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if k >= 8:
                        break
                    r2, c2 = r1 + dr, c1 + dc
                    for j, (r, c) in enumerate(self.positions):
                        if (r, c) == (r2, c2):
                            dist = math.hypot(dr, dc)
                            self.neighbor_idx[i, k] = j
                            self.neighbor_w[i, k] = math.exp(-dist)
                            k += 1
                            break

    # Analysis helpers ------------------------------------------------------------
    def get_lattice_state(self) -> Dict[str, np.ndarray]:
        phases = np.array([h.phase.item() for h in self.heads])
        couplings = np.array([h.coupling_strength.item() for h in self.heads])
        frequencies = np.array([h.intrinsic_freq.item() for h in self.heads])
        positions = np.array(self.positions)
        return dict(phases=phases, couplings=couplings, frequencies=frequencies,
                    positions=positions, lattice_shape=self.lattice_shape)

    def compute_synchronization_order(self) -> float:
        phases = torch.stack([h.phase for h in self.heads])
        complex_phases = torch.exp(1j * phases)
        order_param = torch.abs(torch.mean(complex_phases)).item()
        if self.monitor_lattice:
            self._global_sync_history.append(order_param)
        return order_param

    # Forward ---------------------------------------------------------------------
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        use_lattice_dynamics: bool = True,
    ) -> Tuple[Tensor, List[Tensor]]:

        B, L, _ = query.shape
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, L, L)

        # Fast SDPA path (no lattice)
        if not use_lattice_dynamics and self.use_sdpa and mask is None:
            q = torch.stack([h.W_q(query) for h in self.heads], dim=1)  # (B, n_heads, L, d_k)
            k = torch.stack([h.W_k(key) for h in self.heads], dim=1)
            v = torch.stack([h.W_v(value) for h in self.heads], dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
            out = out.transpose(1, 2).reshape(B, L, self.d_model)
            return self.out_proj(out), []

        # Full lattice path
        device = query.device
        valid = (self.neighbor_idx >= 0).long()
        nbr_idx = self.neighbor_idx.clamp(min=0)
        all_phases = torch.stack([h.phase for h in self.heads])
        nbr_phases = all_phases[nbr_idx] * valid
        phase_diff = nbr_phases - all_phases.unsqueeze(1)
        coupling_forces = (self.neighbor_w * torch.sin(phase_diff)).sum(dim=1)

        with torch.no_grad():
            new_phases = all_phases + 0.01 * (
                torch.stack([h.intrinsic_freq for h in self.heads]) + coupling_forces
            )
            for h_idx, head in enumerate(self.heads):
                head.phase.copy_(new_phases[h_idx].remainder(2 * math.pi))

        outs, attns = [], []
        for head in self.heads:
            out, attn = head(query, key, value,
                             torch.tensor([], device=device),
                             torch.tensor([], device=device), mask)
            outs.append(out)
            attns.append(attn)

        if self.monitor_lattice and attns:
            self._attention_maps_history.append(torch.stack(attns, dim=1).detach().cpu())

        if use_lattice_dynamics:
            self.compute_synchronization_order()

        multi_out = torch.cat(outs, dim=-1)
        return self.out_proj(multi_out), attns

# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class LatticeTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        lattice_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.lattice_attn = LatticeMultiHeadAttention(d_model, n_heads, lattice_shape)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, use_lattice: bool = True) -> Tensor:
        h, _ = self.lattice_attn(x, x, x, mask, use_lattice_dynamics=use_lattice)
        x = self.norm1(x + h)
        return self.norm2(x + self.ff(x))

# --------------------------------------------------------------------------- #
#  Full model
# --------------------------------------------------------------------------- #
class LatticeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        lattice_shape: Optional[Tuple[int, int]] = None,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            LatticeTransformerBlock(d_model, n_heads, d_ff, lattice_shape)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_emb.weight  # tie weights
        self.register_buffer("pos_buf", torch.arange(max_seq_len))
        self.research_mode = False
        self._training_step = 0

    # Research API ------------------------------------------------------------------
    def enable_research_mode(self):
        self.research_mode = True
        for b in self.blocks:
            b.lattice_attn.enable_monitoring()

    def disable_research_mode(self):
        self.research_mode = False
        for b in self.blocks:
            b.lattice_attn.disable_monitoring()

    def get_lattice_states(self) -> List[Dict[str, np.ndarray]]:
        return [b.lattice_attn.get_lattice_state() for b in self.blocks]

    def get_synchronization_metrics(self) -> Dict[str, List[float]]:
        return {f"block_{i}": b.lattice_attn._global_sync_history.copy()
                for i, b in enumerate(self.blocks)}

    # Forward -----------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        causal: bool = True,
        return_dict: bool = False,
        use_lattice_dynamics: bool = True,
    ):
        B, L = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids) + self.pos_emb(self.pos_buf[:L])
        x *= math.sqrt(self.d_model)

        mask = _make_causal_mask(L, device) if causal else None
        # Padding mask (combine with causal if needed)
        if self.pad_token_id is not None:
            pad_mask = (input_ids != self.pad_token_id).unsqueeze(1).expand(B, L, L)
            mask = mask & pad_mask if mask is not None else pad_mask

        for b in self.blocks:
            x = b(x, mask, use_lattice=use_lattice_dynamics)

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict:
            return {"logits": logits, "loss": loss}
        return logits if loss is None else (logits, loss)

# --------------------------------------------------------------------------- #
#  Visualiser
# --------------------------------------------------------------------------- #
class LatticeVisualizer:
    def __init__(self, model: LatticeTransformer):
        self.model = model
        self.state_history: List[Dict] = []
        self.sync_history: List[float] = []
        self.loss_history: List[Dict[str, float]] = []

        if not VISUALIZATION_AVAILABLE:
            warnings.warn("matplotlib/seaborn unavailable – visualiser runs in stub mode.")

    # Research recording ------------------------------------------------------------
    def record_state(self, training_step: int = None):
        if not self.model.research_mode:
            return
        self.state_history.append({
            "step": training_step or len(self.state_history),
            "timestamp": time.time(),
            "lattice_states": self.model.get_lattice_states(),
            "sync_metrics": self.model.get_synchronization_metrics(),
        })

    # Plotting stubs (full code identical to original, omitted for brevity) ----------
    def plot_lattice_snapshot(self, block_idx: int = 0, figsize: Tuple[int, int] = (15, 5)):
        """Plot current state of lattice for specified block."""
        if not VISUALIZATION_AVAILABLE or not self.state_history:
            print("Visualization not available")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Get latest state
        latest_state = self.state_history[-1]['lattice_states'][block_idx]
        phases = latest_state['phases']
        couplings = latest_state['couplings']
        frequencies = latest_state['frequencies']
        positions = latest_state['positions']
        shape = latest_state['lattice_shape']
        
        # Create grids
        phase_grid = np.full(shape, np.nan)
        coupling_grid = np.full(shape, np.nan)
        freq_grid = np.full(shape, np.nan)
        
        for idx, (i, j) in enumerate(positions):
            if idx < len(phases):
                phase_grid[i, j] = phases[idx]
                coupling_grid[i, j] = couplings[idx]
                freq_grid[i, j] = frequencies[idx]
        
        # Plot phase distribution with vectors
        im1 = axes[0].imshow(phase_grid, cmap='hsv', vmin=0, vmax=2*np.pi)
        axes[0].set_title(f'Phase Distribution (Block {block_idx})')
        plt.colorbar(im1, ax=axes[0], label='Phase (rad)')
        
        # Add phase direction arrows
        for idx, (i, j) in enumerate(positions):
            if idx < len(phases):
                phase = phases[idx]
                dx, dy = 0.3 * np.cos(phase), 0.3 * np.sin(phase)
                axes[0].arrow(j, i, dx, dy, head_width=0.1, head_length=0.1,
                            fc='white', ec='white', alpha=0.8)
        
        # Coupling strengths
        im2 = axes[1].imshow(coupling_grid, cmap='viridis')
        axes[1].set_title('Coupling Strengths')
        plt.colorbar(im2, ax=axes[1], label='Coupling')
        
        # Intrinsic frequencies
        im3 = axes[2].imshow(freq_grid, cmap='plasma')
        axes[2].set_title('Intrinsic Frequencies')
        plt.colorbar(im3, ax=axes[2], label='Frequency')
        
        plt.tight_layout()
        return fig

    def plot_synchronization_evolution(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot evolution of synchronization across training."""
        if not VISUALIZATION_AVAILABLE or not self.state_history:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract time series
        steps = [record['step'] for record in self.state_history]
        n_blocks = len(self.state_history[0]['sync_metrics'])
        
        # Plot synchronization by block
        for block_idx in range(n_blocks):
            sync_values = []
            for record in self.state_history:
                block_key = f'block_{block_idx}'
                if block_key in record['sync_metrics'] and record['sync_metrics'][block_key]:
                    sync_values.append(record['sync_metrics'][block_key][-1])
                else:
                    sync_values.append(0.0)
            
            axes[0, 0].plot(steps, sync_values, label=f'Block {block_idx}', alpha=0.8)
        
        axes[0, 0].set_title('Synchronization Order Parameter')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Order Parameter')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # Plot phase distributions over time
        if len(self.state_history) > 1:
            latest_phases = self.state_history[-1]['lattice_states'][0]['phases']
            axes[0, 1].hist(latest_phases, bins=20, alpha=0.7, density=True)
            axes[0, 1].set_title('Current Phase Distribution')
            axes[0, 1].set_xlabel('Phase (rad)')
            axes[0, 1].set_ylabel('Density')
        
        # Plot coupling evolution for first block
        if self.state_history:
            first_block_states = [record['lattice_states'][0] for record in self.state_history]
            n_heads = len(first_block_states[0]['couplings'])
            
            for head_idx in range(min(4, n_heads)):  # Show first 4 heads
                couplings = [state['couplings'][head_idx] for state in first_block_states]
                axes[1, 0].plot(steps, couplings, label=f'Head {head_idx}', alpha=0.7)
            
            axes[1, 0].set_title('Coupling Strength Evolution')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Coupling Strength')
            axes[1, 0].legend()
        
        # Plot loss history if available
        if self.loss_history and len(self.loss_history) > 1:
            loss_steps = [entry['step'] for entry in self.loss_history]
            losses     = [entry['total_loss'] for entry in self.loss_history]
            axes[1, 1].plot(loss_steps, losses, 'b-', linewidth=2)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        return fig

    def create_phase_animation(self, block_idx: int = 0, interval: int = 200):
        """Create animation of phase evolution over training."""
        if not VISUALIZATION_AVAILABLE or len(self.state_history) < 3:
            print("Need at least 3 recorded states for animation")
            return None
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate(frame_idx):
            ax.clear()
            
            if frame_idx < len(self.state_history):
                state = self.state_history[frame_idx]['lattice_states'][block_idx]
                phases = state['phases']
                positions = state['positions']
                shape = state['lattice_shape']
                step = self.state_history[frame_idx]['step']
                
                # Create phase grid
                phase_grid = np.full(shape, np.nan)
                for idx, (i, j) in enumerate(positions):
                    if idx < len(phases):
                        phase_grid[i, j] = phases[idx]
                
                # Plot with consistent color scale
                im = ax.imshow(phase_grid, cmap='hsv', vmin=0, vmax=2*np.pi)
                
                # Add phase vectors
                for idx, (i, j) in enumerate(positions):
                    if idx < len(phases):
                        phase = phases[idx]
                        dx, dy = 0.4 * np.cos(phase), 0.4 * np.sin(phase)
                        ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.15,
                                fc='white', ec='white', alpha=0.9, linewidth=2)
                
                ax.set_title(f'Lattice Phase Dynamics - Step {step}')
                ax.set_xlabel('Lattice Position J')
                ax.set_ylabel('Lattice Position I')
        
        anim = FuncAnimation(fig, animate, frames=len(self.state_history),
                           interval=interval, repeat=True)
        return anim

    # used in research to highlight delimiter effects
    # plots phases of all heads over time with delimiter step marked
    def plot_delimiter_wave(self, step_delim, window=10, block_idx=0):
        if not self.state_history:
            print("No states recorded; call record_state() during training.")
            return None
        states = [s for s in self.state_history
                if abs(s['step'] - step_delim) <= window]
        if not states:                       # fall back to nearest
            states = [min(self.state_history,
                        key=lambda s: abs(s['step'] - step_delim))]
        phases = np.stack([s['lattice_states'][block_idx]['phases'] for s in states])
        plt.imshow(phases, aspect='auto', cmap='hsv', vmin=0, vmax=2*np.pi)
        plt.axhline(10, color='white', lw=2)   # delimiter row
        plt.title('Phase wave around delimiter')
        plt.show()

# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
class LatticeTrainer:
    def __init__(self, model: LatticeTransformer, visualizer: Optional[LatticeVisualizer] = None):
        self.model = model
        self.visualizer = visualizer
        self.training_history: List[Dict[str, float]] = []

    # Loss helpers ------------------------------------------------------------------
    def phase_regularization_loss(self, lambda_sync: float = 0.01, lambda_diversity: float = 0.01) -> Tensor:
        total_sync, total_diversity, n_blocks = 0.0, 0.0, 0
        device = next(self.model.parameters()).device

        for b in self.model.blocks:
            lat = b.lattice_attn
            phases = torch.stack([h.phase for h in lat.heads])
            freqs = torch.stack([h.intrinsic_freq for h in lat.heads])

            complex_phases = torch.exp(1j * phases.squeeze())
            order = torch.abs(torch.mean(complex_phases))
            sync_loss = 1.0 - order

            freq_var = torch.var(freqs)
            diversity_loss = torch.exp(-freq_var)

            total_sync += sync_loss
            total_diversity += diversity_loss
            n_blocks += 1

        if n_blocks == 0:
            return torch.tensor(0.0, device=device)
        total_sync /= n_blocks
        total_diversity /= n_blocks
        return lambda_sync * total_sync + lambda_diversity * total_diversity

    def spatial_coherence_loss(self, lambda_coherence: float = 0.005) -> Tensor:
        loss = 0.0
        for b in self.model.blocks:
            lat = b.lattice_attn
            for i, hi in enumerate(lat.heads):
                pos_i = lat.positions[i]
                nbr_idx = lat.neighbor_idx[i]
                nbr_w = lat.neighbor_w[i]
                valid = nbr_idx >= 0
                if valid.any():
                    my_c = hi.coupling_strength
                    for j_idx in nbr_idx[valid]:
                        hj = lat.heads[j_idx]
                        loss += (my_c - hj.coupling_strength) ** 2
        return lambda_coherence * loss

    # Training step -----------------------------------------------------------------
    def train_step(
        self,
        input_ids: Tensor,
        labels: Tensor,
        optimizer: torch.optim.Optimizer,
        lambda_sync: float = 0.01,
        lambda_diversity: float = 0.01,
        lambda_coherence: float = 0.005,
        use_lattice_dynamics: bool = True,
    ) -> Dict[str, float]:
        optimizer.zero_grad()

        _, ce_loss = self.model(input_ids, labels=labels, use_lattice_dynamics=use_lattice_dynamics) 
        phase_reg = self.phase_regularization_loss(lambda_sync, lambda_diversity)
        spatial = self.spatial_coherence_loss(lambda_coherence)
        total = ce_loss + phase_reg + spatial

        # Ablation measurement
        shift_labels = labels[..., 1:].contiguous()
        with torch.no_grad():
            logits_ablate, _ = self.model(input_ids, labels=labels, use_lattice_dynamics=False)
            ce_ablate = F.cross_entropy(
                logits_ablate[..., :-1, :].contiguous().view(-1, logits_ablate.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        metrics = {
            "total_loss": total.item(),
            "ce_loss": ce_loss.item(),
            "phase_reg_loss": phase_reg.item(),
            "spatial_coherence_loss": spatial.item(),
            "ce_ablate": ce_ablate.item(),
            "step": len(self.training_history),
        }

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        self.training_history.append(metrics)
        if self.visualizer:
            self.visualizer.record_state(len(self.training_history))
            self.visualizer.loss_history.append(metrics)
        return metrics

    # Epoch -------------------------------------------------------------------------
    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        lambda_sync: float = 0.01,
        lambda_diversity: float = 0.01,
        lambda_coherence: float = 0.005,
        use_lattice_dynamics: bool = True,
        verbose: bool = True,
    ) -> Dict[str, float]:
        self.model.train()
        totals = {k: 0.0 for k in ("total_loss", "ce_loss", "phase_reg_loss", "spatial_coherence_loss")}
        n_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            device = next(self.model.parameters()).device
            input_ids, labels = input_ids.to(device), labels.to(device)
            m = self.train_step(
                input_ids, labels, optimizer,
                lambda_sync, lambda_diversity, lambda_coherence,
                use_lattice_dynamics,
            )
            for k in totals:
                totals[k] += m[k]
            n_batches += 1
            if verbose and batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: total={m['total_loss']:.4f} "
                      f"ce={m['ce_loss']:.4f} lattice_gain={m['ce_loss'] - m['ce_ablate']:.4f}")

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # Adaptive schedule -------------------------------------------------------------
    def adaptive_training_schedule(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int = 10,
        warmup_epochs: int = 2,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        results = []
        total_steps = len(dataloader) * n_epochs

        for epoch in range(n_epochs):
            if verbose:
                print(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")

            current_step = epoch * len(dataloader) + 1
            progress = current_step / total_steps

            if progress < 0.2:  # silent phase
                lam_sync = lam_div = lam_coh = 0.0
            elif progress < 0.8:  # ramp
                lam_sync = lam_div = 0.05 * (progress - 0.2) / 0.6
                lam_coh = 0.005 * lam_sync
            else:  # plateau
                lam_sync = lam_div = 0.05
                lam_coh = 0.005

            metrics = self.train_epoch(
                dataloader, optimizer,
                lam_sync, lam_div, lam_coh,
                use_lattice_dynamics=True,
                verbose=verbose,
            )
            results.append(metrics)

            if verbose:
                print(f"  Epoch metrics: total={metrics['total_loss']:.6f} "
                      f"ce={metrics['ce_loss']:.6f} phase={metrics['phase_reg_loss']:.6f} "
                      f"spatial={metrics['spatial_coherence_loss']:.6f}")
                if self.visualizer and self.visualizer.state_history:
                    latest_sync = self.model.get_synchronization_metrics()
                    if latest_sync:
                        avg_sync = np.mean([v[-1] if v else 0.0 for v in latest_sync.values()])
                        print(f"  Avg sync order: {avg_sync:.4f}")
        return results

    # Plotting ----------------------------------------------------------------------
    def plot_training_progress(self, figsize: Tuple[int, int] = (15, 10)):
        """Comprehensive training progress visualization."""
        if not VISUALIZATION_AVAILABLE or not self.training_history:
            return None
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        steps = [h['step'] for h in self.training_history]
        
        # Loss components
        total_losses = [h['total_loss'] for h in self.training_history]
        ce_losses = [h['ce_loss'] for h in self.training_history]
        phase_losses = [h['phase_reg_loss'] for h in self.training_history]
        spatial_losses = [h['spatial_coherence_loss'] for h in self.training_history]
        
        # Plot loss evolution
        axes[0, 0].plot(steps, total_losses, label='Total Loss', linewidth=2, color='black')
        axes[0, 0].plot(steps, ce_losses, label='CE Loss', alpha=0.8, color='blue')
        axes[0, 0].plot(steps, phase_losses, label='Phase Reg', alpha=0.8, color='red')
        axes[0, 0].plot(steps, spatial_losses, label='Spatial Coherence', alpha=0.8, color='green')
        axes[0, 0].set_title('Training Loss Components')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Loss ratios
        phase_ratios = [h['phase_reg_loss'] / h['total_loss'] for h in self.training_history]
        spatial_ratios = [h['spatial_coherence_loss'] / h['total_loss'] for h in self.training_history]
        
        axes[0, 1].plot(steps, phase_ratios, label='Phase Reg Ratio', color='red')
        axes[0, 1].plot(steps, spatial_ratios, label='Spatial Coherence Ratio', color='green')
        axes[0, 1].set_title('Regularization Contribution')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Ratio of Total Loss')
        axes[0, 1].legend()
        
        # Synchronization evolution
        if self.visualizer and self.visualizer.state_history:
            sync_data = []
            sync_steps = []
            
            for record in self.visualizer.state_history:
                sync_metrics = record['sync_metrics']
                avg_sync = np.mean([
                    sync_list[-1] if sync_list else 0.0 
                    for sync_list in sync_metrics.values()
                ])
                sync_data.append(avg_sync)
                sync_steps.append(record['step'])
            
            axes[0, 2].plot(sync_steps, sync_data, 'purple', linewidth=2)
            axes[0, 2].set_title('Average Synchronization')
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('Order Parameter')
            axes[0, 2].set_ylim(0, 1)
        
        # Loss smoothing (moving average)
        window = min(50, len(steps) // 10) if len(steps) > 20 else 5
        if len(total_losses) >= window:
            smoothed_loss = np.convolve(total_losses, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            axes[1, 0].plot(smooth_steps, smoothed_loss, 'black', linewidth=2)
            axes[1, 0].set_title(f'Smoothed Total Loss (window={window})')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_yscale('log')
        
        # Gradient of loss (learning rate effectiveness)
        if len(total_losses) > 1:
            loss_grad = np.gradient(total_losses)
            axes[1, 1].plot(steps, loss_grad, 'orange', alpha=0.7)
            axes[1, 1].set_title('Loss Gradient (Learning Progress)')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss Gradient')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Training efficiency (loss reduction per step)
        if len(total_losses) > 10:
            initial_loss = np.mean(total_losses[:5])
            efficiency = [(initial_loss - loss) / initial_loss for loss in total_losses]
            axes[1, 2].plot(steps, efficiency, 'teal', linewidth=2)
            axes[1, 2].set_title('Training Efficiency')
            axes[1, 2].set_xlabel('Training Step')
            axes[1, 2].set_ylabel('Relative Loss Reduction')
            axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig

# --------------------------------------------------------------------------- #
#  Dataset helpers
# --------------------------------------------------------------------------- #
class SimpleSequenceDataset:
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 1000, pattern_type: str = "periodic"):
        self.seq_len, self.vocab_size, self.pattern_type = seq_len, vocab_size, pattern_type
        self.data = []
        for _ in range(n_samples):
            if pattern_type == "periodic":
                seq = torch.randint(1, vocab_size - 1, (seq_len,))
                seq[2::3] = seq[0]
                seq[4::5] = seq[1]
            elif pattern_type == "hierarchical":
                seq = torch.randint(1, vocab_size - 1, (seq_len,))
                for i in range(seq_len // 4):
                    seq[i * 4 : (i + 1) * 4] = torch.roll(seq[i * 4 : (i + 1) * 4], 1)
            elif pattern_type == "random":
                seq = torch.randint(1, vocab_size - 1, (seq_len,))
            else:
                raise ValueError(f"Unknown pattern_type: {pattern_type}")

            tgt = torch.cat([seq[1:], torch.tensor([0])])
            tgt[tgt == 0] = -100
            self.data.append((seq, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_dataloader(dataset, batch_size: int = 32):
    def collate(batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)

    batches = []
    for i in range(0, len(dataset), batch_size):
        chunk = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batches.append(collate(chunk))
    return batches

# --------------------------------------------------------------------------- #
#  Copy-reverse exit-criterion dataset
# --------------------------------------------------------------------------- #
class CopyReverseDataset:
    def __init__(self, vocab_size=1000, seq_len=128, n_samples=5000):
        self.seq_len = seq_len
        self.data = []
        half = seq_len // 2
        for _ in range(n_samples):
            prefix = torch.randint(3, vocab_size, (half,))
            suffix = prefix.flip(0)
            seq = torch.cat([prefix, torch.tensor([2]), suffix])  # 2 = delimiter
            tgt = torch.cat([seq[1:], torch.tensor([0])])
            self.data.append((seq, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --------------------------------------------------------------------------- #
#  Comprehensive demo
# --------------------------------------------------------------------------- #
def run_comprehensive_demo():
    set_seed(42)
    print("=" * 80)
    print("Lattice Oscillator Transformer")
    print("=" * 80)

    config = dict(
        vocab_size=1000,
        d_model=132,
        n_heads=6,
        n_layers=4,
        d_ff=512,
        max_seq_len=64,
        lattice_shape=(2, 3),
    )

    print("\nModel config:", config)
    model = LatticeTransformer(**config)
    model.enable_research_mode()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model built - {total_params:,} parameters")
    print(f"✓ Research mode enabled for comprehensive monitoring")

    visualizer = LatticeVisualizer(model)
    trainer = LatticeTrainer(model, visualizer)
    print(f"✓ Visualization and training tools initialized")

    # Create datasets with different patterns
    print(f"\n2. Creating Test Datasets...")
    datasets = {
        name: SimpleSequenceDataset(config["vocab_size"], 32, 200, name)
        for name in ("periodic", "hierarchical", "random")
    }
    for name, dataset in datasets.items():
        print(f"✓ {name.capitalize()} dataset: {len(dataset)} samples")

    dataloader = create_dataloader(datasets["periodic"], batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

     # Comparative training test
    print(f"\n3. Running Comparative Training Test...")

    print("\nQuick adaptive training (3 epochs on 10 mini-batches; should promote synchronization)...")
    epoch_results = trainer.adaptive_training_schedule(
        dataloader[:10], optimizer, n_epochs=3, warmup_epochs=1, verbose=True
    )
    print(f"\n✓ Training complete – {len(epoch_results)} epochs recorded")
    print(f"✓ Lattice states and metrics recorded at each training step")
    print(f"✓ Training history contains {len(trainer.training_history)} entries")
    print(f"✓ Visualizer has {len(visualizer.state_history)} recorded states")
    print(f"✓ Visualizer has {len(visualizer.loss_history)} recorded loss entries")
    assert len(visualizer.state_history) == len(trainer.training_history), "State history length mismatch"
    assert len(visualizer.loss_history) == len(trainer.training_history), "Loss history length mismatch"
    assert all('step' in entry for entry in trainer.training_history), "Training history missing 'step' key"
    assert all('step' in entry for entry in visualizer.loss_history), "Loss history missing 'step' key"
    assert all('lattice_states' in state for state in visualizer.state_history), "State history missing 'lattice_states' key"
    assert all('sync_metrics' in state for state in visualizer.state_history), "State history missing 'sync_metrics' key"
    print(f"✓ All data integrity checks passed")

    print(f"\n4. Analyzing Results...")
    
    print("\nFinal lattice snapshot:")
    for idx, state in enumerate(model.get_lattice_states()):
        print(f"  Block {idx}: phase-std={state['phases'].std():.3f} "
              f"sync-order={np.abs(np.mean(np.exp(1j * state['phases']))):.3f} "
              f"coupling-mean={state['couplings'].mean():.3f}")
    
    # Synchronization measure
    print(f"\nLattice Synchronization Metrics:")
    for block_idx, state in enumerate(model.get_lattice_states()):
        phases = state['phases']
        complex_phases = np.exp(1j * phases)
        sync_order = np.abs(np.mean(complex_phases))
        print(f"  Block {block_idx}: sync-order={sync_order:.3f} phase-std={phases.std():.3f}") 

    # Performance comparison
    print(f"\n5. Performance Analysis...")
    
    # Test both modes
    sample_input = torch.randint(0, config['vocab_size'], (4, 32))
    
    # With lattice dynamics
    start_time = time.time()
    with torch.no_grad():
        output_lattice = model(sample_input, use_lattice_dynamics=True)
    lattice_time = time.time() - start_time
    
    # Without lattice dynamics (SDPA fallback)
    start_time = time.time()
    with torch.no_grad():
        output_standard = model(sample_input, use_lattice_dynamics=False)
    standard_time = time.time() - start_time
    
    print(f"  Lattice dynamics: {lattice_time*1000:.2f}ms")
    print(f"  Standard attention: {standard_time*1000:.2f}ms")
    print(f"  Overhead: {(lattice_time/standard_time - 1)*100:.1f}%")
    
    # Visualization demonstration
    if VISUALIZATION_AVAILABLE:
        print(f"\n6. Generating Visualizations...")
        try:

            # Lattice state snapshot
            fig1 = visualizer.plot_lattice_snapshot()
            if fig1:
                f1 = plt.figure(fig1.number)
                print(f"✓ Lattice state visualization created")

            # Synchronization evolution
            fig2 = visualizer.plot_synchronization_evolution()
            if fig2:
                f2 = plt.figure(fig2.number)
                print(f"✓ Synchronization evolution plot created")
            
            # Training progress
            fig3 = trainer.plot_training_progress()
            if fig3:
                f3 = plt.figure(fig3.number)
                print(f"✓ Training progress visualization created")

            # Phase animation
            anim = visualizer.create_phase_animation(block_idx=0, interval=300)
            if anim:
                anim.save('phase_animation.gif')
                print(f"✓ Phase animation created") 

            plt.show()
                    
        except Exception as e:
            print(f"Visualization error: {e}")
    else:
        print(f"\n6. Visualizations unavailable (install matplotlib)")


    print("\n" + "=" * 80)
    print("Demo complete.")
    print("=" * 80)
    return model, visualizer, trainer

# --------------------------------------------------------------------------- #
#  When the white line shows a clear π-phase slip and test exact-match ≥ 98%,
#  the lattice is doing useful work. In other words, the phase wave is
#  successfully propagating the delimiter information to the output. An "exact
#  match" means the entire output sequence is correct. An exact match of 98%
#  means that 98% of all sequences in the test set were copied/reversed
#  perfectly. A 'π-phase slip" means that the phases of the oscillators jump by
#  π (i.e. flip direction) at the delimiter position, indicating that the
#  delimiter has been detected and is influencing the lattice dynamics.
# --------------------------------------------------------------------------- #
def run_phase_wave_test():
    print("Running phase wave test...")
    set_seed(42)
    model = LatticeTransformer(vocab_size=1000, d_model=132, n_heads=6, n_layers=4, d_ff=512)
    model.enable_research_mode()
    viz  = LatticeVisualizer(model)
    trainer= LatticeTrainer(model, viz)
    data = create_dataloader(CopyReverseDataset(), batch_size=32)
    opt  = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer.adaptive_training_schedule(data, opt, n_epochs=10)
    viz.plot_delimiter_wave(step_delim=100)

# --------------------------------------------------------------------------- #
#  Quick sanity check
# --------------------------------------------------------------------------- #
def run_sanity_check():
    print("Running sanity check...")
    set_seed(42)
    model = LatticeTransformer(vocab_size=1000, d_model=132, n_heads=6, n_layers=4, d_ff=512)
    x = torch.randint(0, 100, (4, 20))
    logits_l, loss_l = model(x, labels=x, use_lattice_dynamics=True)
    logits_s, loss_s = model(x, labels=x, use_lattice_dynamics=False)
    print(f"✓ Lattice loss: {loss_l.item():.4f}  Standard loss: {loss_s.item():.4f}")
    
# Uncomment for quick sanity check:
# run_sanity_check()

# Uncomment for phase wave test:
# run_phase_wave_test()

# Uncomment for full demo (may take several minutes):
# run_comprehensive_demo()
