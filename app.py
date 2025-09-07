# Enhanced Lattic Oscillator Transformer
# Combines production-ready architecture with comprehensive research tooling
# Req: Python ≥ 3.8, PyTorch ≥ 2.0, matplotlib, seaborn

import math
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

# Optional visualization imports (graceful fallback if not available)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.animation import FuncAnimation
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Visualization tools unavailable. Install matplotlib and seaborn for full functionality.")

# --------------------------------------------------------------------------- #
#  Core Helper Functions
# --------------------------------------------------------------------------- #
def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Return lower-triangular boolean mask (1 = keep, 0 = masked)."""
    return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

def _check_divisible(d_model: int, n_heads: int) -> None:
    if d_model % n_heads:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

# --------------------------------------------------------------------------- #
#  Enhanced Lattice Attention Head with Monitoring
# --------------------------------------------------------------------------- #
class LatticeAttentionHead(nn.Module):
    """Single attention head with Kuramoto phase oscillator and monitoring capabilities."""

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
        self.d_k = d_k
        self.lattice_pos = lattice_pos
        self.head_id = head_id

        # Standard QKV
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

        # Phase oscillator state (buffers for autograd safety)
        self.register_buffer("phase", torch.tensor(0.0))
        self.register_buffer("intrinsic_freq", torch.tensor(float(intrinsic_freq)))
        self.coupling_strength = nn.Parameter(torch.tensor(coupling_strength))
        self.lattice_constant = nn.Parameter(torch.tensor(1.0))

        # Efficiency pre-computation
        self.register_buffer("scale", torch.tensor(math.sqrt(float(d_k))))
        
        # Research monitoring (optional tracking)
        self._phase_history = []
        self._coupling_history = []
        self._attention_entropy_history = []
        self.monitor_dynamics = False

    def enable_monitoring(self):
        """Enable detailed dynamics monitoring for research."""
        self.monitor_dynamics = True
        self._phase_history.clear()
        self._coupling_history.clear()
        self._attention_entropy_history.clear()

    def disable_monitoring(self):
        """Disable monitoring for production use."""
        self.monitor_dynamics = False

    def get_dynamics_history(self) -> Dict[str, List[float]]:
        """Return recorded dynamics history."""
        return {
            'phases': self._phase_history.copy(),
            'couplings': self._coupling_history.copy(),
            'attention_entropy': self._attention_entropy_history.copy()
        }

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        neighbour_phases: Tensor,
        neighbour_weights: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (output, attn_weights)."""
        B, L, _ = query.shape

        Q = self.W_q(query)  # (B, L, d_k)
        K = self.W_k(key)    # (B, L, d_k)
        V = self.W_v(value)  # (B, L, d_k)

        # Ensure Q, K, V are 3D - flatten any extra dimensions
        if Q.dim() > 3:
            Q = Q.view(B, L, -1)
        if K.dim() > 3:
            K = K.view(B, L, -1)
        if V.dim() > 3:
            V = V.view(B, L, -1)

        # Standard scaled dot-product - use matmul instead of einsum for robustness
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, L, L)

        # Kuramoto phase update
        if neighbour_phases.numel():
            phase_diff = neighbour_phases - self.phase
            coupling_force = (neighbour_weights * torch.sin(phase_diff)).sum()
            new_phase = self.phase + 0.01 * (self.intrinsic_freq + self.coupling_strength * coupling_force)
            
            # Record coupling strength for monitoring
            if self.monitor_dynamics:
                self._coupling_history.append(coupling_force.item())
        else:
            new_phase = self.phase + 0.01 * self.intrinsic_freq
            if self.monitor_dynamics:
                self._coupling_history.append(0.0)

        # Phase wrapping to prevent drift
        self.phase = new_phase.remainder(2 * math.pi)

        # Record phase for monitoring
        if self.monitor_dynamics:
            self._phase_history.append(self.phase.item())

        # Add phase-dependent bias to attention scores
        bias = 0.1 * torch.cos(self.phase)
        scores = scores + bias

        # Apply causal mask if provided
        if mask is not None:
            # Handle different mask dimensions
            if mask.dim() == 2:  # (L, L)
                mask = mask.unsqueeze(0).expand(B, -1, -1)
            elif mask.dim() == 3 and mask.size(0) == 1:  # (1, L, L)
                mask = mask.expand(B, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e4)

        # Compute attention weights
        attn = F.softmax(scores, dim=-1)  # (B, L, L)
        
        # Monitor attention entropy (measure of attention spread)
        if self.monitor_dynamics:
            with torch.no_grad():
                entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean().item()
                self._attention_entropy_history.append(entropy)

        # Apply attention to values - use matmul instead of einsum
        out = torch.matmul(attn, V)  # (B, L, d_k)
        return out, attn.detach()

# --------------------------------------------------------------------------- #
#  Enhanced Multi-Head Attention with Research Features
# --------------------------------------------------------------------------- #
class LatticeMultiHeadAttention(nn.Module):
    """Multi-head attention with lattice organization and research monitoring."""

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

        # Build optimal 2D lattice
        if lattice_shape is None:
            side = int(math.sqrt(n_heads))
            lattice_shape = (side, side + 1) if side * side < n_heads else (side, side)
        self.lattice_shape: Tuple[int, int] = lattice_shape

        # Create lattice-organized heads
        self.heads = nn.ModuleList()
        self.positions: List[Tuple[int, int]] = []
        for idx in range(n_heads):
            row, col = divmod(idx, lattice_shape[1])
            pos = (row, col)
            self.positions.append(pos)
            freq = 1.0 + 0.1 * idx  # Frequency diversity
            head = LatticeAttentionHead(d_model, self.d_k, pos, intrinsic_freq=freq, head_id=idx)
            self.heads.append(head)

        # Pre-compute static neighbor topology
        self._build_neighbours()
        self.out_proj = nn.Linear(d_model, d_model)

        # Research monitoring state
        self.monitor_lattice = False
        self._global_sync_history = []
        self._attention_maps_history = []

    def enable_monitoring(self):
        """Enable comprehensive lattice monitoring."""
        self.monitor_lattice = True
        self._global_sync_history.clear()
        self._attention_maps_history.clear()
        for head in self.heads:
            head.enable_monitoring()

    def disable_monitoring(self):
        """Disable monitoring for production."""
        self.monitor_lattice = False
        for head in self.heads:
            head.disable_monitoring()

    def _build_neighbours(self) -> None:
        """Pre-compute neighbor topology for efficiency."""
        n = len(self.heads)
        self.register_buffer("neighbour_idx", torch.full((n, 8), -1, dtype=torch.long))
        self.register_buffer("neighbour_w", torch.zeros((n, 8)))

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
                            self.neighbour_idx[i, k] = j
                            self.neighbour_w[i, k] = math.exp(-dist)
                            k += 1
                            break

    def get_lattice_state(self) -> Dict[str, np.ndarray]:
        """Extract current lattice state for analysis."""
        phases = np.array([head.phase.item() for head in self.heads])
        couplings = np.array([head.coupling_strength.item() for head in self.heads])
        frequencies = np.array([head.intrinsic_freq.item() for head in self.heads])
        positions = np.array(self.positions)
        
        return {
            'phases': phases,
            'couplings': couplings,
            'frequencies': frequencies,
            'positions': positions,
            'lattice_shape': self.lattice_shape
        }

    def compute_synchronization_order(self) -> float:
        """Compute Kuramoto order parameter (global synchronization measure)."""
        phases = torch.stack([head.phase for head in self.heads])
        complex_phases = torch.exp(1j * phases)
        order_param = torch.abs(torch.mean(complex_phases)).item()
        
        if self.monitor_lattice:
            self._global_sync_history.append(order_param)
        
        return order_param

    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        mask: Optional[Tensor] = None,
        use_lattice_dynamics: bool = True
    ) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass with optional lattice dynamics."""
        B, L, _ = query.shape
        
        # Ensure inputs are properly shaped
        if query.dim() > 3:
            query = query.view(B, L, -1)
        if key.dim() > 3:
            key = key.view(B, L, -1)
        if value.dim() > 3:
            value = value.view(B, L, -1)

        # Prepare mask
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, L, L)

        # Fast path: use optimized SDPA when lattice dynamics disabled
        if not use_lattice_dynamics and self.use_sdpa and mask is None:
            # Reshape for multi-head attention: (B, n_heads, L, d_k)
            q = torch.empty(B, self.n_heads, L, self.d_k, device=query.device, dtype=query.dtype)
            k = torch.empty(B, self.n_heads, L, self.d_k, device=key.device, dtype=key.dtype)
            v = torch.empty(B, self.n_heads, L, self.d_k, device=value.device, dtype=value.dtype)
            
            for h_idx, head in enumerate(self.heads):
                q[:, h_idx, :, :] = head.W_q(query)
                k[:, h_idx, :, :] = head.W_k(key)
                v[:, h_idx, :, :] = head.W_v(value)
            
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
            out = out.transpose(1, 2).reshape(B, L, self.d_model)
            return self.out_proj(out), []

        # Full path with lattice dynamics
        outs, attns = [], []
        for h_idx, head in enumerate(self.heads):
            # Get neighbor information
            idx = self.neighbour_idx[h_idx]
            weights = self.neighbour_w[h_idx]
            valid_mask = idx >= 0
            
            if valid_mask.any():
                neighbor_phases = torch.stack([self.heads[i].phase for i in idx[valid_mask]])
                neighbor_weights = weights[valid_mask]
            else:
                neighbor_phases = torch.tensor([])
                neighbor_weights = torch.tensor([])
            
            # Process through head with lattice coupling
            output, attn = head(query, key, value, neighbor_phases, neighbor_weights, mask)
            outs.append(output)
            attns.append(attn)

        # Record attention maps for analysis
        if self.monitor_lattice and len(attns) > 0:
            stacked_attns = torch.stack(attns, dim=1)  # (B, n_heads, L, L)
            self._attention_maps_history.append(stacked_attns.detach().cpu())

        # Compute global synchronization
        if use_lattice_dynamics:
            self.compute_synchronization_order()

        multi_out = torch.cat(outs, dim=-1)
        return self.out_proj(multi_out), attns

# --------------------------------------------------------------------------- #
#  Enhanced Transformer Block
# --------------------------------------------------------------------------- #
class LatticeTransformerBlock(nn.Module):
    """Transformer block with enhanced lattice attention and monitoring."""

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
#  Enhanced Lattice Transformer with Research Capabilities
# --------------------------------------------------------------------------- #
class LatticeTransformer(nn.Module):
    """
    Production-ready Lattice Transformer with comprehensive research tooling.
    
    Features:
    - Efficient lattice dynamics with SDPA fallback
    - Comprehensive monitoring and visualization
    - Specialized training procedures
    - Production-ready architecture
    """

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

        # Core architecture
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            LatticeTransformerBlock(d_model, n_heads, d_ff, lattice_shape)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Weight tying for parameter efficiency
        self.lm_head.weight = self.token_emb.weight
        self.register_buffer("pos_buf", torch.arange(max_seq_len))

        # Research state
        self.research_mode = False
        self._training_step = 0

    def enable_research_mode(self):
        """Enable comprehensive research monitoring across all components."""
        self.research_mode = True
        for block in self.blocks:
            block.lattice_attn.enable_monitoring()

    def disable_research_mode(self):
        """Disable research monitoring for production use."""
        self.research_mode = False
        for block in self.blocks:
            block.lattice_attn.disable_monitoring()

    def get_lattice_states(self) -> List[Dict[str, np.ndarray]]:
        """Get lattice states from all blocks."""
        return [block.lattice_attn.get_lattice_state() for block in self.blocks]

    def get_synchronization_metrics(self) -> Dict[str, List[float]]:
        """Get synchronization metrics across all blocks."""
        return {
            f'block_{i}': block.lattice_attn._global_sync_history.copy()
            for i, block in enumerate(self.blocks)
        }

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        causal: bool = True,
        return_dict: bool = False,
        use_lattice_dynamics: bool = True,
    ):
        """
        Forward pass with optional lattice dynamics.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels for training
            causal: Whether to use causal masking
            return_dict: Whether to return dictionary
            use_lattice_dynamics: Whether to use lattice coupling (vs standard attention)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings with scaling
        x = self.token_emb(input_ids) + self.pos_emb(self.pos_buf[:L])
        x *= math.sqrt(self.d_model)

        # Causal masking for autoregressive modeling
        mask = _make_causal_mask(L, device) if causal else None

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask, use_lattice=use_lattice_dynamics)

        # Final output projection
        logits = self.lm_head(self.norm(x))

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id,
            )

        if return_dict:
            return {"logits": logits, "loss": loss}
        return logits if loss is None else (logits, loss)

# --------------------------------------------------------------------------- #
#  Research Visualization Tools
# --------------------------------------------------------------------------- #
class LatticeVisualizer:
    """Comprehensive visualization toolkit for lattice dynamics research."""
    
    def __init__(self, model: LatticeTransformer):
        self.model = model
        self.state_history = []
        self.sync_history = []
        self.loss_history = []
        
        if not VISUALIZATION_AVAILABLE:
            print("Warning: Visualization not available. Install matplotlib and seaborn.")

    def record_state(self, training_step: int = None):
        """Record current model state for later analysis."""
        if not self.model.research_mode:
            return
            
        states = self.model.get_lattice_states()
        sync_metrics = self.model.get_synchronization_metrics()
        
        record = {
            'step': training_step or len(self.state_history),
            'timestamp': time.time(),
            'lattice_states': states,
            'sync_metrics': sync_metrics
        }
        
        self.state_history.append(record)

    def plot_lattice_snapshot(self, block_idx: int = 0, figsize: Tuple[int, int] = (15, 5)):
        """Plot current state of lattice for specified block."""
        if not VISUALIZATION_AVAILABLE:
            print("Visualization not available")
            return None
            
        if not self.state_history:
            print("No recorded states. Call record_state() first.")
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
        if self.loss_history:
            loss_steps = [entry['step'] for entry in self.loss_history]
            losses = [entry['loss'] for entry in self.loss_history]
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

# --------------------------------------------------------------------------- #
#  Specialized Training Procedures
# --------------------------------------------------------------------------- #
class LatticeTrainer:
    """Advanced training procedures for lattice transformers."""
    
    def __init__(self, model: LatticeTransformer, visualizer: Optional[LatticeVisualizer] = None):
        self.model = model
        self.visualizer = visualizer
        self.training_history = []
        
    def phase_regularization_loss(self, lambda_sync: float = 0.01, lambda_diversity: float = 0.01) -> Tensor:
        """
        Compute phase regularization losses.
        
        Args:
            lambda_sync: Weight for synchronization loss
            lambda_diversity: Weight for diversity loss
        """
        total_sync_loss = 0.0
        total_diversity_loss = 0.0
        n_blocks = 0
        
        for block in self.model.blocks:
            lattice_attn = block.lattice_attn
            
            # Extract phases and frequencies
            phases = torch.stack([head.phase for head in lattice_attn.heads])
            frequencies = torch.stack([head.intrinsic_freq for head in lattice_attn.heads])
            
            # Synchronization loss (Kuramoto order parameter)
            complex_phases = torch.exp(1j * phases.squeeze())
            order_param = torch.abs(torch.mean(complex_phases))
            sync_loss = 1.0 - order_param  # Encourage synchronization
            
            # Diversity loss (encourage frequency diversity)
            freq_var = torch.var(frequencies)
            diversity_loss = torch.exp(-freq_var)  # Encourage diversity
            
            total_sync_loss += sync_loss
            total_diversity_loss += diversity_loss
            n_blocks += 1
        
        total_sync_loss /= n_blocks
        total_diversity_loss /= n_blocks
        
        return lambda_sync * total_sync_loss + lambda_diversity * total_diversity_loss
    
    def spatial_coherence_loss(self, lambda_coherence: float = 0.005) -> Tensor:
        """Encourage spatial coherence in neighboring attention heads."""
        coherence_loss = 0.0
        
        for block in self.model.blocks:
            lattice_attn = block.lattice_attn
            positions = lattice_attn.positions
            
            # Compute spatial coherence between neighboring heads
            for i, head_i in enumerate(lattice_attn.heads):
                pos_i = positions[i]
                
                # Get neighbor indices for this head
                neighbor_indices = lattice_attn.neighbour_idx[i]
                neighbor_weights = lattice_attn.neighbour_w[i]
                valid_neighbors = neighbor_indices >= 0
                
                if valid_neighbors.any():
                    # Encourage similar coupling strengths for neighbors
                    my_coupling = head_i.coupling_strength
                    for j_idx in neighbor_indices[valid_neighbors]:
                        neighbor_head = lattice_attn.heads[j_idx]
                        coupling_diff = (my_coupling - neighbor_head.coupling_strength) ** 2
                        coherence_loss += coupling_diff
        
        return lambda_coherence * coherence_loss
    
    def train_step(
        self, 
        input_ids: Tensor, 
        labels: Tensor, 
        optimizer: torch.optim.Optimizer,
        lambda_sync: float = 0.01,
        lambda_diversity: float = 0.01,
        lambda_coherence: float = 0.005,
        use_lattice_dynamics: bool = True
    ) -> Dict[str, float]:
        """Single training step with lattice-specific regularization."""
        optimizer.zero_grad()
        
        # Forward pass
        logits, ce_loss = self.model(input_ids, labels=labels, use_lattice_dynamics=use_lattice_dynamics)
        
        # Lattice regularization losses
        phase_reg_loss = self.phase_regularization_loss(lambda_sync, lambda_diversity)
        spatial_coherence_loss = self.spatial_coherence_loss(lambda_coherence)
        
        # Total loss
        total_loss = ce_loss + phase_reg_loss + spatial_coherence_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Record metrics
        metrics = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'phase_reg_loss': phase_reg_loss.item(),
            'spatial_coherence_loss': spatial_coherence_loss.item(),
            'step': len(self.training_history)
        }
        
        self.training_history.append(metrics)
        
        # Record state for visualization
        if self.visualizer:
            self.visualizer.record_state(len(self.training_history))
            self.visualizer.loss_history.append(metrics)
        
        return metrics
    
    def train_epoch(
        self, 
        dataloader, 
        optimizer: torch.optim.Optimizer,
        lambda_sync: float = 0.01,
        lambda_diversity: float = 0.01, 
        lambda_coherence: float = 0.005,
        use_lattice_dynamics: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch with lattice regularization."""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'ce_loss': 0.0, 
            'phase_reg_loss': 0.0,
            'spatial_coherence_loss': 0.0
        }
        
        n_batches = 0
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            # Move to device if needed
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Training step
            metrics = self.train_step(
                input_ids, labels, optimizer,
                lambda_sync, lambda_diversity, lambda_coherence,
                use_lattice_dynamics
            )
            
            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            n_batches += 1
            
            # Periodic logging
            if verbose and batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {metrics['total_loss']:.4f} "
                      f"(CE: {metrics['ce_loss']:.4f}, "
                      f"Phase: {metrics['phase_reg_loss']:.4f}, "
                      f"Spatial: {metrics['spatial_coherence_loss']:.4f})")
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics
    
    def adaptive_training_schedule(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int = 10,
        warmup_epochs: int = 2,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Adaptive training with evolving lattice regularization.
        
        - Warmup: Focus on standard loss, minimal lattice regularization
        - Main training: Gradually increase lattice effects
        - Fine-tuning: Balance all loss components
        """
        epoch_results = []
        
        for epoch in range(n_epochs):
            if verbose:
                print(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")
            
            # Adaptive regularization schedule
            if epoch < warmup_epochs:
                # Warmup phase: minimal lattice regularization
                lambda_sync = 0.001
                lambda_diversity = 0.001
                lambda_coherence = 0.0001
                if verbose:
                    print("Phase: Warmup (minimal lattice regularization)")
            elif epoch < n_epochs - 2:
                # Main training: progressive lattice regularization
                progress = (epoch - warmup_epochs) / (n_epochs - warmup_epochs - 2)
                lambda_sync = 0.001 + progress * 0.009  # 0.001 -> 0.01
                lambda_diversity = 0.001 + progress * 0.009
                lambda_coherence = 0.0001 + progress * 0.0049  # 0.0001 -> 0.005
                if verbose:
                    print(f"Phase: Progressive training (λ_sync={lambda_sync:.4f})")
            else:
                # Fine-tuning: full regularization
                lambda_sync = 0.01
                lambda_diversity = 0.01
                lambda_coherence = 0.005
                if verbose:
                    print("Phase: Fine-tuning (full regularization)")
            
            # Train epoch
            epoch_metrics = self.train_epoch(
                dataloader, optimizer,
                lambda_sync, lambda_diversity, lambda_coherence,
                use_lattice_dynamics=True,
                verbose=verbose
            )
            
            epoch_results.append(epoch_metrics)
            
            if verbose:
                print(f"Epoch {epoch + 1} Results:")
                print(f"  Total Loss: {epoch_metrics['total_loss']:.6f}")
                print(f"  CE Loss: {epoch_metrics['ce_loss']:.6f}")
                print(f"  Phase Reg: {epoch_metrics['phase_reg_loss']:.6f}")
                print(f"  Spatial Coherence: {epoch_metrics['spatial_coherence_loss']:.6f}")
                
                # Show synchronization metrics if available
                if self.visualizer and self.visualizer.state_history:
                    latest_sync = self.model.get_synchronization_metrics()
                    if latest_sync:
                        avg_sync = np.mean([
                            sync_list[-1] if sync_list else 0.0
                            for sync_list in latest_sync.values()
                        ])
                        print(f"  Avg Synchronization: {avg_sync:.4f}")
        
        return epoch_results

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
#  Enhanced Dataset and Training Utilities
# --------------------------------------------------------------------------- #
class SimpleSequenceDataset:
    """Enhanced dataset with configurable patterns for testing lattice dynamics."""
    
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 1000, pattern_type: str = 'periodic'):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pattern_type = pattern_type
        self.data = []
        
        for _ in range(n_samples):
            if pattern_type == 'periodic':
                # Periodic pattern (good for testing synchronization)
                seq = torch.randint(1, vocab_size - 1, (seq_len,))
                seq[2::3] = seq[0]  # Every 3rd token matches first
                seq[4::5] = seq[1]  # Every 5th token matches second
            elif pattern_type == 'hierarchical':
                # Hierarchical pattern (good for testing spatial coherence)
                seq = torch.randint(1, vocab_size - 1, (seq_len,))
                # Create nested structure
                for i in range(seq_len // 4):
                    seq[i*4:(i+1)*4] = torch.roll(seq[i*4:(i+1)*4], 1)
            elif pattern_type == 'random':
                # Pure random (baseline)
                seq = torch.randint(1, vocab_size - 1, (seq_len,))
            else:
                raise ValueError(f"Unknown pattern_type: {pattern_type}")
            
            # Target is next token prediction
            target = torch.cat([seq[1:], torch.tensor([0])])
            self.data.append((seq, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_dataloader(dataset, batch_size: int = 32):
    """Create simple dataloader with proper collation."""
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)
    
    batches = []
    for i in range(0, len(dataset), batch_size):
        chunk = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batches.append(collate_fn(chunk))
    
    return batches

# --------------------------------------------------------------------------- #
#  Comprehensive Demo and Testing
# --------------------------------------------------------------------------- #
def run_comprehensive_demo():
    """Run comprehensive demonstration of enhanced lattice transformer."""
    print("=" * 80)
    print("Enhanced Lattice-Dynamic Transformer Demo")
    print("=" * 80)
    
    # Model configuration
    config = {
        'vocab_size': 100,
        'd_model': 128,
        'n_heads': 6,  # 2x3 lattice
        'n_layers': 2,
        'd_ff': 512,
        'max_seq_len': 64,
        'lattice_shape': (2, 3)
    }
    
    print(f"\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model and enable research mode
    print(f"\n1. Creating Enhanced Lattice Transformer...")
    model = LatticeTransformer(**config)
    model.enable_research_mode()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    print(f"✓ Research mode enabled for comprehensive monitoring")
    
    # Create visualization and training tools
    visualizer = LatticeVisualizer(model)
    trainer = LatticeTrainer(model, visualizer)
    print(f"✓ Visualization and training tools initialized")
    
    # Create datasets with different patterns
    print(f"\n2. Creating Test Datasets...")
    datasets = {
        'periodic': SimpleSequenceDataset(config['vocab_size'], 32, 200, 'periodic'),
        'hierarchical': SimpleSequenceDataset(config['vocab_size'], 32, 200, 'hierarchical'),
        'random': SimpleSequenceDataset(config['vocab_size'], 32, 200, 'random')
    }
    
    for name, dataset in datasets.items():
        print(f"✓ {name.capitalize()} dataset: {len(dataset)} samples")
    
    # Comparative training test
    print(f"\n3. Running Comparative Training Test...")
    
    # Test with periodic data (should show synchronization)
    dataloader = create_dataloader(datasets['periodic'], batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    print(f"\nTraining on periodic data (should promote synchronization)...")
    
    # Quick training run
    n_epochs = 3
    epoch_results = trainer.adaptive_training_schedule(
        dataloader[:10],  # Subset for demo
        optimizer, 
        n_epochs=n_epochs,
        warmup_epochs=1,
        verbose=True
    )
    
    print(f"\n4. Analyzing Results...")
    
    # Show lattice evolution
    lattice_states = model.get_lattice_states()
    sync_metrics = model.get_synchronization_metrics()
    
    print(f"\nLattice Analysis:")
    for block_idx, state in enumerate(lattice_states):
        phases = state['phases']
        couplings = state['couplings']
        
        print(f"  Block {block_idx}:")
        print(f"    Phase range: [{phases.min():.3f}, {phases.max():.3f}]")
        print(f"    Phase std: {phases.std():.3f}")
        print(f"    Coupling range: [{couplings.min():.3f}, {couplings.max():.3f}]")
        
        # Synchronization measure
        complex_phases = np.exp(1j * phases)
        sync_order = np.abs(np.mean(complex_phases))
        print(f"    Synchronization: {sync_order:.3f}")
    
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
                print(f"✓ Lattice state visualization created")
            
            # Synchronization evolution
            fig2 = visualizer.plot_synchronization_evolution()
            if fig2:
                print(f"✓ Synchronization evolution plot created")
            
            # Training progress
            fig3 = trainer.plot_training_progress()
            if fig3:
                print(f"✓ Training progress visualization created")
            
            # Note: In interactive environment, uncomment to display:
            # plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    else:
        print(f"\n6. Visualizations unavailable (install matplotlib)")
    
    print(f"\n" + "=" * 80)
    print(f"DEMO COMPLETE")
    print(f"=" * 80)
    print(f"✅ Enhanced Lattice Transformer successfully demonstrated")
    print(f"✅ Production-ready architecture with research tooling")
    print(f"✅ Lattice dynamics with performance optimization")
    print(f"✅ Comprehensive monitoring and visualization")
    print(f"✅ Specialized training procedures")
    
    # Final recommendations
    print(f"\nRecommended Usage:")
    print(f"  • Use model.enable_research_mode() for detailed analysis")
    print(f"  • Use model.disable_research_mode() for production")
    print(f"  • Set use_lattice_dynamics=False for pure performance")
    print(f"  • Use trainer.adaptive_training_schedule() for best results")
    
    return model, visualizer, trainer

# --------------------------------------------------------------------------- #
#  Quick Sanity Check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Quick functionality test
    print("Running quick sanity check...")
    
    model = LatticeTransformer(vocab_size=100, d_model=64, n_heads=6, n_layers=2, d_ff=256)
    x = torch.randint(0, 100, (4, 20))
    
    # Test both modes
    logits_lattice, loss_lattice = model(x, labels=x, use_lattice_dynamics=True)
    logits_standard, loss_standard = model(x, labels=x, use_lattice_dynamics=False)
    
    print(f"✓ Lattice mode: {logits_lattice.shape}, loss: {loss_lattice.item():.4f}")
    print(f"✓ Standard mode: {logits_standard.shape}, loss: {loss_standard.item():.4f}")
    print(f"✓ Quick sanity check passed!")
    
    print(f"\nRun run_comprehensive_demo() for full demonstration.")
    
    # Uncomment for full demo:
    # run_comprehensive_demo()
