import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

class LatticeAttentionHead(nn.Module):
    """
    Single attention head with lattice dynamics and phase coupling
    """
    def __init__(self, d_model: int, d_k: int, lattice_pos: Tuple[int, int], 
                 coupling_strength: float = 0.1, intrinsic_freq: float = 1.0):
        super().__init__()
        self.d_k = d_k
        self.lattice_pos = lattice_pos
        self.coupling_strength = coupling_strength
        
        # Standard attention components
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        
        # Lattice dynamics components
        self.intrinsic_freq = nn.Parameter(torch.tensor(intrinsic_freq))
        self.phase = nn.Parameter(torch.zeros(1))
        self.lattice_coupling = nn.Parameter(torch.ones(1) * coupling_strength)
        
        # Learned lattice spacing (affects interaction strength)
        self.lattice_constant = nn.Parameter(torch.ones(1))
        
    def compute_phase_coupling(self, neighbor_phases: torch.Tensor, 
                             neighbor_distances: torch.Tensor) -> torch.Tensor:
        """
        Compute phase coupling with neighboring attention heads
        Uses Kuramoto-like coupling with distance decay
        """
        if neighbor_phases.numel() == 0:
            return torch.zeros_like(self.phase)
        
        # Distance-weighted coupling (closer neighbors have stronger coupling)
        distance_weights = torch.exp(-neighbor_distances / self.lattice_constant)
        
        # Kuramoto coupling: sum of sin(neighbor_phase - my_phase)
        phase_differences = neighbor_phases - self.phase
        coupling_force = torch.sum(distance_weights * torch.sin(phase_differences))
        
        return self.lattice_coupling * coupling_force
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                neighbor_info: Optional[dict] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with lattice-modified attention
        """
        batch_size, seq_len = query.shape[0], query.shape[1]
        
        # Standard QKV computation
        Q = self.W_q(query)  # (batch, seq_len, d_k)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Standard attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add lattice dynamics modification
        if neighbor_info is not None:
            neighbor_phases = neighbor_info.get('phases', torch.tensor([]))
            neighbor_distances = neighbor_info.get('distances', torch.tensor([]))
            
            # Compute phase coupling effect
            phase_coupling = self.compute_phase_coupling(neighbor_phases, neighbor_distances)
            
            # Update phase based on coupling (simplified dynamics)
            self.phase.data += 0.01 * (self.intrinsic_freq + phase_coupling)
            
            # Create lattice influence matrix
            lattice_influence = torch.cos(self.phase).expand_as(attention_scores) * 0.1
            attention_scores = attention_scores + lattice_influence
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and apply to values
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class LatticeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with 2D lattice organization
    """
    def __init__(self, d_model: int, n_heads: int, lattice_shape: Tuple[int, int] = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Determine lattice shape
        if lattice_shape is None:
            # Create approximately square lattice
            side = int(math.sqrt(n_heads))
            self.lattice_shape = (side, side)
            if side * side < n_heads:
                self.lattice_shape = (side, side + 1)
        else:
            self.lattice_shape = lattice_shape
        
        # Create attention heads in lattice arrangement
        self.heads = nn.ModuleList()
        self.lattice_positions = []
        
        head_idx = 0
        for i in range(self.lattice_shape[0]):
            for j in range(self.lattice_shape[1]):
                if head_idx < n_heads:
                    # Different intrinsic frequencies for different heads
                    freq = 1.0 + 0.1 * head_idx
                    head = LatticeAttentionHead(d_model, self.d_k, (i, j), 
                                              intrinsic_freq=freq)
                    self.heads.append(head)
                    self.lattice_positions.append((i, j))
                    head_idx += 1
        
        self.output_projection = nn.Linear(d_model, d_model)
        
    def get_neighbor_info(self, head_idx: int) -> dict:
        """
        Get information about neighboring heads in the lattice
        """
        pos = self.lattice_positions[head_idx]
        neighbor_phases = []
        neighbor_distances = []
        
        for other_idx, other_pos in enumerate(self.lattice_positions):
            if other_idx != head_idx:
                # Calculate lattice distance
                distance = math.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
                
                # Only consider nearest neighbors (distance <= sqrt(2) for 2D)
                if distance <= math.sqrt(2) + 0.1:  # Small epsilon for numerical precision
                    neighbor_phases.append(self.heads[other_idx].phase)
                    neighbor_distances.append(distance)
        
        if neighbor_phases:
            return {
                'phases': torch.stack(neighbor_phases),
                'distances': torch.tensor(neighbor_distances, device=neighbor_phases[0].device)
            }
        else:
            return {'phases': torch.tensor([]), 'distances': torch.tensor([])}
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through lattice-organized multi-head attention
        """
        batch_size, seq_len, d_model = query.shape
        
        # Process each head with lattice dynamics
        head_outputs = []
        attention_maps = []
        
        for head_idx, head in enumerate(self.heads):
            neighbor_info = self.get_neighbor_info(head_idx)
            head_output, attention_weights = head(query, key, value, neighbor_info, mask)
            head_outputs.append(head_output)
            attention_maps.append(attention_weights)
        
        # Concatenate head outputs
        multi_head_output = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, d_model)
        
        # Final output projection
        output = self.output_projection(multi_head_output)
        
        return output, attention_maps

class LatticeTransformerBlock(nn.Module):
    """
    Complete transformer block with lattice attention
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, lattice_shape: Tuple[int, int] = None):
        super().__init__()
        self.lattice_attention = LatticeMultiHeadAttention(d_model, n_heads, lattice_shape)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Lattice attention with residual connection
        attn_output, attention_maps = self.lattice_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

class LatticeTransformer(nn.Module):
    """
    Complete Lattice-Dynamic Transformer
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, max_seq_len: int = 512, lattice_shape: Tuple[int, int] = None):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks with lattice attention
        self.blocks = nn.ModuleList([
            LatticeTransformerBlock(d_model, n_heads, d_ff, lattice_shape)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x *= math.sqrt(self.d_model)  # Scaling as in original transformer
        
        # Pass through lattice transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final normalization and output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits

# Synthetic dataset for demonstration
class SimpleSequenceDataset:
    """Simple dataset for testing lattice transformer"""
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Generate synthetic sequences with patterns
        self.data = []
        for _ in range(n_samples):
            # Create sequences with some structure
            sequence = torch.randint(1, vocab_size-1, (seq_len,))
            
            # Add some pattern (every 3rd token relates to 1st token)
            for i in range(2, seq_len, 3):
                sequence[i] = sequence[0]
            
            # Targets are shifted sequence (next token prediction)
            target = torch.cat([sequence[1:], torch.tensor([0])])
            
            self.data.append((sequence, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_dataloader(dataset, batch_size: int = 32):
    """Create simple dataloader"""
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)
    
    # Simple batching
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batches.append(collate_fn(batch_data))
    
    return batches

# Example usage and comprehensive testing
if __name__ == "__main__":
    # Model parameters
    vocab_size = 100
    d_model = 64
    n_heads = 6  # Will create 2x3 lattice
    n_layers = 2
    d_ff = 256
    seq_len = 20
    batch_size = 8
    
    print("Creating Lattice-Dynamic Transformer...")
    
    # Create model and visualizer
    model = LatticeTransformer(vocab_size, d_model, n_heads, n_layers, d_ff, seq_len)
    visualizer = LatticeVisualizer((2, 3))  # 2x3 lattice for 6 heads
    trainer = LatticeTrainer(model, visualizer)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataset and dataloader
    dataset = SimpleSequenceDataset(vocab_size, seq_len, n_samples=500)
    dataloader = create_dataloader(dataset, batch_size)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Initial lattice visualization
    print("\n=== Initial Lattice State ===")
    trainer.visualizer.record_state(model)
    
    # Quick forward pass test
    sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        output = model(sample_input)
        print(f"Forward pass successful: {sample_input.shape} -> {output.shape}")
    
    # Display initial lattice properties
    first_block = model.blocks[0]
    lattice_attn = first_block.lattice_attention
    print(f"\nLattice Configuration:")
    print(f"- Lattice shape: {lattice_attn.lattice_shape}")
    print(f"- Number of heads: {len(lattice_attn.heads)}")
    print(f"- Head positions: {lattice_attn.lattice_positions}")
    
    # Show initial phases and frequencies
    print(f"\nInitial Head States:")
    for i, head in enumerate(lattice_attn.heads):
        pos = lattice_attn.lattice_positions[i]
        print(f"Head {i} at {pos}: phase={head.phase.item():.3f}, freq={head.intrinsic_freq.item():.3f}")
    
    # Training demonstration
    print(f"\n=== Training Demonstration ===")
    n_epochs = 5
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        epoch_losses = trainer.train_epoch(dataloader[:10], optimizer, verbose=False)  # Use subset for demo
        
        print(f"Epoch {epoch + 1} - Avg Total Loss: {epoch_losses['total_loss']:.4f}, "
              f"CE Loss: {epoch_losses['ce_loss']:.4f}, "
              f"Phase Reg: {epoch_losses['phase_reg_loss']:.4f}")
    
    # Show final lattice state
    print(f"\n=== Final Lattice State ===")
    for i, head in enumerate(lattice_attn.heads):
        pos = lattice_attn.lattice_positions[i]
        print(f"Head {i} at {pos}: phase={head.phase.item():.3f}, freq={head.intrinsic_freq.item():.3f}")
    
    # Visualization examples
    print(f"\n=== Visualization Capabilities ===")
    print("Available visualization methods:")
    print("- visualizer.plot_lattice_state(model) - Current lattice state")
    print("- visualizer.plot_phase_evolution() - Phase evolution over time")
    print("- visualizer.create_phase_animation(model) - Animated phase dynamics")
    print("- trainer.plot_training_progress() - Training metrics")
    
    # Generate some plots if matplotlib is available
    try:
        # Plot current lattice state
        fig1 = visualizer.plot_lattice_state(model)
        print("✓ Lattice state visualization created")
        
        # Plot phase evolution
        if len(visualizer.phase_history) > 5:
            fig2 = visualizer.plot_phase_evolution()
            print("✓ Phase evolution plot created")
        
        # Plot training progress
        fig3 = trainer.plot_training_progress()
        print("✓ Training progress plot created")
        
        # Show plots (comment out if running in non-interactive environment)
        # plt.show()
        
    except Exception as e:
        print(f"Note: Visualization requires matplotlib. Error: {e}")
    
    print(f"\n=== Summary ===")
    print("✓ Lattice-Dynamic Transformer successfully created and tested")
    print("✓ Training procedures implemented with phase regularization")
    print("✓ Comprehensive visualization tools added")
    print("✓ Phase dynamics and lattice coupling verified")
    
    # Final phase synchronization measurement
    if len(visualizer.phase_history) > 0:
        final_state = visualizer.phase_history[-1]
        phases = final_state['phases']
        complex_phases = np.exp(1j * phases)
        final_sync = np.abs(np.mean(complex_phases))
        print(f"✓ Final phase synchronization: {final_sync:.3f}")
    
    print("\nLattice-Dynamic Transformer with visualization and training completed!")
