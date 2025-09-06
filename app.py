# Req: Python ≥ 3.8, PyTorch ≥ 2.0
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Return lower-triangular boolean mask (1 = keep, 0 = masked)."""
    return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))


def _check_divisible(d_model: int, n_heads: int) -> None:
    if d_model % n_heads:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")


# --------------------------------------------------------------------------- #
#  Single head with phase oscillator
# --------------------------------------------------------------------------- #
class LatticeAttentionHead(nn.Module):
    """Single attention head whose logits are biased by a Kuramoto phase oscillator."""

    def __init__(
        self,
        d_model: int,
        d_k: int,
        lattice_pos: Tuple[int, int],
        coupling_strength: float = 0.1,
        intrinsic_freq: float = 1.0,
    ):
        super().__init__()
        self.d_k = d_k
        self.lattice_pos = lattice_pos

        # Standard QKV
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

        # Phase oscillator (buffer → autograd safe)
        self.register_buffer("phase", torch.tensor(0.0))
        self.register_buffer("intrinsic_freq", torch.tensor(float(intrinsic_freq)))
        self.coupling_strength = nn.Parameter(torch.tensor(coupling_strength))
        self.lattice_constant = nn.Parameter(torch.tensor(1.0))

        # Pre-computed sqrt(d_k) for efficiency
        self.register_buffer("scale", torch.tensor(math.sqrt(float(d_k))))

    # --------------------------------------------------------------------- #
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        neighbour_phases: Tensor,  # (n_neighbours,)
        neighbour_weights: Tensor,  # (n_neighbours,)
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (output, attn_weights)."""
        B, L, _ = query.shape

        Q = self.W_q(query)  # (B, L, d_k)
        K = self.W_k(key)
        V = self.W_v(value)

        # Standard scaled dot-product
        scores = torch.einsum("bqd,bkd->bqk", Q, K) / self.scale  # (B, L, L)

        # Kuramoto update (single step)
        if neighbour_phases.numel():
            phase_diff = neighbour_phases - self.phase
            coupling = (neighbour_weights * torch.sin(phase_diff)).sum()
            new_phase = self.phase + 0.01 * (self.intrinsic_freq + self.coupling_strength * coupling)
        else:
            new_phase = self.phase + 0.01 * self.intrinsic_freq
        # Clamp to avoid drift
        self.phase = new_phase.remainder(2 * math.pi)

        # Add phase bias (cosine) to *entire* score matrix
        bias = 0.1 * torch.cos(self.phase)
        scores = scores + bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bql,blk->bqk", attn, V)
        return out, attn.detach()


# --------------------------------------------------------------------------- #
#  Multi-head wrapper with cached neighbour topology
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

        # Build 2-D lattice
        if lattice_shape is None:
            side = int(math.sqrt(n_heads))
            lattice_shape = (side, side + 1) if side * side < n_heads else (side, side)
        self.lattice_shape: Tuple[int, int] = lattice_shape

        # Instantiate heads
        self.heads = nn.ModuleList()
        self.positions: List[Tuple[int, int]] = []
        for idx in range(n_heads):
            row, col = divmod(idx, lattice_shape[1])
            self.positions.append((row, col))
            freq = 1.0 + 0.1 * idx
            self.heads.append(
                LatticeAttentionHead(d_model, self.d_k, (row, col), intrinsic_freq=freq)
            )

        # Pre-compute neighbour lists (static)
        self._build_neighbours()

        self.out_proj = nn.Linear(d_model, d_model)

    # --------------------------------------------------------------------- #
    def _build_neighbours(self) -> None:
        """Cache neighbour indices and distance weights once."""
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

    # --------------------------------------------------------------------- #
    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        B, L, _ = query.shape
        if mask is not None and mask.dim() == 2:  # (B,K) → (B,1,K,K)
            mask = mask.unsqueeze(1).unsqueeze(1)

        # Fast path: vanilla sdpa when no lattice dynamics requested
        if self.use_sdpa and mask is None:
            q = (
                self.out_proj.weight.new_empty(B, L, self.n_heads, self.d_k)
                .transpose(1, 2)
                .contiguous()
            )
            k = v = q
            for h_idx, head in enumerate(self.heads):
                q[:, h_idx] = head.W_q(query)
                k[:, h_idx] = head.W_k(key)
                v[:, h_idx] = head.W_v(value)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
            out = out.transpose(1, 2).reshape(B, L, self.d_model)
            return self.out_proj(out), []

        # Full path with oscillator dynamics
        outs, attns = [], []
        for h_idx, head in enumerate(self.heads):
            idx = self.neighbour_idx[h_idx]
            w = self.neighbour_w[h_idx]
            mask_valid = idx >= 0
            phases = self.heads[idx[mask_valid]].phase
            weights = w[mask_valid]
            o, a = head(query, key, value, phases, weights, mask)
            outs.append(o)
            attns.append(a)

        multi = torch.cat(outs, dim=-1)
        return self.out_proj(multi), attns


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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h, _ = self.lattice_attn(x, x, x, mask)
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

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            LatticeTransformerBlock(d_model, n_heads, d_ff, lattice_shape)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Tie weights (optional)
        self.lm_head.weight = self.token_emb.weight

        self.register_buffer("pos_buf", torch.arange(max_seq_len))

    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        causal: bool = True,
        return_dict: bool = False,
    ):
        B, L = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids) + self.pos_emb(self.pos_buf[:L])
        x *= math.sqrt(self.d_model)

        mask = None
        if causal:
            mask = _make_causal_mask(L, device)

        for blk in self.blocks:
            x = blk(x, mask)

        logits = self.lm_head(self.norm(x))

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
#  Dataset / training helpers (unchanged API)
# --------------------------------------------------------------------------- #
class SimpleSequenceDataset:
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 1000):
        self.seq_len = seq_len
        self.data = []
        for _ in range(n_samples):
            seq = torch.randint(1, vocab_size - 1, (seq_len,))
            # trivial pattern
            seq[2::3] = seq[0]
            tgt = torch.cat([seq[1:], torch.tensor([0])])
            self.data.append((seq, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloader(dataset, batch_size: int = 32):
    def collate(batch):
        src, tgt = zip(*batch)
        return torch.stack(src), torch.stack(tgt)

    batches = []
    for i in range(0, len(dataset), batch_size):
        chunk = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batches.append(collate(chunk))
    return batches


# --------------------------------------------------------------------------- #
#  Quick sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = LatticeTransformer(vocab_size=100, d_model=64, n_heads=6, n_layers=2, d_ff=256)
    x = torch.randint(0, 100, (4, 20))
    logits, loss = model(x, labels=x, return_dict=False)
    print("logits:", logits.shape, "loss:", loss.item())
