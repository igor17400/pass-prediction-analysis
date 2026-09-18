"""Graph attention network over the freeze frame.

Every visible player is a node; the graph is fully connected and attention weights the
edges, with a learned bias from each pair's relative offset (an edge feature). One extra
token carries the pass itself (start, end, length, angle, pressure) and is read out for
the prediction. Equivalent to a transformer encoder over player tokens with edge-aware
attention, which is why the same code can honestly be called either.
"""
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

from features import NODE_COLS, PASS_COLS

CACHE = Path(__file__).resolve().parents[1] / "cache"
WALL_S = 12 * 60
MAX_NODES = 24


def build_tensors(nodes: pl.DataFrame, passes: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """passes rows define the order; returns node feats [B,N,F], pad mask [B,N] (True = pad), pass feats [B,P]."""
    order = passes.select("pid").with_row_index("row")
    n = nodes.join(order, on="pid").filter(pl.col("node") < MAX_NODES)
    x = np.zeros((len(passes), MAX_NODES, len(NODE_COLS)), dtype=np.float32)
    pad = np.ones((len(passes), MAX_NODES), dtype=bool)
    r, c = n["row"].to_numpy(), n["node"].to_numpy()
    x[r, c] = n.select(NODE_COLS).to_numpy().astype(np.float32)
    pad[r, c] = False
    return torch.from_numpy(x), torch.from_numpy(pad), torch.from_numpy(passes.select(PASS_COLS).to_numpy().astype(np.float32))


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.edge = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, heads))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h: torch.Tensor, rel: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, L, _ = h.shape
        bias = self.edge(rel).permute(0, 3, 1, 2)  # [B,H,L,L]
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf")).reshape(B * self.h, L, L)
        a, _ = self.attn(self.n1(h), self.n1(h), self.n1(h), attn_mask=bias, need_weights=False)
        h = h + a
        return h + self.ff(self.n2(h))


class PlayerGAT(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, layers: int = 3):
        super().__init__()
        self.node_in = nn.Linear(len(NODE_COLS), d)
        self.pass_in = nn.Linear(len(PASS_COLS), d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor, pad: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # token 0 is the pass, positioned at the pass start; players follow
        pos = torch.cat([p[:, None, :2], x[:, :, :2]], 1)  # normalised (x, y) of every token
        rel = pos[:, :, None, :] - pos[:, None, :, :]
        rel = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], -1)  # [B,L,L,3]: dx, dy, dist
        h = torch.cat([self.pass_in(p)[:, None], self.node_in(x)], 1)
        pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        for b in self.blocks:
            h = b(h, rel, pad)
        return self.out(h[:, 0]).squeeze(-1)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model: nn.Module, x, pad, p, dev, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(x[i : i + bs].to(dev), pad[i : i + bs].to(dev), p[i : i + bs].to(dev))).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy()
