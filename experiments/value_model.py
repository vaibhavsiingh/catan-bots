"""Dedicated state-value transformer (same entity-token encoder, value-only heads).

Heads: win logit, dvp (short-horizon VP-diff change), margin (final VP diff / 10).
"""

import torch
import torch.nn as nn

from encode import TILE_F, NODE_F, EDGE_F, PLAYER_F, GLOBAL_F, N_TOKENS


class ValueNet(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(f, d) for f in (TILE_F, NODE_F, EDGE_F, PLAYER_F, GLOBAL_F)])
        self.pos = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        # attention-pool over all tokens + global token -> heads
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, tile, node, edge, player, glob):
        x = torch.cat([p(t) for p, t in zip(self.proj, (tile, node, edge, player, glob.unsqueeze(1)))], 1)
        h = self.norm(self.encoder(x + self.pos))
        pooled, _ = self.pool(self.pool_q.expand(h.size(0), -1, -1), h, h)
        out = self.head(torch.cat([pooled[:, 0], h[:, -1]], -1))
        return out[:, 0], out[:, 1], out[:, 2]  # win_logit, dvp, margin
