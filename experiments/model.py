"""Entity-token transformer: board objects are tokens, legal actions are scored by pointing at them."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import (TILE_F, NODE_F, EDGE_F, PLAYER_F, GLOBAL_F, N_TOKENS,
                    ACTION_TYPES, ACT_RES_F, OFF_GLOBAL)


class CatanTransformer(nn.Module):
    def __init__(self, d=96, layers=3, heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(f, d) for f in (TILE_F, NODE_F, EDGE_F, PLAYER_F, GLOBAL_F)])
        self.pos = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.type_emb = nn.Embedding(len(ACTION_TYPES), d)
        self.res_proj = nn.Linear(ACT_RES_F, d)
        self.null_ptr = nn.Parameter(torch.zeros(2, d))
        self.ptr_proj = nn.Linear(d, d)
        self.ptr2_proj = nn.Linear(d, d)
        self.score = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def encode(self, tile, node, edge, player, glob):
        x = torch.cat([p(t) for p, t in zip(self.proj, (tile, node, edge, player, glob.unsqueeze(1)))], 1)
        return self.norm(self.encoder(x + self.pos))

    def forward(self, tile, node, edge, player, glob, a_int, a_res, a_mask):
        """a_int: (B,A,3) [type, ptr, ptr2] (-1 = none); a_mask: (B,A) bool legal.
        Returns action logits (B,A) and value logit (B,)."""
        h = self.encode(tile, node, edge, player, glob)
        B, A, _ = a_int.shape
        g = h[:, OFF_GLOBAL]

        def gather(idx, null):
            safe = idx.clamp(min=0)
            out = torch.gather(h, 1, safe.unsqueeze(-1).expand(B, A, h.size(-1)))
            return torch.where((idx >= 0).unsqueeze(-1), out, null)

        a = (self.type_emb(a_int[..., 0].clamp(min=0))
             + self.ptr_proj(gather(a_int[..., 1], self.null_ptr[0]))
             + self.ptr2_proj(gather(a_int[..., 2], self.null_ptr[1]))
             + self.res_proj(a_res))
        logits = self.score(torch.cat([a, g.unsqueeze(1).expand(B, A, -1)], -1)).squeeze(-1)
        logits = logits.masked_fill(~a_mask, -1e9)
        return logits, self.value(g).squeeze(-1)
