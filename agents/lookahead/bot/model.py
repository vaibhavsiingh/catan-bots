"""The transformer: 76 tokens in (1 global, me, opponent, 19 tiles, 54 nodes), 294 move scores out,
plus a value head (chance that the player to move wins).

Each place scores its own moves: node tokens score settle/city there, the two node tokens at the
ends of an edge score the road between them, tile tokens score the robber, the global token scores
everything else. Edges are not tokens (that would double the sequence length and halve the speed on
CPU); their features are added to the nodes they touch instead."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bot.encode import F_EDGE, F_GLOBAL, F_ME, F_NODE, F_OPP, F_TILE, Encoder, MoveTable, Topology

N_TOKENS = 3 + 19 + 54          # global, me, opponent, tiles, nodes
MAX_HOPS = 8                    # graph distances above this share one bucket
ARCHS = ("base", "G", "GD")     # base: plain attention; G: + hop-distance bias; GD: + dynamic bias templates
N_TEMPLATES = 8


def hop_buckets(topo):
    """(76, 76) long: shortest-path hops between tokens on the board graph (tile-node and node-node
    adjacency), capped at MAX_HOPS; the global/me/opp tokens get their own bucket (MAX_HOPS + 1),
    Graphormer's virtual-node style. 0 = the token itself."""
    import collections
    adj = collections.defaultdict(set)
    for t, nodes in enumerate(topo.tile_nodes):
        for n in nodes:
            adj[3 + t].add(22 + n)
            adj[22 + n].add(3 + t)
    for a, b in topo.edges:
        adj[22 + a].add(22 + b)
        adj[22 + b].add(22 + a)
    D = torch.full((N_TOKENS, N_TOKENS), MAX_HOPS + 1, dtype=torch.long)
    for src in range(3, N_TOKENS):
        dist = {src: 0}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for v, k in dist.items():
            D[src, v] = min(k, MAX_HOPS)
    for i in range(3):
        D[i, i] = 0
    return D


class CatanNet(nn.Module):
    def __init__(self, topo, d=64, layers=3, heads=4, f_opp=F_OPP, arch="base"):
        super().__init__()
        assert arch in ARCHS, arch
        self.arch, self.heads = arch, heads
        self.config = dict(d=d, layers=layers, heads=heads, f_opp=f_opp)
        if arch != "base":
            self.config["arch"] = arch
        self.proj_global = nn.Linear(F_GLOBAL, d)
        self.proj_me = nn.Linear(F_ME, d)
        self.proj_opp = nn.Linear(f_opp, d)
        self.proj_tile = nn.Linear(F_TILE, d)
        self.proj_node = nn.Linear(F_NODE + F_EDGE, d)     # node features + summary of touching edges
        # Which token is which (tile 5 vs tile 6, node 12 vs node 13). Same scale as the projected
        # features, so position information is not drowned out (the tic-tac-toe init bug).
        self.token_id = nn.Parameter(torch.randn(1, N_TOKENS, d) * 0.5)

        # Fixed board wiring: which edges touch which node, and the two ends of every edge.
        node_edge = torch.zeros(54, 72)
        ends = torch.tensor(topo.edges)                     # (72, 2)
        node_edge[ends[:, 0], torch.arange(72)] = 1
        node_edge[ends[:, 1], torch.arange(72)] = 1
        self.register_buffer("node_edge", node_edge / 3)    # a node touches at most 3 edges
        self.register_buffer("edge_a", ends[:, 0].clone())
        self.register_buffer("edge_b", ends[:, 1].clone())

        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        if arch in ("G", "GD"):
            # A learned attention bias per (layer, head, hop distance), zero at the start: a checkpoint
            # of the base network loads and plays exactly as before until training moves it.
            self.register_buffer("hops", hop_buckets(topo))
            self.hop_bias = nn.Parameter(torch.zeros(layers, heads, MAX_HOPS + 2))
        if arch == "GD":
            # Leela's smolgen, lite: the position (mean of its tokens) picks a mix of learned 76x76
            # attention templates per head. The mixer's last layer starts at zero (no bias at first).
            self.templates = nn.Parameter(torch.randn(N_TEMPLATES, N_TOKENS, N_TOKENS) * 0.02)
            self.mixer = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, heads * N_TEMPLATES))
            nn.init.zeros_(self.mixer[2].weight)
            nn.init.zeros_(self.mixer[2].bias)

        self.node_head = nn.Linear(d, 2)                    # settle here, city here
        self.road_head = nn.Sequential(nn.Linear(d + F_EDGE, d), nn.GELU(), nn.Linear(d, 1))
        self.tile_head = nn.Linear(d, 1)                    # robber here
        self.global_head = nn.Linear(d, 95)                 # roll, end turn, buy, cards, trades, discards
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # Auxiliary target (KataGo-style): both players' final victory points / 10. Many more bits per
        # game than win/loss, which helps the value head learn instead of memorising games.
        self.vp_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.zeros_(self.vp_head[2].weight)              # start at the average final score, so the
        with torch.no_grad():                                # new head's first gradients are small
            self.vp_head[2].bias.copy_(torch.tensor([0.75, 0.65]))

    def forward(self, x, mask=None, aux=False):
        """x: dict of float tensors with a batch dimension. mask: (B, 294) bool of legal moves.
        Returns (logits (B, 294), value logits (B,)), plus final-VP predictions (B, 2) if aux=True."""
        B = x["node"].shape[0]
        edge_to_node = torch.einsum("ne,bef->bnf", self.node_edge, x["edge"])
        tokens = torch.cat([
            self.proj_global(x["global"])[:, None],
            self.proj_me(x["me"])[:, None],
            self.proj_opp(x["opp"])[:, None],
            self.proj_tile(x["tile"]),
            self.proj_node(torch.cat([x["node"], edge_to_node], -1)),
        ], 1) + self.token_id
        if self.arch == "base":
            h = self.norm(self.encoder(tokens))
        else:
            h = self.norm(self._biased_encoder(tokens))
        g, tiles, nodes = h[:, 0], h[:, 3:22], h[:, 22:]

        node_scores = self.node_head(nodes)                                  # (B, 54, 2)
        pair = nodes[:, self.edge_a] + nodes[:, self.edge_b]                 # (B, 72, d), symmetric
        road_scores = self.road_head(torch.cat([pair, x["edge"]], -1))[..., 0]
        logits = torch.cat([
            node_scores[..., 0], node_scores[..., 1],                       # settle 0-53, city 54-107
            road_scores,                                                     # roads 108-179
            self.tile_head(tiles)[..., 0],                                   # robber 180-198
            self.global_head(g),                                             # global 199-293
        ], 1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)        # -1e9, not -inf: 0 * -inf = NaN in the loss
        if aux:
            return logits, self.value_head(g)[:, 0], self.vp_head(g)
        return logits, self.value_head(g)[:, 0]


    def _biased_encoder(self, tokens):
        """The same encoder layers, with an additive attention bias (B * heads, 76, 76) per layer."""
        B, H = tokens.shape[0], self.heads
        dyn = 0
        if self.arch == "GD":
            w = self.mixer(tokens.mean(1)).view(B, H, N_TEMPLATES)
            dyn = torch.einsum("bhk,kij->bhij", w, self.templates)                # (B, H, 76, 76)
        h = tokens
        for l, layer in enumerate(self.encoder.layers):
            bias = self.hop_bias[l][:, self.hops][None] + dyn                       # (1 or B, H, 76, 76)
            h = _layer_with_bias(layer, h, bias, H)
        return h


def _layer_with_bias(layer, x, bias, heads):
    """nn.TransformerEncoderLayer (norm_first, gelu, no dropout) with an additive attention bias, using
    its own weights; scaled_dot_product_attention keeps it about as fast as the fused original."""
    B, L, d = x.shape
    attn = layer.self_attn
    qkv = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.view(B, L, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    x = x + attn.out_proj(a.transpose(1, 2).reshape(B, L, d))
    return x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))


def losses(logits, value_logit, target, won, vp_pred=None, vp=None, value_weight=0.5, aux_weight=0.5,
           policy_weight=1.0):
    """Policy: cross-entropy against the target. Value: did the player to move win (BCE).
    Aux: final VP of (mover, opponent) / 10 (MSE). Labels of -1 mean "no label" and are skipped.
    Returns (total, policy, value, aux)."""
    policy = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    has = won >= 0
    value = (F.binary_cross_entropy_with_logits(value_logit[has], won[has]) if has.any()
             else value_logit.sum() * 0)
    total = policy_weight * policy + value_weight * value
    aux = torch.zeros(())
    if vp_pred is not None and vp is not None:
        has_vp = vp[:, 0] >= 0
        if has_vp.any():
            aux = F.mse_loss(vp_pred[has_vp], vp[has_vp] / 10)
            total = total + aux_weight * aux
    return total, policy, value, aux


_LOADED = {}            # path -> (model, encoder, move table); load once per process


def load(path, board_map):
    """A checkpoint as (model in eval mode, encoder, move table), cached per path."""
    if path not in _LOADED:
        ckpt = torch.load(path, map_location="cpu")
        topo = Topology(board_map)
        cfg = ckpt["config"]
        model = CatanNet(topo, d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"], f_opp=cfg["f_opp"],
                         arch=cfg.get("arch", "base"))
        model.load_state_dict(ckpt["model"], strict=False)   # older checkpoints have no vp_head
        model.eval()
        enc = Encoder(topo, see_opponent_hand=cfg["f_opp"] > 16)
        _LOADED[path] = (model, enc, MoveTable(topo))
    return _LOADED[path]
