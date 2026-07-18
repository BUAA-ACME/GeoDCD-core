import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.cluster import KMeans


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CausalTransformerBlock(nn.Module):
    """Transformer encoder with causal masking for time series."""

    def __init__(self, input_dim, output_dim, d_model=64, nhead=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        B, T, C = x.shape
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        if not hasattr(self, 'causal_mask') or self.causal_mask.size(0) != T:
            mask = self._generate_square_subsequent_mask(T).to(x.device)
            self.register_buffer('causal_mask', mask, persistent=False)

        # NOTE: self.causal_mask is already an additive causal mask (upper-triangular
        # -inf). Do NOT also pass is_causal=True: on PyTorch >= 2.1 that raises
        # "Explicit attn_mask cannot be provided with is_causal=True".
        out = self.transformer(x, mask=self.causal_mask)
        return self.output_proj(out)


class CausalGraphLayer(nn.Module):
    """Learns a sparse causal graph via basis decomposition."""

    def __init__(self, N, d_model, num_bases=4, max_k=32):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.num_bases = min(num_bases, d_model)
        self.max_k = max_k

        self.adjacency = nn.Parameter(
            torch.ones(N, self.max_k) * 0.5 + torch.randn(N, self.max_k) * 0.01
        )
        self.basis_weights = nn.Parameter(torch.randn(self.num_bases, N, self.max_k) * 0.02)
        self.channel_coeffs = nn.Parameter(torch.randn(d_model, self.num_bases))
        self.register_buffer('last_neighbor_indices', None)

    def forward(self, z, neighbor_indices):
        B, N, T, C = z.shape
        k_curr = neighbor_indices.size(1)
        self.last_neighbor_indices = neighbor_indices

        bases = self.basis_weights[:, :, :k_curr]
        adj = self.adjacency[:, :k_curr]
        eff_weights = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
        edge_weights = eff_weights * adj.unsqueeze(0)

        flat_indices = neighbor_indices.view(-1)
        z_flat = z.view(B, N, -1)
        z_neigh_flat = z_flat[:, flat_indices, :]
        z_neigh = z_neigh_flat.view(B, N, k_curr, T, C)

        w_aligned = edge_weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(3)
        z_out = (z_neigh * w_aligned).sum(dim=2)
        return torch.tanh(z_out)

    def get_sparse_dynamic_weights(self, z, neighbor_indices):
        """Compute dynamic causal strengths via modulation of static weights."""
        B, N, T, C = z.shape
        k_curr = neighbor_indices.size(1)

        bases = self.basis_weights[:, :, :k_curr]
        adj = self.adjacency[:, :k_curr]
        eff_weights = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
        static_edge_weights = eff_weights * adj.unsqueeze(0)

        flat_indices = neighbor_indices.view(-1)
        z_flat = z.view(B, N, -1)
        z_neigh_flat = z_flat[:, flat_indices, :]
        z_neigh = z_neigh_flat.view(B, N, k_curr, T, C)

        w_aligned = static_edge_weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(3)
        z_out = (z_neigh * w_aligned).sum(dim=2)

        tanh_grad = 1.0 - torch.tanh(z_out).pow(2)
        dynamic_W_full = tanh_grad.unsqueeze(2) * w_aligned
        dynamic_vals = dynamic_W_full.abs().mean(dim=-1)
        return neighbor_indices, dynamic_vals

    def get_soft_graph(self):
        """Return the dense learned adjacency matrix (N x N)."""
        if self.last_neighbor_indices is None:
            return None
        with torch.no_grad():
            indices = self.last_neighbor_indices
            k = indices.size(1)
            bases = self.basis_weights[..., :k]
            W = torch.einsum('ck,knm->cnm', self.channel_coeffs, bases)
            W_mag = W.abs().mean(dim=0)
            sparse_w = self.adjacency[:, :k].abs() * W_mag
            dense = torch.zeros(self.N, self.N, device=self.adjacency.device)
            dense.scatter_(1, indices, sparse_w)
            return dense.T

    def structural_l1_loss(self):
        return torch.sum(torch.abs(self.adjacency))


class GeoDCDLayer(nn.Module):
    """Single hierarchy level: encoder + causal graph + prediction head."""

    def __init__(self, N, d_model=64, nhead=4, num_layers=2,
                 num_bases=4, max_k=100):
        super().__init__()
        self.geo_encoder = CausalTransformerBlock(1, d_model, d_model, nhead, num_layers)
        self.pred_head = nn.Linear(d_model, 1)
        self.graph = CausalGraphLayer(N, d_model, num_bases, max_k)

    def forward(self, x, mask):
        B, N, T = x.shape
        z = self.geo_encoder(x.reshape(B * N, T, 1)).view(B, N, T, -1)
        zhat_next = self.graph(z, neighbor_indices=mask)
        x_pred = self.pred_head(
            zhat_next[..., :-1, :].squeeze(-1)
        ).view(B, N, T - 1)
        return x_pred, z


class GeometricPooler(nn.Module):
    """Learnable soft clustering with KMeans-initialized centroids."""

    def __init__(self, num_patches, d_coord, shift_scale=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.d_coord = d_coord
        self.centroids = nn.Parameter(torch.randn(num_patches, d_coord))
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.refine = nn.Sequential(
            nn.Linear(d_coord, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, num_patches),
        )
        self._initialized = False
        self.register_buffer('S_matrix', None)

    def _kmeans_init(self, coords):
        coords_np = coords.detach().float().cpu().numpy()
        c_mean = coords_np.mean(axis=0)
        c_std = coords_np.std(axis=0) + 1e-5
        coords_norm = (coords_np - c_mean) / c_std
        kmeans = KMeans(n_clusters=self.num_patches, random_state=42, n_init=10)
        kmeans.fit(coords_norm)
        self.centroids.data.copy_(
            torch.tensor(kmeans.cluster_centers_, dtype=torch.float32,
                         device=coords.device)
        )
        self._initialized = True

    def forward(self, x, coords):
        if not self._initialized:
            self._kmeans_init(coords)

        c_mean = coords.mean(dim=0, keepdim=True)
        c_std = coords.std(dim=0, keepdim=True) + 1e-5
        coords_norm = (coords - c_mean) / c_std

        dists_sq = torch.cdist(coords_norm, self.centroids).pow(2)
        spatial_logits = -dists_sq
        refine_logits = self.refine(coords_norm)
        logits = spatial_logits + 0.1 * refine_logits

        temp = self.log_temp.exp().clamp(min=0.1, max=10.0)

        if self.training:
            S = F.gumbel_softmax(logits / temp, tau=1.0, hard=False, dim=-1)
        else:
            S = F.softmax(logits / temp, dim=-1)

        self.S_matrix = S.detach()
        return S.unsqueeze(0).expand(x.shape[0], -1, -1)

    def get_entropy_loss(self):
        """Encourage crisp (low-entropy) cluster assignments."""
        if self.S_matrix is None:
            return torch.tensor(0.0)
        S = self.S_matrix
        entropy = -(S * (S + 1e-10).log()).sum(dim=-1)
        return entropy.mean()


class CrossLevelBridge(nn.Module):
    """Gated information flow between adjacent hierarchy levels."""

    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_fine, z_coarse, S):
        B, N_fine, T, C = z_fine.shape
        S_norm = S / (S.sum(dim=1, keepdim=True) + 1e-8)
        z_broadcast = torch.einsum('nk,bktc->bntc', S_norm, z_coarse)

        gate = torch.sigmoid(self.gate_proj(torch.cat([z_fine, z_broadcast], dim=-1)))
        value = self.value_proj(z_broadcast)
        z_refined = z_fine + gate * value
        return self.norm(z_refined)


class GeoDCD(nn.Module):
    """Geometric Dynamic Causal Discovery model with hierarchical structure."""

    def __init__(self, N, coords, hierarchy=(32, 8), d_model=64, num_bases=4,
                 penalty_factor=5.0, max_k=32, shift_scale=0.1):
        super().__init__()
        self.dims = [N] + list(hierarchy)
        self.num_levels = len(self.dims)
        self.penalty_factor = penalty_factor
        self.max_k = max_k
        self.shift_scale = shift_scale
        self.d_model = d_model

        coords = torch.tensor(coords).float() if not torch.is_tensor(coords) else coords
        self.register_buffer('coords', coords)
        d_coord = coords.shape[1]

        self.layers = nn.ModuleList()
        self.poolers = nn.ModuleList()
        self.bridges = nn.ModuleList()

        for i in range(self.num_levels):
            self.layers.append(
                GeoDCDLayer(self.dims[i], d_model, num_bases=num_bases, max_k=self.max_k)
            )
            if i < self.num_levels - 1:
                self.poolers.append(GeometricPooler(self.dims[i + 1], d_coord, shift_scale))
                self.register_buffer(
                    f'structure_S_{i}',
                    torch.zeros(self.dims[i], self.dims[i + 1])
                )
                self.bridges.append(CrossLevelBridge(d_model))

    def get_structural_l1_loss(self):
        return sum(layer.graph.structural_l1_loss() for layer in self.layers)

    def get_entropy_loss(self):
        loss = torch.tensor(0.0, device=self.coords.device)
        for pooler in self.poolers:
            loss = loss + pooler.get_entropy_loss()
        return loss

    def get_consistency_loss(self):
        """Multi-scale consistency: coarse graph ≈ aggregated fine graph."""
        loss = torch.tensor(0.0, device=self.coords.device)
        for i in range(self.num_levels - 1):
            S_buf = getattr(self, f'structure_S_{i}', None)
            if S_buf is None or S_buf.sum() == 0:
                continue

            G_fine = self.layers[i].graph.get_soft_graph()
            G_coarse = self.layers[i + 1].graph.get_soft_graph()
            if G_fine is None or G_coarse is None:
                continue

            S_norm = S_buf / (S_buf.sum(dim=0, keepdim=True) + 1e-8)
            G_aggregated = S_norm.t() @ G_fine @ S_norm

            g_agg_max = G_aggregated.abs().max() + 1e-8
            g_coarse_max = G_coarse.abs().max() + 1e-8
            loss = loss + F.mse_loss(
                G_aggregated / g_agg_max,
                G_coarse.detach() / g_coarse_max
            )
        return loss

    def _get_knn_indices(self, coords, coarse_graph=None, structure_S=None, max_k=None):
        if max_k is None:
            max_k = self.max_k
        dists = torch.cdist(coords, coords)

        if (coarse_graph is not None) and (structure_S is not None):
            parents = structure_S.argmax(dim=1)
            coarse_bin = (coarse_graph > 0.001).float()
            prior_mask = coarse_bin[parents][:, parents]

            same_parent = (parents.unsqueeze(1) == parents.unsqueeze(0)).float()
            prior_mask = torch.max(prior_mask, same_parent)
            penalty = 1.0 + self.penalty_factor * (1.0 - prior_mask)
            dists = dists * penalty

        if self.training and coarse_graph is None:
            dists = dists + torch.randn_like(dists) * 0.01

        threshold = dists.mean() + self.penalty_factor * dists.std()
        dists = torch.where(
            dists <= threshold,
            dists,
            torch.tensor(float('inf'), device=dists.device)
        )

        k = min(max_k, coords.shape[0])
        _, indices = torch.topk(dists, k, dim=1, largest=False)
        return indices

    def forward(self, x):
        # Stage 1: Bottom-Up Aggregation
        xs, curr_coords = [x], self.coords
        coords_list, S_list = [curr_coords], []

        for i in range(self.num_levels - 1):
            S = self.poolers[i](xs[-1], curr_coords)
            S_list.append(S[0])
            with torch.no_grad():
                getattr(self, f'structure_S_{i}').copy_(S[0])

            S_norm = S[0] / (S[0].sum(0, keepdim=True) + 1e-6)
            x_next = torch.matmul(xs[-1].permute(0, 2, 1), S_norm).permute(0, 2, 1)
            xs.append(x_next)

            curr_coords = torch.mm(S_norm.t(), curr_coords)
            coords_list.append(curr_coords)

        # Stage 2: Top-Down Causal Discovery with Cross-Level Refinement
        results_dict = {}
        upper_level_graph = None
        upper_level_z = None

        for i in reversed(range(self.num_levels)):
            current_S = S_list[i] if i < len(S_list) else None

            mask = self._get_knn_indices(
                coords=coords_list[i],
                coarse_graph=upper_level_graph,
                structure_S=current_S,
                max_k=self.layers[i].graph.max_k
            )

            x_pred, z = self.layers[i](xs[i], mask=mask)

            if upper_level_z is not None and current_S is not None and i < len(self.bridges):
                z_refined = self.bridges[i](z, upper_level_z, current_S)
                zhat_refined = self.layers[i].graph(z_refined, neighbor_indices=mask)
                x_pred_refined = self.layers[i].pred_head(
                    zhat_refined[..., :-1, :].squeeze(-1)
                ).view(x_pred.shape)
                x_pred = x_pred + x_pred_refined

            upper_level_graph = self.layers[i].graph.get_soft_graph()
            upper_level_z = z

            results_dict[i] = {
                'level': i, 'x_pred': x_pred, 'x_target': xs[i],
                'S': current_S, 'k_used': coords_list[i].shape[0]
            }

        return [results_dict[i] for i in range(self.num_levels)]

    def forward_dynamic_sparse(self, x):
        """Fast sparse dynamic inference on the finest level."""
        B, N, T = x.shape
        layer0 = self.layers[0]

        mask = self._get_knn_indices(
            coords=self.coords, coarse_graph=None,
            structure_S=None, max_k=layer0.graph.max_k
        )

        z = layer0.geo_encoder(x.reshape(B * N, T, 1)).view(B, N, T, -1)
        indices, values = layer0.graph.get_sparse_dynamic_weights(z, mask)
        return indices, values
