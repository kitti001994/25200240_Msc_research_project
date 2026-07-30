"""
ST-GCN + FiLM model

Architecture:
Input: X (B, N, W, 1) - parcel BOLD windows
       cond (B, W, C) - HRF-convolved condition regressors

    STGCNBlock x 3                                     
        1. GCNConv (spatial) -> (B·W, N, C_out)        
        2. FiLM modulation <- condition vector
        3. Depthwise Conv1d (temporal) -> (B, N, W, C_out)
        4. ReLU + optional residual

  Global mean pooling over N and W -> (B, C_last)
  Dropout -> Linear -> Sigmoid -> (B, 1) [autistic probability]

FiLM:
For each STGCNBlock the condition regressors (B, W, C) are passed through
a small MLP that predicts Gamma and shift Beta parameters.
It's applied to the spatial GCN output before temporal convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv

import config


# FiLM layer
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    Generates time-varying scale (Gamma) and shift (Beta) parameters from the
    HRF-convolved condition regressors and applies them to the spatial
    GCN output.
    """

    def __init__(self, channels: int, cond_dim: int, hidden_dim: int = config.FILM_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, channels * 2)
        self._init_weights()

    def _init_weights(self):
        # Initialise so that gamma ~ 0 and Beta ~ 0 at the start -> identity transform
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters: x : (B, N, W, C) - post-GCN feature map
                    cond : (B, W, cond_dim)

        Returns: (B, N, W, C) - modulated feature map
        """
        h = F.gelu(self.fc1(cond)) # (B, W, hidden)
        params = self.fc2(h) # (B, W, 2·C)
        gamma, beta = torch.chunk(params, 2, dim=-1) # each (B, W, C)

        # Broadcast over the node dimension N
        gamma = gamma.unsqueeze(1) # (B, 1, W, C)
        beta  = beta.unsqueeze(1)

        # Residual FiLM: x * (1 + Gamma) + Beta
        return x * (1.0 + gamma) + beta


# ST-GCN Block
class STGCNBlock(nn.Module):
    """
    One Spatial-Temporal GCN block with FiLM conditioning.

    Processing order:
        spatial GCN -> FiLM -> temporal depthwise conv -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        cond_dim: int,
        dropout: float = config.DROPOUT,
        use_film: bool = True,
    ):
        super().__init__()
        self.use_film = use_film
        self.register_buffer("edge_index",  edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # Spatial graph convolution (applied each time step)
        self.gcn = GCNConv(in_channels, out_channels, normalize=True, add_self_loops=False)

        # FiLM modulation
        self.film = FiLM(out_channels, cond_dim) if use_film else None

        # Temporal depthwise convolution (per-node, across time)
        padding = kernel_size // 2
        self.temporal = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_channels, # depthwise
        )
        self.bn = nn.BatchNorm1d(out_channels)

        # Residual projection
        self.residual = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters: x : (B, N, W, C_in)
                    cond : (B, W, cond_dim)

        Returns: (B, N, W, C_out)
        """
        B, N, W, C_in = x.shape
        residual = self.residual(x) # (B, N, W, C_out)

        # 1. Spatial GCN
        x_s = x.permute(0, 2, 1, 3) # (B, W, N, C_in)
        x_s = x_s.reshape(B * W, N, C_in) # (B*W, N, C_in)
        x_s = self.gcn(x_s, self.edge_index, self.edge_weight)
        # (B·W, N, C_out)
        C_out = x_s.shape[-1]
        x_s = x_s.view(B, W, N, C_out).permute(0, 2, 1, 3) # (B, N, W, C_out)

        # 2. FiLM modulation
        if self.use_film:
            x_s = self.film(x_s, cond) # (B, N, W, C_out)

        # 3. Temporal depthwise convolution (per node)
        x_t = x_s.permute(0, 1, 3, 2) # (B, N, C_out, W)
        x_t = x_t.reshape(B * N, C_out, W)
        x_t = self.temporal(x_t) # (B·N, C_out, W)
        x_t = self.bn(x_t)
        x_t = x_t.view(B, N, C_out, W).permute(0, 1, 3, 2) # (B, N, W, C_out)

        # 4. Residual + activation
        out = self.relu(x_t + residual)
        out = self.dropout(out)

        return out


# Full model
class STGCN_FiLM(nn.Module):

    def __init__(
        self,
        n_parcels: int,
        window_trs: int,
        in_channels: int,
        hidden_channels: list,
        temporal_kernel: int,
        cond_dim: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        dropout: float = config.DROPOUT,
        use_film: bool = True,
        n_static_features: int = 0,
    ):
        super().__init__()

        self.n_parcels = n_parcels
        self.window_trs = window_trs
        self.use_film = use_film
        self.n_static_features = n_static_features

        # Input -> expand from 1 channel to first hidden size
        in_ch = in_channels
        self.blocks = nn.ModuleList()
        for out_ch in hidden_channels:
            self.blocks.append(
                STGCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=temporal_kernel,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    cond_dim=cond_dim,
                    dropout=dropout,
                    use_film=use_film,
                )
            )
            in_ch = out_ch

        last_ch = hidden_channels[-1]

        # Classifier input size: graph embedding + optional static features.
        # static features (age, gender, FSIQ) are concatenated after
        # global pooling: h = [mean_pool(x) || static_feat]
        cls_in     = last_ch + n_static_features
        cls_hidden = max(cls_in // 2, 16) # scale with combined input width
        self.pool_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1), # binary logit
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        static_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters: x : (B, N, W, 1) - parcel BOLD windows
                    cond : (B, W, C) - HRF-convolved condition regressors
                    static_feat : (B, F) or None - z-scored demographic features.
                    Concatenated to the graph embedding after global pooling.
                    When n_static_features > 0.
        Returns: logits : (B,) - for BCEWithLogitsLoss
        """
        for block in self.blocks:
            x = block(x, cond) # (B, N, W, C_out)

        # Global average pool over nodes and time -> (B, C_last)
        x = x.mean(dim=[1, 2])

        # static demographic features
        # h = [mean_pool(BOLD) || static_feat]
        if self.n_static_features > 0 and static_feat is not None:
            x = torch.cat([x, static_feat], dim=-1) # (B, C_last + F)

        x = self.pool_dropout(x)
        logits = self.classifier(x).squeeze(-1) # (B,)
        return logits

def build_model(edge_index, edge_weight, n_parcels=None, n_static_features=0):
    n_parcels = n_parcels or config.N_PARCELS
    return STGCN_FiLM(
        n_parcels=n_parcels,
        window_trs=config.WINDOW_TRS,
        in_channels=config.IN_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        temporal_kernel=config.TEMPORAL_KERNEL,
        cond_dim=config.N_CONDITIONS,
        edge_index=edge_index,
        edge_weight=edge_weight,
        dropout=config.DROPOUT,
        use_film=True,
        n_static_features=n_static_features,
    )


def build_model_no_film(edge_index, edge_weight, n_parcels=None, n_static_features=0):
    """
    ST-GCN without FiLM.
    """
    n_parcels = n_parcels or config.N_PARCELS
    return STGCN_FiLM(
        n_parcels=n_parcels,
        window_trs=config.WINDOW_TRS,
        in_channels=config.IN_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        temporal_kernel=config.TEMPORAL_KERNEL,
        cond_dim=config.N_CONDITIONS,
        edge_index=edge_index,
        edge_weight=edge_weight,
        dropout=config.DROPOUT,
        use_film=False,
        n_static_features=n_static_features,
    )


def load_model_from_checkpoint(checkpoint_path, edge_index, edge_weight, device, n_parcels=None):
    """
    Load the model from checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    n_p = n_parcels or ckpt.get("n_parcels", config.N_PARCELS)
    n_static = ckpt.get("n_static_features", 0)
    if ckpt.get("use_film", True):
        model = build_model(
            edge_index, edge_weight, n_parcels=n_p,
            n_static_features=n_static,
        ).to(device)
    else:
        model = build_model_no_film(
            edge_index, edge_weight, n_parcels=n_p,
            n_static_features=n_static,
        ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
