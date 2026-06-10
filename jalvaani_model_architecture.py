"""
JalVaani AI — Day 2 Model Architecture
GroundwaterNet: deep MLP with a skip connection, for groundwater depth
prediction (mbgl). Standalone module so Day 3/4 can import it cleanly:

    from jalvaani_model_architecture import GroundwaterNet
"""
import torch
import torch.nn as nn

N_FEATURES = 45


class GroundwaterNet(nn.Module):
    """
    Input: 45 features (scaled)

    Block 1: Linear(45 -> 256)  + BatchNorm + ReLU + Dropout(0.3)
    Block 2: Linear(256 -> 128) + BatchNorm + ReLU + Dropout(0.2)
    Skip:    Block 1 output (256) concatenated to Block 2 output (128)
    Block 3: Linear(384 -> 64)  + BatchNorm + ReLU + Dropout(0.1)
    Block 4: Linear(64 -> 32)   + ReLU
    Output:  Linear(32 -> 1)
    """

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # skip connection: cat([block2_out(128), block1_out(256)]) = 384
        self.block3 = nn.Sequential(
            nn.Linear(384, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.block4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(torch.cat([b2, b1], dim=1))  # skip connection
        b4 = self.block4(b3)
        return self.head(b4)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CorrectedPhysicsLoss(nn.Module):
    """
    JalVaani Day 2 (corrected) physics-guided loss.

    L_total = L_pred + lambda_m * L_monotonic + lambda_s * L_spatial

    TERM 1 — L_pred: MSE on (scaled) predictions vs targets.

    TERM 2 — Monotonic depletion, WITHIN grid cell only:
      At the same ~0.1-degree grid cell, depth (mbgl) should increase
      over time. Samples are sorted by (grid_cell, year); only
      consecutive pairs in the SAME cell are penalized:
          violations = relu(pred[t] - pred[t+1])
          L_monotonic = mean(violations^2) over within-cell pairs
      Fix for v1: no longer compares unrelated stations across India.

    TERM 3 — Spatial smoothness, distance cutoff 0.5 degrees (~55 km):
      Only pairs with lat/lon distance < 0.5 are penalized:
          L_spatial = mean( (pred_i - pred_j)^2 / max(d_ij, 0.01) )
      Fix for v1: no penalty between far-apart batch neighbors.

    forward(pred, target, phys) where phys = raw (unscaled)
    [latitude, longitude, year_normalized] per sample.
    """

    def __init__(self, lambda_m: float = 0.01, lambda_s: float = 0.005,
                 radius: float = 0.5):
        super().__init__()
        self.lambda_m = lambda_m
        self.lambda_s = lambda_s
        self.radius = radius
        self.mse = nn.MSELoss()

    def forward(self, pred, target, phys):
        l_pred = self.mse(pred, target)
        p = pred.squeeze(1)
        zero = pred.new_zeros(())
        lat, lon, year = phys[:, 0], phys[:, 1], phys[:, 2]

        # TERM 2 — within-grid-cell monotonic depletion (vectorized:
        # lexicographic sort by (cell, year), penalize same-cell pairs)
        if self.lambda_m > 0 and p.numel() > 1:
            lat_r = torch.round(lat.double() * 10) / 10
            lon_r = torch.round(lon.double() * 10) / 10
            cell = lat_r * 1000 + lon_r          # unique id per 0.1-deg cell
            key = cell * 10 + year.double()      # sort by cell, then year
            order = torch.argsort(key)
            cell_s, p_s = cell[order], p[order]
            same_cell = cell_s[:-1] == cell_s[1:]
            if same_cell.any():
                viol = torch.relu(p_s[:-1] - p_s[1:])[same_cell]
                l_mono = (viol ** 2).mean()
            else:
                l_mono = zero
        else:
            l_mono = zero

        # TERM 3 — spatial smoothness with 0.5-degree cutoff
        if self.lambda_s > 0 and p.numel() > 1:
            latlon = torch.stack([lat, lon], dim=1)
            d = torch.cdist(latlon, latlon)
            mask = d < self.radius
            mask.fill_diagonal_(False)           # i != j
            if mask.any():
                diff2 = (p.unsqueeze(1) - p.unsqueeze(0)) ** 2
                w = 1.0 / d.clamp(min=0.01)
                l_spatial = (diff2 * w)[mask].mean()
            else:
                l_spatial = zero
        else:
            l_spatial = zero

        total = l_pred + self.lambda_m * l_mono + self.lambda_s * l_spatial
        return total, l_pred.detach(), l_mono.detach(), l_spatial.detach()
