#incTNP with batching strategy explored
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TNPTransformerFullyMaskedEncoder
from ..utils.helpers import preprocess_observations
from .base import BatchedCausalTNP
from .tnp import TNPDecoder


class IncTNPBatchedEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerFullyMaskedEncoder],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]", "return: [m, n, dz]"
    )
    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        m, n, _ = x.shape
        # Treats sequence as just x and y. y_tgt is set to just be 0s to
        y_like = torch.zeros(y.shape).to(y)
        y_tgt = torch.cat((y_like, torch.ones(y.shape[:-1] + (1,)).to(y)), dim=-1)
        y_ctx = torch.cat((y, torch.zeros(y.shape[:-1] + (1,)).to(y)), dim=-1)

        # Encodes x and y
        x_encoded = self.x_encoder(x)
        y_ctx_encoded = self.y_encoder(y_ctx)
        y_tgt_encoded = self.y_encoder(y_tgt)

        # Embeds data
        zc = torch.cat((x_encoded, y_ctx_encoded), dim=-1)
        zt = torch.cat((x_encoded, y_tgt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        # Creates masks. 
        # A target point can only attend to preceding context points.
        mask_ca = torch.tril(torch.ones(n, n, dtype=torch.bool, device=zc.device), diagonal=-1)
        mask_ca = mask_ca.unsqueeze(0).expand(m, -1, -1) # [m, n, n]
        # Causal masking for context -> a context point can only attend to itself and previous context points.
        mask_sa = torch.tril(torch.ones(n, n, dtype=torch.bool, device=zc.device), diagonal=0)
        mask_sa = mask_sa.unsqueeze(0).expand(m, -1, -1) # [m, n, n]

        zt = self.transformer_encoder(zc, zt, mask_sa=mask_sa, mask_ca=mask_ca)
        # Note - may want to slice zt to zt[:, 1:, :] because we don't use p(y_0) in loss (zero shot case)
        return zt



class IncTNPBatched(BatchedCausalTNP):
    def __init__(
        self,
        encoder: IncTNPBatchedEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
