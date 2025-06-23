#incTNP with batching strategy explored. This variant supports a start token (allowing for conditioning on empty context)
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import TNPTransformerFullyMaskedEncoder
from ..utils.helpers import preprocess_observations
from .base import BatchedCausalTNP
from .tnp import TNPDecoder
from ..utils.helpers import preprocess_observations


class IncTNPBatchedEncoderPrior(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerFullyMaskedEncoder],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
        embed_dim: int, # This is dz and is used for the learnable empty token
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

        # Learnable empty token used to represent start / no context
        self.empty_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.empty_token, mean=0.0, std=0.02)

    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]",
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]",
        "return: [m, n_t_or_n_minus_one, dz]",
    )
    def forward(
        self, x: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None,
        xc: Optional[torch.Tensor] = None, yc: Optional[torch.Tensor] = None, xt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Checks that it either provides (x,y) OR (xc, yc, xt) but not both. This is used to determine whether train / prediction is happening
        assert (xc is None and yc is None and xt is None and y is not None and x is not None) or (xc is not None and yc is not None and xt is not None and x is None and y is None), "Invalid encoder call. Can't differentiate between prediction or training call"

        if x is not None and y is not None: return self.train_encoder(x, y)
        else: return self.predict_encoder(xc, yc , xt)

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def predict_encoder(self, xc: torch.Tensor, yc:torch.Tensor, xt:torch.Tensor):
        # At prediction time we essentially become identically to incTNP basic
        # (I.e.) just self attention over the context points and no cross attention mask.
        m, nc, _ = xc.shape
        yc, yt = preprocess_observations(xt, yc)

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zt = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        mask_sa = torch.tril(torch.ones(nc, nc, dtype=torch.bool, device=zc.device), diagonal=0)
        mask_sa = mask_sa.unsqueeze(0).expand(m, -1, -1) # [m, n, n]

        zt = self.transformer_encoder(zc, zt, mask_sa=mask_sa, mask_ca=None)
        
        return zt       


    @check_shapes(
        "x: [m, n, dx]", "y: [m, n, dy]","return: [m, n_minus_one, dz]"
    )
    def train_encoder(self, x: torch.Tensor, y:torch.Tensor):
        m, n, dy = y.shape
        # Treats sequence as just x and y. y_tgt is set to just be 0s to
        y_like = torch.zeros((m, n, dy)).to(y)
        y_tgt = torch.cat((y_like, torch.ones(y_like.shape[:-1] + (1,)).to(y)), dim=-1)

        y_ctx = torch.cat((y, torch.zeros(y.shape[:-1] + (1,)).to(y)), dim=-1)

        # Encodes x and y
        x_encoded = self.x_encoder(x)
        x_tgt_encoded = x_encoded
        y_ctx_encoded = self.y_encoder(y_ctx)
        y_tgt_encoded = self.y_encoder(y_tgt)

        # Embeds data
        zc = torch.cat((x_encoded, y_ctx_encoded), dim=-1)
        zt = torch.cat((x_tgt_encoded, y_tgt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        # Adds dummy start token to zc
        self.empty_token.expand(m, -1, -1)
        zc = torch.cat((self.empty_token, zc), dim=1)

        # Creates masks. 
        # A target point can only attend to preceding context points (plus dummy token)
        mask_ca = torch.tril(torch.ones(n, n + 1, dtype=torch.bool, device=zc.device), diagonal=0)
        mask_ca = mask_ca.unsqueeze(0).expand(m, -1, -1) # [m, n + 1, n]
        # Causal masking for context -> a context point can only attend to itself and previous context points (including dummy token).
        mask_sa = torch.tril(torch.ones(n + 1, n + 1, dtype=torch.bool, device=zc.device), diagonal=0)
        mask_sa = mask_sa.unsqueeze(0).expand(m, -1, -1) # [m, n + 1, n + 1]

        zt = self.transformer_encoder(zc, zt, mask_sa=mask_sa, mask_ca=mask_ca)
        
        assert len(zt.shape) == 3 and zt.shape[0] == m and zt.shape[1] == n, "Return encoder shape wrong"
        return zt



class IncTNPBatchedPrior(BatchedCausalTNP):
    def __init__(
        self,
        encoder: IncTNPBatchedEncoderPrior,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
