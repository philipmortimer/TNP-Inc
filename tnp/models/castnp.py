from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.transformer import ISTEncoder, PerceiverEncoder, TNPTransformerMaskedEncoder
from ..utils.helpers import preprocess_observations
from .base import ConditionalNeuralProcess
from .tnp import TNPDecoder
import warnings

# TNP using causal attention mask - breaking context permutation invariance
class TNPEncoderMasked(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerMaskedEncoder, PerceiverEncoder, ISTEncoder],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

        if not isinstance(self.transformer_encoder, TNPTransformerMaskedEncoder): # TODO: add support for perceiver encoder and IST encoder
            warnings.warn("Perceiver Encoder and IST Encoder not currently supported for masked TNP encoder.")

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
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

        # Causal masked attention for context - using mask=None will get same behaviour as non masked
        #m, nc, _ = xc.shape # Number of context points
        #causal_mask = nn.Transformer.generate_square_subsequent_mask(nc, device=zc.device)
        #causal_mask = causal_mask.unsqueeze(0).expand(m, -1, -1).contiguous() # [m, nc, nc]
        zt = self.transformer_encoder(zc, zt)
        return zt


class TNPCausal(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: TNPEncoderMasked,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
