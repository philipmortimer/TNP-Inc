# Autoregressive TNP - using the outline from Nguyen and Grover (https://arxiv.org/pdf/2207.04179) https://github.com/tung-nd/TNP-pytorch
# But merging with the style from this code base for baseline TNPs https://github.com/cambridge-mlg/tnp
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from .tnp import TNPDecoder
from ..utils.helpers import preprocess_observations
from ..networks.transformer import ISTEncoder, PerceiverEncoder, TNPTransformerMaskedEncoder


class ARTNPEncoder(nn.module):
    def __init__(
        self,
        transformer_encoder: Union[TNPTransformerMaskedEncoder, PerceiverEncoder, ISTEncoder],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

        if not isinstance(self.transformer_encoder, TNPTransformerMaskedEncoder): # TODO: add support for perceiver encoder and IST encoder
            warnings.warn("Perceiver Encoder and IST Encoder not currently supported for autoreg TNP encoder as cant do masking.")

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]?", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        

        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)

        zc = torch.cat((xc_encoded, yc_encoded), dim=-1)
        zt0 = torch.cat((xt_encoded, yt_encoded), dim=-1)
        zc = self.xy_encoder(zc)
        zt0 = self.xy_encoder(zt0)

        # Constructs mask so that target points attend to all previous targets and the whole context set

        
        zt = self.transformer_encoder(zc, zt0, mask=ar_mask)
        return zt

class TNP_AR(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: ARTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)