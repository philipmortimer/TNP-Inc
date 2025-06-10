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
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # During training we use teacher forcing and have access to yt (train mode).
        # During evaluation (eval mode) yt is not provided and will be none.
        if yt is None:
            print("IMPLEMENT ME") # TODO add
        else:
            x_y_ctx = torch.cat((xc, batch.yc), dim=-1)
            x_0_tar = torch.cat((xt, torch.zeros_like(batch.yt)), dim=-1)

           # TODO Implement

        return zt

class TNP_AR(ARConditionalNeuralProcess):
    def __init__(
        self,
        encoder: ARTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)