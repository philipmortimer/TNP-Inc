import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..networks.deepset import DeepSet
from .base import ConditionalNeuralProcess
from .tnp import TNPDecoder


class CNPEncoder(nn.Module):
    def __init__(
        self,
        deepset: DeepSet,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.deepset = deepset
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, .]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        yc_encoded = self.y_encoder(yc)

        zc = self.deepset(xc_encoded, yc_encoded)

        # Use same context representation for every target point.
        zc = einops.repeat(zc, "m d -> m n d", n=xt.shape[-2])

        # Concatenate xt to zc.
        zc = torch.cat((zc, xt_encoded), dim=-1)

        return zc


class CNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: CNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
