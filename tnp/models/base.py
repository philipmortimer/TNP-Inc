from abc import ABC

import torch
from check_shapes import check_shapes
from torch import nn

from ..likelihoods.base import UniformMixtureLikelihood


class BaseNeuralProcess(nn.Module, ABC):
    """Represents a neural process base class"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood


class ConditionalNeuralProcess(BaseNeuralProcess):
    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))


class LatentNeuralProcess(BaseNeuralProcess):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, likelihood: nn.Module):
        likelihood = UniformMixtureLikelihood(likelihood)
        super().__init__(encoder, decoder, likelihood)

    @check_shapes("xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]")
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, num_samples: int = 1
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt, num_samples), xt))


class ARConditionalNeuralProcess(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "yt: [m, nt_, dy]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor,
    ) -> torch.distributions.Distribution:
        out_tmp = self.predict(xc, yc, xt, num_samples=3)
        print("Done")
        exit(0)

        return out_tmp
        if self.training:
            # Train in AR mode.
            return self.likelihood(self.decoder(self.encoder(xc, yc, xt, yt), xt))

        # Test in normal mode.
        dist_tmp = self.likelihood(self.decoder(self.encoder(xc, yc, xt, yt), xt)) # Delete later - for testing
        return dist_tmp
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))
