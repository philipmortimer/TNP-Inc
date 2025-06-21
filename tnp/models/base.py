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
        if self.training:
            # Train in AR mode.
            return self.likelihood(self.decoder(self.encoder(xc, yc, xt, yt), xt))

        # Test in normal mode.
        return self.likelihood(self.decoder(self.encoder(xc, yc, xt), xt))

# Used specifically for tnpa only at the moment
class ARTNPNeuralProcess(BaseNeuralProcess):
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
        predict_without_yt_tnpa: bool,
    ) -> torch.distributions.Distribution:
        if predict_without_yt_tnpa:
            # Uses a monte carlo sampled distribution (unrolling over multiple samples)
            return self.predictive_distribution_monte_carlo(xc, yc, xt)
        else:
            # Uses teach forcing setup
            return self.likelihood(self.decoder(self.encoder(xc, yc, xt, yt), xt))

# Used for the batched causal TNP
class BatchedCausalTNP(BaseNeuralProcess):
    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "yt: [m, nt, dy]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor,
    ) -> torch.distributions.Distribution:
        x = torch.cat((xc, xt), dim=1)
        y = torch.cat((yc, yt), dim=1)
        # AR style training
        if self.training:
            # Decoder doesnt look at zero shot context (first point)
            return self.likelihood(self.decoder(self.encoder(x=x, y=y)))
        else: # Inference time
            return self.likelihood(self.decoder(self.encoder(xc=xc, yc=yc, xt=xt), xt))
        
