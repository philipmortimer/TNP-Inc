import einops
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional

from ..networks.setconv import SetConvGridDecoder, SetConvGridEncoder
from .base import ConditionalNeuralProcess
from .tnp import TNPDecoder
from ..networks.fourier_embed import FourierEmbedderHadISD


class ConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        grid_encoder: SetConvGridEncoder,
        grid_decoder: SetConvGridDecoder,
        z_encoder: nn.Module,
        hadisd_mode: bool = False, 
        fourier_encoder: Optional[FourierEmbedderHadISD] = None,
    ):
        super().__init__()

        self.conv_net = conv_net
        self.grid_encoder = grid_encoder
        self.grid_decoder = grid_decoder
        self.z_encoder = z_encoder

        self.hadisd_mode = hadisd_mode # hadisd is special case
        self.fourier_encoder = fourier_encoder

    @check_shapes(
        "xc: [m, nc, dx]",
        "yc: [m, nc, dy]",
        "xt: [m, nt, dx]",
        "return: [m, nt, dz]",
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        if self.hadisd_mode: # handles this data type in a bespoke way - hacky but quick cos input is 4d
            flag = torch.ones_like(yc[..., :1])
            elev = xc[..., 2:3]
            time = xc[..., 3:4]
            # Passes these features into fourier embedder
            elev_time_vec = torch.cat((elev, time), dim=-1)
            elev_time_fourier = self.fourier_encoder(elev_time_vec)
            
            z_feats = torch.cat((yc, flag, elev_time_fourier), dim=-1) 
            xc_coords = xc[..., :2] # Cuts out time and elevation for CNN
            xt_coords = xt[..., :2]    

            x_grid, z_grid = self.grid_encoder(xc_coords, z_feats)
            z_grid = self.z_encoder(z_grid)
            z_grid = self.conv_net(z_grid)
            zt = self.grid_decoder(x_grid, z_grid, xt_coords)
            return zt
        else: # Original path
            # Add density.
            yc = torch.cat((yc, torch.ones(yc.shape[:-1] + (1,)).to(yc)), dim=-1)

            # Encode to grid.
            x_grid, z_grid = self.grid_encoder(xc, yc)

            # Encode to z.
            z_grid = self.z_encoder(z_grid)

            # Convolve.
            z_grid = self.conv_net(z_grid)

            # Decode.
            zt = self.grid_decoder(x_grid, z_grid, xt)
            return zt


class GriddedConvCNPEncoder(nn.Module):
    def __init__(
        self,
        conv_net: nn.Module,
        z_encoder: nn.Module,
    ):
        super().__init__()
        self.conv_net = conv_net
        self.z_encoder = z_encoder

    @check_shapes(
        "mc: [m, ...]",
        "y: [m, ..., dy]",
        "mt: [m, ...]",
        "return: [m, dt, dz]",
    )
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.Tensor:
        mc_ = einops.repeat(mc, "m n1 n2 -> m n1 n2 d", d=y.shape[-1])
        yc = y * mc_
        z_grid = torch.cat((yc, mc_), dim=-1)
        z_grid = self.z_encoder(z_grid)
        z_grid = self.conv_net(z_grid)
        zt = torch.stack([z_grid[i][mt[i]] for i in range(mt.shape[0])])
        return zt


class ConvCNP(ConditionalNeuralProcess):
    def __init__(
        self,
        encoder: ConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)
        self.likelihood.min_noise = 1e-5 #  Adds little noise here because sometimes scale is exactly 0 - check to ensure this is small enough to not impact other perf - hacky


class GriddedConvCNP(nn.Module):
    def __init__(
        self,
        encoder: GriddedConvCNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

    @check_shapes("mc: [m, ...]", "y: [m, ..., dy]", "mt: [m, ...]")
    def forward(
        self, mc: torch.Tensor, y: torch.Tensor, mt: torch.Tensor
    ) -> torch.distributions.Distribution:
        return self.likelihood(self.decoder(self.encoder(mc, y, mt)))
