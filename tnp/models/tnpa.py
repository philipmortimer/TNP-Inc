# Autoregressive TNP - using the outline from Nguyen and Grover (https://arxiv.org/pdf/2207.04179) https://github.com/tung-nd/TNP-pytorch
# But merging with the style from this code base for baseline TNPs https://github.com/cambridge-mlg/tnp
# Review original TNP codebase for optional tweaks not included here to tnpa
from typing import Optional, Union

import torch
from check_shapes import check_shapes
from torch import nn

from .tnp import TNPDecoder
from ..utils.helpers import preprocess_observations
from ..networks.transformer import TransformerEncoder
from .base import ARConditionalNeuralProcess


class ARTNPEncoder(nn.Module):
    def __init__(
        self,
        transformer_encoder: Union[TransformerEncoder],
        xy_encoder: nn.Module,
        x_encoder: nn.Module = nn.Identity(),
        y_encoder: nn.Module = nn.Identity(),
        bound_std: bool = False, # Smooths history - note no support for pretraining here at the moment. Doesn't quite support all same function as in tnpa from original codebase.
    ):
        super().__init__()

        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.bound_std = bound_std

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]", "return: [m, n, dz]"
    )
    def forward(
        self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert ((yt == None) != self.training), "Debug statement ensuring yt matches training mode expectation"
        # During training we use teacher forcing and have access to yt (train mode).
        # During evaluation (eval mode) yt is not provided and will be none.
        if yt is None:
            print("IMPLEMENT ME") # TODO add
            exit(0)
        else:
            return self._forward_train_pass(xc, yc, xt, yt)

    def _predict():
        

    # Does a single training pass using teacher forcing
    def _forward_train_pass(self, xc, yc, xt, yt):
        m = xc.shape[0]
        # Preprocesses observations
        yc = torch.cat((yc, torch.zeros(yc.shape[:-1] + (1,)).to(yc)), dim=-1)
        yt = torch.cat((yt, torch.ones(yt.shape[:-1] + (1,)).to(yt)), dim=-1)

        # Encodes x and y
        x = torch.cat((xc, xt), dim=1)
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = x_encoded.split((xc.shape[1], xt.shape[1]), dim=1)

        y = torch.cat((yc, yt), dim=1)
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = y_encoded.split((yc.shape[1], yt.shape[1]), dim=1)
        
        # Concats ctx with fake and real target points
        inp = self._construct_input(xc_encoded, yc_encoded, xt_encoded, yt_encoded)
        mask, num_tar = self._create_mask(num_ctx=xc.shape[1], num_tar=xt.shape[1])
        mask = mask.unsqueeze(0).expand(m, -1, -1) # [m, nc + 2*nt, nc+2*nt]  Broadcast mask to batch

        # Embeds data and runs through transformer encoder
        embeddings = self.xy_encoder(inp)
        out = self.transformer_encoder(embeddings, mask=mask)

        return out[:, -num_tar:]

    def _construct_input(self, xc, yc, xt, yt):
        x_y_ctx = torch.cat((xc, yc), dim=-1)
        x_0_tar = torch.cat((xt, torch.zeros_like(yt)), dim=-1)
        if self.training and self.bound_std:
            yt_noise = yt + 0.05 * torch.randn_like(yt) # add noise to the past to smooth the model
            x_y_tar = torch.cat((xt, yt_noise), dim=-1)
        else:
            x_y_tar = torch.cat((xt, yt), dim=-1)
        inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1) # [m, nc + 2*nt, dx + dy + 1] (probably - depends on encoders etc)
        return inp

    def _create_mask(self, num_ctx, num_tar):
        num_all = num_ctx + num_tar
        mask = torch.zeros((num_all+num_tar, num_all+num_tar), device='cuda').fill_(float('-inf'))
        mask[:, :num_ctx] = 0.0 # all points attend to context points
        mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1) # each real target point attends to itself and precedding real target points
        mask[num_all:, num_ctx:num_all].triu_(diagonal=0) # each fake target point attends to preceeding real target points

        return mask, num_tar

class TNPA(ARConditionalNeuralProcess):
    def __init__(
        self,
        encoder: ARTNPEncoder,
        decoder: TNPDecoder,
        likelihood: nn.Module,
    ):
        super().__init__(encoder, decoder, likelihood)