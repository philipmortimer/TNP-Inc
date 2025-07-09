# Autoregressive neural process - test time only.
# Based on https://arxiv.org/pdf/2303.14468 - takes a normal NP model and treats predicted target points as context points
# Inspired by https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional, Union, Literal
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch

class ARNP:
    def __init__(
        self,
        np_model: nn.Module,
    ):

    @check_shapes(
        "xt: [m, nt, dx]", "yt: [m, nt, dy]"
    )
    def shuffle_targets(xt, yt, order: Literal["random", "given", "left-to-right"]):
        m, nt, dx = xt.shape
        _, _, dy = yt.shape
        device = xt.device
        if order == "given":
            return xt, yt
        elif order == "random":
            perm = torch.rand(m, nt, device=device).argsort(dim=1)
            perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            xt_shuffled = torch.gather(xt, 1, perm_x)
            yt_shuffled = torch.gather(yt, 1, perm_y)
            return xt_shuffled, yt_shuffled
        elif order == "left-to-right"
            assert dx == 1, "left-to-right ordering only supported for one dimensional dx"
            perm = torch.argsort(xt.squeeze(-1), dim=1)
            perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            xt_sorted = torch.gather(xt, 1, perm_x)
            yt_sorted = torch.gather(yt, 1, perm_y)
            return xt_sorted, yt_sorted

    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]", "return: [m]"
    )
    @torch.no_grad
    def ar_loglik(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
        normalise: bool = True, order: Literal["random", "given", "left-to-right"] = "random" ) -> torch.Tensor:
        xt, yt = shuffle_targets(xt, yt, order)
        m, nt, dx = xt.shape
        _, nc, dy = yc.shape
        log_probs = torch.zeros((m), device=xt.device)
        for i in range(nt):
            # Sets context and target
            xt_sel = xt[:,i:i+1,:]
            yt_sel = yt[:,i:i+1,:]    
            xc_it = torch.cat((xc, xt[:, :i, :]), dim=1)
            yc_it = torch.cat((yc, yt[:, :i, :]), dim=1)
            batch = Batch(xc=xc_it, yc=yc_it, xt=xt_sel, yt=yt_sel, x=torch.cat((xc_it, xt_sel), dim=1), y=torch.cat((yc_it, yt_sel), dim=1))

            # Prediction + log prob
            pred_dist = np_pred_fn(self.np_model, batch)
            log_probs += pred_dist.log_prob(yt_sel).sum(-1, -2)
        if normalise:
            log_probs /= (nt * dy)
        return log_probs