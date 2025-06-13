import torch
from torch import nn

from ..data.base import Batch, ImageBatch
from ..models.base import (
    ARConditionalNeuralProcess,
    ConditionalNeuralProcess,
    LatentNeuralProcess,
    ARTNPNeuralProcess,
)
from ..models.convcnp import GriddedConvCNP


def np_pred_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
    predict_without_yt_tnpa: bool = False, # Used for tnpa to allow teacher forcing by default but support predictions without access to yt
) -> torch.distributions.Distribution:
    if isinstance(model, GriddedConvCNP):
        assert isinstance(batch, ImageBatch)
        pred_dist = model(mc=batch.mc_grid, y=batch.y_grid, mt=batch.mt_grid)
    elif isinstance(model, ConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt)
    elif isinstance(model, LatentNeuralProcess):
        pred_dist = model(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, num_samples=num_samples
        )
    elif isinstance(model, ARConditionalNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt)
    elif isinstance(model, ARTNPNeuralProcess):
        pred_dist = model(xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt, predict_without_yt_tnpa=predict_without_yt_tnpa)
    else:
        raise ValueError

    return pred_dist


def np_loss_fn(
    model: nn.Module,
    batch: Batch,
    num_samples: int = 1,
) -> torch.Tensor:
    """Perform a single training step, returning the loss, i.e.
    the negative log likelihood.

    Arguments:
        model: model to train.
        batch: batch of data.

    Returns:
        loss: average negative log likelihood.
    """
    pred_dist = np_pred_fn(model, batch, num_samples)
    loglik = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()

    return -loglik
