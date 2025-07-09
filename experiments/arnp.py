# Autoregressive neural process - test time only.
# Based on https://arxiv.org/pdf/2303.14468 - takes a normal NP model and treats predicted target points as context points
# Inspired by https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional, Union, Literal
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch
from plot_adversarial_perms import get_model
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import RBFKernel
from functools import partial
from tqdm import tqdm
import numpy as np


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]",
)
@torch.no_grad
def _shuffle_targets(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
    order: Literal["random", "given", "left-to-right", "variance"]):
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
    elif order == "left-to-right":
        assert dx == 1, "left-to-right ordering only supported for one dimensional dx"
        perm = torch.argsort(xt.squeeze(-1), dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
        xt_sorted = torch.gather(xt, 1, perm_x)
        yt_sorted = torch.gather(yt, 1, perm_y)
        return xt_sorted, yt_sorted
    elif order == "variance":
        # Predicts all target points conditioned on context points and orders (highest variance first) - this is obviously much more expensive
        batch = Batch(xc=xc, yc=yc, xt=xt, yt=yt, x=torch.cat((xc, xt), dim=1), y=torch.cat((yc, yt), dim=1))
        pred_dist = np_pred_fn(np_model, batch)
        var = pred_dist.variance.mean(-1) # Gets variance (averaged over dy) [m, nt]
        perm = torch.argsort(var, dim=1, descending=True)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
        xt_sorted = torch.gather(xt, 1, perm_x)
        yt_sorted = torch.gather(yt, 1, perm_y)
        return xt_sorted, yt_sorted



@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]", "return: [m]"
)
@torch.no_grad
def ar_loglik(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
    normalise: bool = True, order: Literal["random", "given", "left-to-right", "variance"] = "random") -> torch.Tensor:
    xt, yt = _shuffle_targets(np_model, xc, yc, xt, yt, order)
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
        pred_dist = np_pred_fn(np_model, batch)
        log_probs += pred_dist.log_prob(yt_sel).sum(dim=(-1, -2))
    if normalise:
        log_probs /= (nt * dy)
    return log_probs


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
)
@torch.no_grad
def ar_predict(model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,
    order: Literal["random", "given", "left-to-right", "variance"] = "random"):
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape




# -------------------------------------------------------------------------------------------------------

# Compares NP models in AR mode on RBF set
def compare_rbf_models(base_out_txt_file: str, device: str = "cuda"):
    # Hypers to select - also look at dataset hypers
    ordering = "variance"
    # End of hypers
    # List of models to compare
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml',
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-a3qwpptn:v200', 'TNP-D')
    incTNP = ('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-8mxfyfnw:v200', 'incTNP')
    batchedTNP = ('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', 'incTNP-Batched')
    priorBatched = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'incTNP-Batched (Prior)')
    cnp = ('experiments/configs/synthetic1dRBF/gp_cnp_rangesame.yml',
        'pm846-university-of-cambridge/cnp-rbf-rangesame/model-uywfyrx7:v200', 'CNP')
    conv_cnp = ('experiments/configs/synthetic1dRBF/gp_convcnp_rangesame.yml',
        'pm846-university-of-cambridge/convcnp-rbf-rangesame/model-uj54q1ya:v200', 'ConvCNP')
    models = [tnp_plain, incTNP, batchedTNP, priorBatched, cnp, conv_cnp]
    # RBF Dataset
    min_nc = 1
    max_nc = 64
    nt= 128
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4_096
    batch_size = 16
    noise_std = 0.1
    deterministic = True
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    data = list(gen_test)
    # Main loop - loads each model than compares writes performances to a text file
    out_txt = ""
    for (model_yml, model_wab, model_name) in models:
        ll_list = []
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        for batch in tqdm(data, desc=f'{model_name} eval'):
            ll = ar_loglik(np_model=model, xc=batch.xc.to(device), yc=batch.yc.to(device),
                xt=batch.xt.to(device), yt=batch.yt.to(device), normalise=True, order=ordering)
            mean_ll = torch.mean(ll).item() # Goes from [m] to a float
            ll_list.append(mean_ll)
        ll_average = np.mean(ll_list)
        mod_sum = ("-" * 20) + f"\nModel: {model_name}\nMean LL: {mean_ll}\n"
        print(mod_sum)
        out_txt += mod_sum
    with open(out_txt_file + f'_{ordering}.txt', 'w') as file:
        file.write(out_txt)


if __name__ == "__main__":
    compare_rbf_models(base_out_txt_file="experiments/plot_results/ar/ar_rbf_comp")