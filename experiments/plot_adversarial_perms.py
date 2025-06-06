# Plotting script to plot various permutations of causal tnp - particularly highlight the extrema
import numpy as np
import torch
from scipy import stats
from check_shapes import check_shapes
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.lightning_utils import LitWrapper
import time
import warnings
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import RBFKernel
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import wandb
from tnp.data.base import Batch, GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch
from tnp.utils.np_functions import np_pred_fn
from typing import Callable, List, Tuple, Union, Optional
from torch import nn
import copy

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# Looks at results of random permutations of the context set
@check_shapes(
    "xc: [1, nc, dx]", "yc: [1, nc, dy]", "xt: [1, nt, dx]", "yt: [1, nt, dy]"
)
@torch.no_grad()
def gather_rand_perms(tnp_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, 
    no_permutations: int, device: str='cuda', batch_size: int=16):
    tot_time = time.time()
    inf_time = 0
    data_time = 0
    xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
    _, nc, dx = xc.shape
    perms_left = no_permutations
    log_p_list = []
    perm_list = []
    while perms_left > 0:
        # Batches permutations together to speed up computation
        data_start_time = time.time()
        batch_size_perm = min(batch_size, perms_left)

        # Permutations generated. Can use torch.randperm but torch.randn + argsort has lower constant (much faster) despite worse big O
        #perms = torch.stack([torch.randperm(nc, device=device) for _ in range(batch_size_perm)])
        keys = torch.randn(batch_size_perm, nc, device=device)
        perms = keys.argsort(dim=-1) 

        xc_perm_batched = xc.expand(batch_size_perm, -1, -1).gather(1, perms.unsqueeze(-1).expand(-1, -1, xc.shape[-1]))
        yc_perm_batched = yc.expand(batch_size_perm, -1, -1).gather(1, perms.unsqueeze(-1).expand(-1, -1, yc.shape[-1]))

        xt_batched = xt.expand(batch_size_perm, -1, -1)
        yt_batched = yt.expand(batch_size_perm, -1, -1)

        data_time += time.time() - data_start_time
        inf_start_time = time.time()
        log_p = tnp_model(xc_perm_batched, yc_perm_batched, xt_batched).log_prob(yt_batched).sum(dim=(-1, -2))
        inf_time += time.time() - inf_start_time

        log_p_list.append(log_p)
        perm_list.append(perms)
        perms_left -= batch_size_perm
    log_p = torch.cat(log_p_list)[:no_permutations]
    perms = torch.cat(perm_list)[:no_permutations]
    end_time = time.time()
    return perms, log_p, (data_time, inf_time, end_time - tot_time)


# Based off of plot.py but adapted for this case
def plot_perm(
    *,
    model: Union[nn.Module,
                 Callable[..., torch.distributions.Distribution]],
    xc: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    yt: torch.Tensor,
    perm: torch.Tensor,
    file_name: str,
    annotate: bool = True, # Number the points or not
    figsize: Tuple[float, float] = (8.0, 6.0),
    x_range: Tuple[float, float] = (-2.0, 2.0),
    y_lim: Tuple[float, float] = (-3.0, 3.0),
    points_per_dim: int = 64,
    savefig: bool = True,
    logging: bool = True,
    pred_fn: Callable = np_pred_fn,
    gt_pred: Optional[GroundTruthPredictor] = None,
):
    # Permutes context and converts everything to the same device
    device   = xc.device
    perm     = perm.to(device)
    xc_perm  = xc[:, perm, :]
    yc_perm  = yc[:, perm, :]
    model = model.to(device)

    # Generates batch synthetically
    batch = SyntheticBatch(xc=xc_perm, yc=yc_perm, xt=xt, yt=yt, x=torch.cat([xc_perm, xt], dim=1), 
        y=torch.cat([yc_perm, yt], dim=1), gt_pred=gt_pred)
    plot_batch = copy.deepcopy(batch)

    steps   = int(points_per_dim * (x_range[1] - x_range[0]))
    x_plot  = torch.linspace(*x_range, steps, device=device)[None, :, None]
    plot_batch.xt = x_plot

    with torch.no_grad():
        y_plot_pred_dist = pred_fn(model, plot_batch) # Gets model predictions over grid
        yt_pred_dist = pred_fn(model, batch) # Get model predictions for poiints
    model_nll = -yt_pred_dist.log_prob(yt).sum() / batch.yt[..., 0].numel()
    mean, std = y_plot_pred_dist.mean, y_plot_pred_dist.stddev

    # Make figure for plotting
    fig = plt.figure(figsize=figsize)
    # Plot context and target points
    x_ctx = xc_perm[0, :, 0].cpu()
    y_ctx = yc_perm[0, :, 0].cpu()
    plt.scatter(x_ctx, y_ctx, c="k", s=30, label="Context")
    # Labels context set ordering
    if annotate:
        for j, (xj, yj) in enumerate(zip(x_ctx, y_ctx), 1):
            plt.annotate(str(j), (xj, yj), textcoords="offset points", xytext=(5, 5), fontsize=15)

    plt.scatter(xt[0, :, 0].cpu(), yt[0, :, 0].cpu(), c="r", s=30, label="Target")

    # Plot model predictions
    plt.plot(
        x_plot[0, :, 0].cpu(),
        mean[0, :, 0].cpu(),
        c="tab:blue",
        lw=3,
    )

    plt.fill_between(
        x_plot[0, :, 0].cpu(),
        mean[0, :, 0].cpu() - 2.0 * std[0, :, 0].cpu(),
        mean[0, :, 0].cpu() + 2.0 * std[0, :, 0].cpu(),
        color="tab:blue",
        alpha=0.2,
        label="Model",
    )
    title_str = f"$N = {xc.shape[1]}$ NLL = {model_nll:.3f}"

    # Adds groundtruth
    if isinstance(batch, SyntheticBatch) and batch.gt_pred is not None:
        with torch.no_grad():
            gt_mean, gt_std, _ = batch.gt_pred(
                xc=xc,
                yc=yc,
                xt=x_plot,
            )
            _, _, gt_loglik = batch.gt_pred(
                xc=xc,
                yc=yc,
                xt=xt,
                yt=yt,
            )
            gt_nll = -gt_loglik.sum() / batch.yt[..., 0].numel()

        # Plot ground truth
        plt.plot(
            x_plot[0, :, 0].cpu(),
            gt_mean[0, :].cpu(),
            "--",
            color="tab:purple",
            lw=3,
        )

        plt.plot(
            x_plot[0, :, 0].cpu(),
            gt_mean[0, :].cpu() + 2 * gt_std[0, :].cpu(),
            "--",
            color="tab:purple",
            lw=3,
        )

        plt.plot(
            x_plot[0, :, 0].cpu(),
            gt_mean[0, :].cpu() - 2 * gt_std[0, :].cpu(),
            "--",
            color="tab:purple",
            label="Ground truth",
            lw=3,
        )

        title_str += f" GT NLL = {gt_nll:.3f}"

    plt.title(title_str, fontsize=24)
    plt.grid()

    # Set axis limits
    plt.xlim(x_range)
    plt.ylim(y_lim)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(loc="bottom right", fontsize=20)
    plt.tight_layout()

    fname = f"{file_name}.png"
    if wandb.run is not None and logging:
        wandb.log({fname: wandb.Image(fig)})
    elif savefig:
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


# Generates plots of permutations
@check_shapes(
    "perms: [K, nc]", "log_p: [K]", "xc: [1, nc, dx]", "yc: [1, nc, dy]", "xt: [1, nt, dx]", "yt: [1, nt, dy]"
)
def visualise_perms(tnp_model, perms: torch.tensor, log_p: torch.tensor, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, 
    folder_path: str="plot_results/adversarial", file_id: str=str(random.randint(0, 1000000)), gt_pred: Optional[GroundTruthPredictor] = None):
    log_p, indices = torch.sort(log_p)
    perms = perms[indices]
    #print(perms)
    #print(log_p)
    # Visualises permutations of various centiles
    perf_int = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    for perc in perf_int:
        perc_idx = round((perc / 100) * (len(perms) - 1))
        perm, log_prob = perms[perc_idx], log_p[perc_idx]
        file_name = f"{folder_path}/seq_perm_{perc:03d}_id_{file_id}"
        plot_perm(model=tnp_model, xc=xc, yc=yc, xt=xt, yt=yt, perm=perm, savefig=True, file_name=file_name, gt_pred=gt_pred)

    # Visualises distribution of permutations / LLs (TODO: implemeny maybe using kendall's tau)








if __name__ == "__main__":
    # E.g. run with: python experiments/plot_adversarial_perms.py --config experiments/configs/synthetic1d/gp_plain_tnp.yml
    experiment = initialize_experiment() # Gets config file
    model_arch = experiment.model # Gets type of model
    # RBF kernel params
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    # Data generator params
    nc, nt = 10, 100
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 1
    batch_size = 1
    deterministic = True
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=0.1,
        deterministic=True, kernel=kernels)
    data = next(iter(gen_val))
    tnp_model = None
    useWandb = True # Defines if weights and biases model is to be used
    wanddName = 'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200'
    wanddName = 'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v0'
    if useWandb:
        run = wandb.login()
        artifact = wandb.Api().artifact(wanddName, type='model')
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")
        print(ckpt_file)
        lit_model = (
            LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                ckpt_file, model=model_arch,
            )
        )
        tnp_model = lit_model.model
        tnp_model.eval()

    else:
        model_arch.to('cuda')
        model_arch.eval()
        tnp_model=model_arch
    print("Starting search")
    perms, log_p, (data_time, inference_time, total_time) = gather_rand_perms(tnp_model, data.xc, data.yc, data.xt, data.yt, 
        no_permutations=9000000, device='cuda', batch_size=1024)
    #print(log_p)
    print(f"Data time: {data_time:.2f}s, Inference time: {inference_time:.2f}s, Total time: {total_time:.2f}s")
    visualise_perms(tnp_model, perms, log_p, data.xc, data.yc, data.xt, data.yt,
        folder_path="plot_results/adversarial", file_id=str(random.randint(0, 1000000)), gt_pred=data.gt_pred)