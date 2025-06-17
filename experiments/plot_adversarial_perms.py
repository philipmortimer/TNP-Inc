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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import random
import os
import wandb
from tnp.data.base import Batch, GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch
from tnp.utils.np_functions import np_pred_fn, np_loss_fn
from typing import Callable, List, Tuple, Union, Optional
from torch import nn
import copy
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import hiyapyco
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tnp.utils.experiment_utils import deep_convert_dict, extract_config


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
        nt, dy = yt_batched.shape[-2:]
        log_p = tnp_model(xc_perm_batched, yc_perm_batched, xt_batched).log_prob(yt_batched).sum(dim=(-1, -2)) / (nt * dy)

        #batch = SyntheticBatch(xc=xc_perm_batched, yc=yc_perm_batched, xt=xt_batched, yt=yt_batched, 
        #    x=torch.cat([xc_perm_batched, xt_batched], dim=1), y=torch.cat([yc_perm_batched, yt_batched], dim=1))
        #yt_pred_dist = np_pred_fn(tnp_model, batch)
        #log_p = -np_loss_fn(tnp_model, batch)
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

    plt.legend(fontsize=20)
    plt.tight_layout()

    fname = f"{file_name}.png"
    if wandb.run is not None and logging:
        wandb.log({fname: wandb.Image(fig)})
    elif savefig:
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()

    plt.close()

# Gets a subset of perms and log_p evenly spaced across all sampled perms
def get_spaced_examples( perms: torch.Tensor, log_p: torch.Tensor, max_perms_plot: int = 20) -> (torch.Tensor, torch.Tensor):
    K, nc = perms.shape
    # Selects subset of lines if required
    if K > max_perms_plot:
        indices_plot = torch.linspace(0, K - 1, steps=max_perms_plot).long() # Every nth line - so we get wide range
        #indices_plot = torch.randperm(K)[:max_perms_plot] # Random selection of lines
        # Select the evenly spaced permutations and their log probabilities
        perms = perms[indices_plot]
        log_p = log_p[indices_plot]  
    return (perms, log_p) 

# Gets the best and worst perms only to plot
def get_best_and_worst( perms: torch.Tensor, log_p: torch.Tensor, top_and_bottom_n: int = 2) -> (torch.Tensor, torch.Tensor):
    K, nc = perms.shape
    if K < top_and_bottom_n * 2:# Too few perms
        return (perms, log_p)
    log_p_new =  torch.cat((log_p[:top_and_bottom_n], log_p[-top_and_bottom_n:]))
    perms_new =  torch.cat((perms[:top_and_bottom_n,:], perms[-top_and_bottom_n:,:]))
    return (perms_new, log_p_new) 


def plot_parallel_coordinates_bezier(
    perms: torch.Tensor,
    log_p: torch.Tensor,
    xc: torch.Tensor,
    xt: torch.Tensor,
    file_name: str,
    curvature_strength: float = 0.2,
    alpha_line: float = 0.4,
    plot_targets: bool = False,
):
    K, nc = perms.shape

    # Convert to cpu
    perms = perms.cpu()
    log_p = log_p.cpu()
    xc = xc.squeeze().cpu()
    xt = xt.squeeze().cpu()
    

    # Colourmap
    sm = ScalarMappable(cmap=plt.get_cmap('plasma'), norm=Normalize(vmin=log_p.min(), vmax=log_p.max()))
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(15, 10))
    positions = [i for i in range(nc)]

    # Plots each permutation to graph
    perm_xs = xc[perms] # [max_perms_plot, nc]
    for i in range(perm_xs.shape[0]):
        line_color = sm.to_rgba(log_p[i])
        points = np.column_stack((positions, perm_xs[i])) # Shape [nc, 2]

        # Bulds a path by introducing a random point to define a curve for the two lines
        path_cmds = [mpath.Path.MOVETO]
        path_pts = [points[0]]
        # Plots lines between points - using bezier curves to enable seeing different lines that go between same two points
        for j in range(len(points) - 1):
            p0 = points[j]
            p1 = points[j+1]
            
            midpoint = (p0 + p1) / 2.0
            perp_vec = np.array([-(p1[1] - p0[1]), p1[0] - p0[0]]) # Perpendicular vector to p1 - p0
            norm = np.linalg.norm(perp_vec)
            perp_vec = perp_vec / norm
            random_offset = (np.random.rand() - 0.5) * 2 * curvature_strength
            control_point = midpoint + perp_vec * random_offset # Defines offset point for curve
            path_cmds.extend([mpath.Path.CURVE3, mpath.Path.CURVE3])
            path_pts.extend([control_point, p1])

        path = mpath.Path(path_pts, path_cmds)
        patch = mpatches.PathPatch(
            path, 
            facecolor='none', 
            edgecolor=line_color, 
            linewidth=1.0,
            alpha=alpha_line
        )
        ax.add_patch(patch)

    # Grid of black dots to clearly show where context points are - overlayed at each point in sequence
    positions_grid, xc_values_grid = np.meshgrid(positions, xc.numpy())
    ax.scatter(
        positions_grid, 
        xc_values_grid, 
        c='black', 
        s=15,
        alpha=0.6, 
        zorder=3, # to ensure they are plotted over lines
        label='Context Point Locations'
    )


    # Adds target locations as red lines to give indicator of why sampling certain points may be good (i.e. close to target)
    if plot_targets:
        for target_x_val in xt:
            ax.axhline(y=target_x_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
        ax.plot([], [], color='red', linestyle='--', label='Target X-Coordinates')

    # Aesthetics
    ax.set_xlabel("Context Point Order", fontsize=16)
    ax.set_ylabel("X-Coordinate of Context Point", fontsize=16)
    ax.set_title(f"Parallel Coordinates Plot of {perm_xs.shape[0]} Permutations. NC={xc.shape[0]} NT={xt.shape[0]} K={K:,}", fontsize=20)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([i+1 for i in positions])
    ax.tick_params(axis='x', rotation=45)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Log-Likelihood", fontsize=16, rotation=270, labelpad=20)
    x_min, x_max = xc.min() - 0.5, xc.max() + 0.5
    if plot_targets: x_min, x_max = min(x_min, xt.min() - 0.2), max(x_max, xt.max() + 0.2)
    ax.set_ylim(x_min, x_max)
    if plot_targets: ax.legend()

    fname = f"{file_name}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=300)

    plt.close()


# Plots range of likelihoods with different permutations
def plot_log_p_bins(log_p, file_name, nc, nt, plain_tnp_perf=None):
    fig, ax = plt.subplots(figsize=(15, 10))

    lp_mean = log_p.mean()
    ax.axvline(lp_mean, color="grey", linestyle=":", linewidth=2.0, label=fr"Mean ($\mu={lp_mean:.2f}$)")
    lp_median = log_p.median()
    ax.axvline(lp_median, color="black", linestyle=":", linewidth=2.0, label=fr"Median ($\mathrm{{median}} = {lp_median:.2f}$)")

    # Histogram
    ax.hist(log_p, bins='auto', density=True)
    ax.set_xlabel("Log-Likelihood", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_title(rf"Fluctuation in Log-Likelihood over Different Permutations of Data (NC={nc} NT={nt} K={log_p.shape[0]:,})", fontsize=20)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(file_name + "_withoutbasetnp.png", bbox_inches="tight", dpi=300)
    # Adds red line to show the performance of a plain tnp model if it is given
    if plain_tnp_perf is not None:
        ax.axvline(plain_tnp_perf, color="red", linestyle="--", linewidth=2.5, 
            label=fr"TNP-D ($\ell={{{plain_tnp_perf:.2f}}}$)")
        ax.legend(frameon=False, fontsize=10, loc="best")
        plt.tight_layout()
        plt.savefig(file_name + "_withbasetnp.png", bbox_inches="tight", dpi=300)
    plt.close()




# Generates plots of permutations
@check_shapes(
    "perms: [K, nc]", "log_p: [K]", "xc: [1, nc, dx]", "yc: [1, nc, dy]", "xt: [1, nt, dx]", "yt: [1, nt, dy]"
)
def visualise_perms(tnp_model, perms: torch.tensor, log_p: torch.tensor, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, 
    folder_path: str="plot_results/adversarial", file_id: str=str(random.randint(0, 1000000)), gt_pred: Optional[GroundTruthPredictor] = None,
    plain_tnp_model = None):
    log_p, indices = torch.sort(log_p)
    perms = perms[indices]
    #print(perms)
    #print(log_p)
    file_name = f"{folder_path}/plain_tnp_id_{file_id}"
    plot_perm(model=plain_tnp_model, xc=xc, yc=yc, xt=xt, yt=yt, perm=perms[0], savefig=True, file_name=file_name, gt_pred=gt_pred, annotate=False)
    # Visualises permutations of various centiles (ie best, worst, median etc)
    perf_int = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
    for perc in perf_int:
        perc_idx = round((perc / 100) * (len(perms) - 1))
        perm, log_prob = perms[perc_idx], log_p[perc_idx]
        file_name = f"{folder_path}/seq_perm_{perc:03d}_id_{file_id}"
        plot_perm(model=tnp_model, xc=xc, yc=yc, xt=xt, yt=yt, perm=perm, savefig=True, file_name=file_name, gt_pred=gt_pred, annotate=True)

    # Parralel coordinates plot to see permutations ordering
    file_name = f"{folder_path}/parr_cord_id_{file_id}"
    plot_targets = xt.shape[1] <= 5 # Plot targets if there are not too many of them
    perms_spaced, log_p_spaced = get_spaced_examples(perms=perms, log_p=log_p, max_perms_plot=20)
    perms_extreme, log_p_extreme = get_best_and_worst(perms=perms, log_p=log_p, top_and_bottom_n=2)
    plot_parallel_coordinates_bezier(perms=perms_spaced,log_p=log_p_spaced, xc=xc, xt=xt, 
        file_name=file_name+"_spaced", plot_targets=plot_targets, alpha_line=0.4)
    plot_parallel_coordinates_bezier(perms=perms_extreme,log_p=log_p_extreme, xc=xc, xt=xt, 
        file_name=file_name+"_extreme", plot_targets=plot_targets, alpha_line=0.9)

    # Bins log probabilities to show variation in log probability with differing permutations
    plain_tnp_mean = None
    if plain_tnp_model is not None: 
        batch = SyntheticBatch(xc=xc, yc=yc, xt=xt, yt=yt, x=torch.cat([xc, xt], dim=1), y=torch.cat([yc, yt], dim=1))
        nt, dy = yt.shape[-2:]
        plain_tnp_mean = (plain_tnp_model(xc, yc, xt).log_prob(yt).sum(dim=(-1, -2)) / (nt * dy)).item()
    plot_log_p_bins(log_p.cpu(), f"{folder_path}/bins_dist_id_{file_id}", xc.shape[1], xt.shape[1], plain_tnp_mean)

def get_model(config_path, weights_and_bias_ref, device='cuda'):
    raw_config = deep_convert_dict(
        hiyapyco.load(
            config_path,
            method=hiyapyco.METHOD_MERGE,
            usedefaultyamlloader=True,
        )
    )

    # Initialise experiment, make path.
    config, _ = extract_config(raw_config, None)
    config = deep_convert_dict(config)

    # Instantiate experiment and load checkpoint.
    pl.seed_everything(config.misc.seed)
    experiment = instantiate(config)
    experiment.config = config
    pl.seed_everything(experiment.misc.seed)

    # Loads weights and bias model
    artifact = wandb.Api().artifact(weights_and_bias_ref, type='model')
    artifact_dir = artifact.download()
    ckpt_file = os.path.join(artifact_dir, "model.ckpt")
    lit_model = (
        LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
            ckpt_file, model=experiment.model,
        )
    )
    model = lit_model.model
    model.to(device)
    return model





if __name__ == "__main__":
    # E.g. run with: python experiments/plot_adversarial_perms.py
    # RBF kernel params
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    # Data generator params
    nc, nt = 9, 32
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 1
    batch_size = 1
    deterministic = True
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=0.1,
        deterministic=True, kernel=kernels)
    data = next(iter(gen_val))
    # Gets plain model - ensure these strings are correct
    plain_model = get_model('experiments/configs/synthetic1dRBF/gp_plain_tnp.yml', 
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200')
    plain_model.eval()

    masked_model = get_model('experiments/configs/synthetic1dRBF/gp_causal_tnp.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200')
    masked_model.eval()

    # Sorts context in order
    xc = data.xc
    yc = data.yc
    xc, indices = torch.sort(xc, dim=1)
    yc = torch.gather(yc, dim=1, index=indices)
    print("Starting search")
    perms, log_p, (data_time, inference_time, total_time) = gather_rand_perms(masked_model, xc, yc, data.xt, data.yt, 
        no_permutations=10_000_000, device='cuda', batch_size=2048)
    print(f"Data time: {data_time:.2f}s, Inference time: {inference_time:.2f}s, Total time: {total_time:.2f}s")
    visualise_perms(masked_model, perms, log_p, xc, yc, data.xt, data.yt,
        folder_path="experiments/plot_results/adversarial", file_id=str(random.randint(0, 1000000)), gt_pred=data.gt_pred, 
        plain_tnp_model=plain_model)