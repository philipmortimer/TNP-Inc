# Plotting code for hadISD during validation
import copy
import os
from typing import Callable, List, Tuple, Union
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from tnp.data.hadISD import HadISDBatch, normalise_time, scale_pred_temp_dist
from tnp.utils.np_functions import np_pred_fn
import wandb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import datetime


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# 6 key plots per item
# 1) Show context and target stations with dots on the map
# 2) Show context station ordering (order of points)
# 3) Extrapolate predictions onto whole grid (i.e. not just stations)
# 4) Prediction at target stations
# 5) True station readings
# 6) show error at target stations
def plot_hadISD(
    model: nn.Module,
    batches: List[HadISDBatch],
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    name: str = "plot",
    savefig: bool = False,
    logging: bool = True,
    model_lbl: str="Model",
    pred_fn: Callable = np_pred_fn,
):
    for i in range(num_fig):
        batch = batches[i]
        BATCH_IDX = 0 # This is implicit from the original code - we only take the first item per batch to plot
        assert BATCH_IDX == 0, "Check logic for combined kernel with non zero plot batch index"
        xc = batch.xc[:BATCH_IDX+1] # same as batch.xc[:1] (i.e. first batch item)
        yc = batch.yc[:BATCH_IDX+1]
        xt = batch.xt[:BATCH_IDX+1]
        yt = batch.yt[:BATCH_IDX+1]
        unnorm_time = batch.unnormalised_time[BATCH_IDX]
        x = torch.cat((xc, xt), dim=1)
        y = torch.cat((yc, yt), dim=1)

        # Batch that can be used for the stations (i.e. predict on a given set of stations given context only m = 1)
        batch_pred = HadISDBatch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt,
            mean_temp=batch.mean_temp, std_temp=batch.std_temp, mean_elev=batch.mean_elev, std_elev=batch.std_elev,
            lat_range=batch.lat_range, long_range=batch.long_range, unnormalised_time=unnorm_time, ordering=batch.ordering)
        
        # Makes a batch within the lat and long range of all points (i.e. gridded). context is same just targets different
        # For computation reasons this grid can not be too big
        # TODO precache this grid with true z estimates for grid - currently just set Z as constant for simplicity which is ofc not true
        N_POINTS = 100 # defines N x N grid
        lat_points = np.linspace(batch.lat_range[0], batch.lat_range[1], N_POINTS)
        long_points = np.linspace(batch.long_range[0], batch.long_range[1], N_POINTS)
        lat_grid, lon_grid = np.meshgrid(lat_points, long_points)
        time = np.full(shape=(N_POINTS*N_POINTS), fill_value=unnorm_time.cpu())
        time = normalise_time(time)
        # Z constaint hack: TODO precache with true values
        z_const_temp = 30.0
        elevation = np.full(shape=(N_POINTS * N_POINTS), fill_value=z_const_temp) * batch.std_elev + batch.mean_elev
        # Convert stuff to tensors
        lat = torch.tensor(lat_grid.flatten(), device=xc.device, dtype=xc.dtype)
        long = torch.tensor(lon_grid.flatten(),  device=xc.device, dtype=xc.dtype)
        time = torch.tensor(time, device=xc.device, dtype=xc.dtype) # [N]
        elevation = torch.tensor(elevation, device=xc.device, dtype=xc.dtype) #[N]
        xt_grid = torch.stack((lat, long, time, elevation), dim=-1) # [N, 4]
        # Shuffles data
        if batch.ordering == "random":
            indices = torch.randperm(N_POINTS * N_POINTS)
            xt_grid = xt_grid[indices]
        else:
            raise ValueError("Unspoorted plotting ordering type")
        xt_grid = xt_grid.unsqueeze(0) # adds batch dim of 1
        # Creates batch to predict all temp readings on grid
        batch_grid = HadISDBatch(x=None, y=None, xc=xc, yc=yc, xt=xt_grid, yt=None,
            mean_temp=batch.mean_temp, std_temp=batch.std_temp, mean_elev=batch.mean_elev, std_elev=batch.std_elev,
            lat_range=batch.lat_range, long_range=batch.long_range, unnormalised_time=unnorm_time, ordering=batch.ordering)

        # Gets predictive distributions and scales to correct units
        with torch.no_grad():
            yt_pred_dist = pred_fn(model, batch_pred)
            y_gridded_pred_dist = pred_fn(model, batch_grid, predict_without_yt_tnpa=True)
        yt_pred_dist = scale_pred_temp_dist(batch_pred, yt_pred_dist)
        y_gridded_pred_dist = scale_pred_temp_dist(batch_grid, y_gridded_pred_dist)
        # Computes NLL
        nll = -yt_pred_dist.log_prob(batch_pred.yt).sum() / batch_pred.yt[..., 0].numel()
        _, nc, _ = batch_pred.xc.shape
        _, nt, _ = batch_pred.xt.shape
        # Converts points to true long / lat value and to cpu for for plotting
        long_ctx = (((xc[:, :,1].cpu() + 1.0) / 2.0) * (batch_pred.long_range[1] - batch_pred.long_range[0])) + batch_pred.long_range[0]
        lat_ctx = (((xc[:, :,0].cpu() + 1.0) / 2.0) * (batch_pred.lat_range[1] - batch_pred.lat_range[0])) + batch_pred.lat_range[0]
        long_tgt = (((xt[:, :,1].cpu() + 1.0) / 2.0) * (batch_pred.long_range[1] - batch_pred.long_range[0])) + batch_pred.long_range[0]
        lat_tgt = (((xt[:, :,0].cpu() + 1.0) / 2.0) * (batch_pred.lat_range[1] - batch_pred.lat_range[0])) + batch_pred.lat_range[0]
        long_grid = (((batch_grid.xt[:, :,1].cpu() + 1.0) / 2.0) * (batch_pred.long_range[1] - batch_pred.long_range[0])) + batch_pred.long_range[0]
        lat_grid = (((batch_grid.xt[:, :,0].cpu() + 1.0) / 2.0) * (batch_pred.lat_range[1] - batch_pred.lat_range[0])) + batch_pred.lat_range[0]

        proj = ccrs.PlateCarree()
        batch_time_str = convert_time_to_str(unnorm_time.cpu().item())
        # 1) Show context and target stations
        title_a = f"NC={nc} NT={nt} - {batch_time_str}"
        fig_a, ax_a = init_earth_fig(title_a, figsize, proj, batch_pred.lat_range, batch_pred.long_range)
        ax_a.scatter(long_ctx, lat_ctx, c="k", s=10, label="Context")
        ax_a.scatter(long_tgt, lat_tgt, c="r", s=10, label="Target")
        ax_a.legend()
        save_plot(fig_a, name, i, "A", logging, savefig)


    exit(0)

# Converts number of hours since 1st Jan 1931 into a formatted string
def convert_time_to_str(unnorm_time: int):
    ZERO_TIME = datetime.datetime(1931, 1, 1)
    final_datetime = ZERO_TIME + datetime.timedelta(hours=unnorm_time)
    return final_datetime.strftime("%H:00 %d %B %Y")

# Creates earth map figure outline to be used for plotting
def init_earth_fig(title, figsize, proj, lat_range, long_range):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([*long_range, *lat_range], crs=proj)
    ax.set_title(title)
    return fig, ax


# Saves each plot
def save_plot(fig, name, i, panel, logging, savefig):
    tag = f"{name}/{i:03d}_{panel}"
    if wandb.run is not None and logging:
        wandb.log({tag: wandb.Image(fig)})
    elif savefig:
        base_folder = f"{name}"
        save_name = base_folder + f"/{i:03d}_{panel}.png"
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        fig.savefig(out / f"{i:03d}_{panel}.png", bbox="tight")
    else:
        plt.show()
    plt.close(fig)
