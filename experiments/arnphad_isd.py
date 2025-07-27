# Autoregressive neural process - test time only. For HADISD (copy of arnp.py but for hadISD essentially)
# Based on https://arxiv.org/pdf/2303.14468 - takes a normal NP model and treats predicted target points as context points
# Inspired by https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional, Union, Literal, Callable, Tuple
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch
from tnp.models.incUpdateBase import IncUpdateEff, IncUpdateEffFixed
from plot_adversarial_perms import get_model
from functools import partial
from tqdm import tqdm
import numpy as np
import torch.distributions as td
import os
import matplotlib.pyplot as plt
import matplotlib
from tnp.data.hadISD import HadISDDataGenerator
from plot_hadISD import plot_hadISD
import numpy as np
from tnp.utils.data_loading import adjust_num_batches
from data_temp.data_processing.elevations import get_cached_elevation_grid


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.titlesize"]= 14


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]",
)
@torch.no_grad
def _shuffle_targets(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: Optional[torch.Tensor],
    order: Literal["random", "given", "left-to-right", "variance"]):
    m, nt, dx = xt.shape
    _, _, dy = yc.shape
    device = xt.device
    if order == "given":
        perm = torch.arange(nt, device=device).repeat(m, 1)
        return xt, yt, perm
    elif order == "random":
        perm = torch.rand(m, nt, device=device).argsort(dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_shuffled = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_shuffled = torch.gather(yt, 1, perm_y)
        else: yt_shuffled = None
        return xt_shuffled, yt_shuffled, perm
    elif order == "left-to-right":
        assert dx == 1, "left-to-right ordering only supported for one dimensional dx"
        perm = torch.argsort(xt.squeeze(-1), dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else: yt_sorted = None
        return xt_sorted, yt_sorted, perm
    elif order == "variance":
        # Predicts all target points conditioned on context points and orders (highest variance first) - this is obviously much more expensive
        batch = Batch(xc=xc, yc=yc, xt=xt, yt=None, x=None, y=None)
        pred_dist = np_pred_fn(np_model, batch)
        var = pred_dist.variance.mean(-1) # Gets variance (averaged over dy) [m, nt]
        perm = torch.argsort(var, dim=1, descending=True)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else: yt_sorted = None
        return xt_sorted, yt_sorted, perm



@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]"
)
@torch.no_grad
def ar_metrics(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
    normalise: bool = True, order: Literal["random", "given", "left-to-right", "variance"] = "random") -> torch.Tensor:
    xt, yt, _ = _shuffle_targets(np_model, xc, yc, xt, yt, order)
    np_model.eval()
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape
    log_probs = torch.zeros((m), device=xt.device)
    squared_errors = torch.zeros((m), device=xt.device, dtype=torch.float64)
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

        squared_errors += (pred_dist.mean - yt_sel).to(squared_errors.dtype).pow(2).sum(dim=(-1, -2))
    if normalise:
        log_probs /= (nt * dy)
    rmse = torch.sqrt(squared_errors / (nt * dy)).to(xt.dtype)
    return log_probs, rmse


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
)
@torch.no_grad
def ar_predict(model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,
    order: Literal["random", "given", "left-to-right", "variance"] = "random",
    num_samples: int = 10,
    prioritise_fixed: bool = False, # If incremental updates are available prioritise fixed or true dynamic algorithm
    device: str = "cuda", # Device for computing
    device_ret: str = "cpu", # Return device
    use_flash: bool = False, # Use flash kernel if posible
    ):
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape
    xc, yc, xt = xc.to(device), yc.to(device), xt.to(device)

    xc_stacked = xc.repeat_interleave(num_samples, dim=0)
    yc_stacked = yc.repeat_interleave(num_samples, dim=0)
    xt_stacked = xt.repeat_interleave(num_samples, dim=0)

    xt_stacked, _, perm = _shuffle_targets(model, xc_stacked, yc_stacked, xt_stacked, None, order) # Should I shuffle before or after stacking?

    yt_preds_mean, yt_preds_std = torch.empty((m * num_samples, nt, dy), device=device), torch.empty((m * num_samples, nt, dy), device=device)

    is_fixed_inc_update = isinstance(model, IncUpdateEffFixed)
    is_inc_gen_update = isinstance(model, IncUpdateEff)
    is_fixed_inc_update = (is_fixed_inc_update and prioritise_fixed) or (is_fixed_inc_update and not is_inc_gen_update)
    is_inc_gen_update = (is_inc_gen_update and  not prioritise_fixed) or (is_inc_gen_update and not is_fixed_inc_update)
    assert is_fixed_inc_update != is_inc_gen_update or (not is_fixed_inc_update and not is_inc_gen_update), "Xor onf fixed vs inc update"
    if is_inc_gen_update:
        model.init_inc_structs(m=xc_stacked.shape[0], max_nc=nc+nt, device=device,use_flash=use_flash)
        model.update_ctx(xc=xc_stacked, yc=yc_stacked,use_flash=use_flash)
    elif is_fixed_inc_update:
        model.init_inc_structs_fixed(m=xc_stacked.shape[0], max_nc=nc+nt, xt=xt_stacked, dy=dy, device=device, use_flash=use_flash)

    for i in range(nt):
        xt_tmp = xt_stacked[:, i:i+1,:]
        if is_inc_gen_update:
            pred_dist = model.query(xt=xt_tmp, dy=dy,use_flash=use_flash)
        elif is_fixed_inc_update:
            pred_dist = model.query_fixed(tgt_start_ind=i, tgt_end_ind=i+1, use_flash=use_flash)
        else:
            batch = Batch(xc=xc_stacked, yc=yc_stacked, xt=xt_tmp, yt=None, x=None, y=None)
            pred_dist = np_pred_fn(model, batch)
        assert isinstance(pred_dist, td.Normal), "Must predict a gaussian"
        pred_mean, pred_std = pred_dist.mean, pred_dist.stddev
        yt_preds_mean[:,i:i+1,:] = pred_mean
        yt_preds_std[:,i:i+1,:] = pred_std
        # Samples from the predictive distribution and updates the context
        if i < nt - 1:
            yt_sampled = pred_dist.sample() # [m * num_samples, 1, dy]
            if is_inc_gen_update:
                model.update_ctx(xc=xt_tmp, yc=yt_sampled,use_flash=use_flash)
            elif is_fixed_inc_update:
                model.update_ctx_fixed(xc=xt_tmp, yc=yt_sampled, use_flash=use_flash)
            else:
                xc_stacked = torch.cat((xc_stacked, xt_tmp), dim=1)
                yc_stacked = torch.cat((yc_stacked, yt_sampled), dim=1)
                
    # Unshuffles the target ordering to be in line with what was passed in
    inv_perm = perm.argsort(dim=1)
    idx = inv_perm.unsqueeze(-1).expand(-1, -1, dy)
    yt_preds_mean = yt_preds_mean.gather(dim=1, index=idx)
    yt_preds_std  = yt_preds_std .gather(dim=1, index=idx)

    yt_preds_mean = yt_preds_mean.view(num_samples, m, nt, dy)
    yt_preds_std = yt_preds_std.view(num_samples, m, nt, dy)
    # Permutes to [m, nt, dy, num_samples]
    yt_preds_mean = yt_preds_mean.permute(1,2,3,0)
    yt_preds_std  = yt_preds_std.permute(1,2,3,0)
    mix  = td.Categorical(torch.full((m, nt, dy, num_samples), 1.0 / num_samples, device=device_ret))
    comp = td.Normal(yt_preds_mean.to(device_ret), yt_preds_std.to(device_ret))
    approx_dist = td.MixtureSameFamily(mix, comp)

    # For sample draws return raw samples and run through model again for smooth samples (see paper / code)
    return approx_dist



# -------------------------------------------------------------------------------------------------------

# Measures timings of different models - this from some perspecticves is v similar to ar.py (doesnt use dataset per se)
def measure_perf_timings():
    # Measure hypers
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 1 # Number of runs to aggregate data over
    token_step = 500 # How many increments of tokens to go up in
    min_nt, max_nt = 1, 2003
    dx, dy, m = 4, 1, 1
    nc_start = 1
    num_samples=50 # Samples to unroll in ar_predict
    device = "cuda"
    order="random"
    prioritise_fixed = False
    plot_name_folder = "experiments/plot_results/hadar/perf/"
    # End of measure hypers
    models = get_model_list()
    max_high = 2
    xc = (torch.rand((m, nc_start, dx), device=device) * max_high * 2) - max_high
    yc = (torch.rand((m, nc_start, dy), device=device) * max_high * 2) - max_high
    target_sizes = np.arange(start=min_nt, stop=max_nt, step=token_step, dtype=int)
    runtime = np.zeros((len(models), aggregate_over, len(target_sizes)))
    memory = np.zeros((len(models), aggregate_over, len(target_sizes)))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval() 
        for t_index, nt in tqdm(enumerate(target_sizes), desc=f'Targ {model_name}'):
            xt = (torch.rand((m, nt, dx), device=device) * max_high * 2) - max_high
            yt = (torch.rand((m, nt, dy), device=device) * max_high * 2) - max_high

            for j in range(burn_in + aggregate_over):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                starter.record()
                with torch.no_grad():
                    pred_dist = ar_predict(model=model, xc=xc, yc=yc, xt=xt, order=order, num_samples=num_samples,
                        device=device, device_ret=device, prioritise_fixed=prioritise_fixed)
                # Measures time and memory
                ender.record()
                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                runtime_ms = starter.elapsed_time(ender)
                # Stores results
                write_idx = j - burn_in
                if write_idx >= 0:
                    runtime[model_idx, write_idx, t_index] = runtime_ms
                    memory[model_idx, write_idx, t_index] = peak_memory_mb
    # Aggregates results
    runtime = np.mean(runtime, axis=1) # [no_models, len(target_sizes)]
    memory = np.mean(memory, axis=1)
    # Plots runtime
    runtime_file_name = plot_name_folder + f'runtime_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(target_sizes, runtime[model_idx] / 1000.0, label=model_name)
    ax.set_xlabel('Target Size')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    ax.set_title(f'Runtime of AR NPs (S={num_samples} NC={nc_start})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(runtime_file_name, dpi=300)
    # Plots memory
    memory_file_name = plot_name_folder + f'memory_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(target_sizes, memory[model_idx], label=model_name)
    ax.set_xlabel('Target Size')
    ax.set_ylabel('Memory Usage (MB)')
    ax.legend()
    ax.set_title(f'Memory Usage of AR NPs (S={num_samples} NC={nc_start})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(memory_file_name, dpi=300)


def measure_perf_timings_hadisd_plot():
    # Measure hypers
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 1 # Number of runs to aggregate data over
    token_step = 50 # How many increments of tokens to go up in
    min_nc, max_nc = 1, 2003
    nt = 250
    dx, dy, m = 4, 1, 1
    nc_start = 1
    num_samples=50 # Samples to unroll in ar_predict
    device = "cuda"
    order="random"
    prioritise_fixed = False # can achieve flash with hacky use_flash = true in base mha function alongside non fixed code
    plot_name_folder = "experiments/plot_results/hadar/perf_plt/"
    tnp_plain = ('experiments/configs/hadISD/had_tnp_plain.yml',
        'pm846-university-of-cambridge/plain-tnp-had/model-o20d6s1q:v99', 'TNP-D')
    batchedTNP = ('experiments/configs/hadISD/had_incTNP_batched.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-had/model-z5nlguxq:v99', 'incTNP-Batched')
    cnp = ('experiments/configs/hadISD/had_cnp.yml',
        'pm846-university-of-cambridge/cnp-had/model-suqmhf9v:v99', 'CNP')
    conv_cnp = ('experiments/configs/hadISD/had_convcnp.yml',
        'pm846-university-of-cambridge/convcnp-had/model-p4f775ey:v98', 'ConvCNP (50 x 50)')   
    conv_cnp_big = ('experiments/configs/hadISD/alt_variants/had_big_convcnp.yml',
        '', 'ConvCNP (100 x 100)')
    conv_cnp_125 = ('experiments/configs/hadISD_csd3/alt_variants/had_125_convcnp.yml',
        '', 'ConvCNP (125 x 125)') 
    conv_cnp_150 = ('experiments/configs/hadISD_csd3/alt_variants/had_between_convcnp.yml',
        '', 'ConvCNP (150 x 150)')      
    models = [tnp_plain, batchedTNP, cnp, conv_cnp_big, conv_cnp_125, conv_cnp_150]
    # End of measure hypers
    max_high = 2
    xt = (torch.rand((m, nt, dx), device=device) * max_high * 2) - max_high
    yt = (torch.rand((m, nt, dy), device=device) * max_high * 2) - max_high
    context_sizes = np.arange(start=min_nc, stop=max_nc, step=token_step, dtype=int)
    runtime = np.zeros((len(models), aggregate_over, len(context_sizes)))
    memory = np.zeros((len(models), aggregate_over, len(context_sizes)))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        use_flash = False
        if model_name == "incTNP-Batched":
            use_flash=True
        model = get_model(model_yml, model_wab, seed=False, device=device, instantiate_only_model=True, load_mod_weights=False)
        model.eval() 
        for t_index, nc in tqdm(enumerate(context_sizes), desc=f'Targ {model_name}'):
            xc = (torch.rand((m, nc, dx), device=device) * max_high * 2) - max_high
            yc = (torch.rand((m, nc, dy), device=device) * max_high * 2) - max_high

            for j in range(burn_in + aggregate_over):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                starter.record()
                if use_flash:
                    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                        pred_dist = ar_predict(model=model, xc=xc, yc=yc, xt=xt, order=order, num_samples=num_samples,
                            device=device, device_ret=device, prioritise_fixed=prioritise_fixed, use_flash=use_flash)
                else:
                    with torch.no_grad():
                        pred_dist = ar_predict(model=model, xc=xc, yc=yc, xt=xt, order=order, num_samples=num_samples,
                            device=device, device_ret=device, prioritise_fixed=prioritise_fixed)
                # Measures time and memory
                ender.record()
                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                runtime_ms = starter.elapsed_time(ender)
                # Stores results
                write_idx = j - burn_in
                if write_idx >= 0:
                    runtime[model_idx, write_idx, t_index] = runtime_ms
                    memory[model_idx, write_idx, t_index] = peak_memory_mb
    # Aggregates results
    runtime = np.mean(runtime, axis=1) # [no_models, len(target_sizes)]
    memory = np.mean(memory, axis=1)
    # Plots runtime
    runtime_file_name = plot_name_folder + f'runtime_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(context_sizes, runtime[model_idx] / 1000.0, label=model_name)
    ax.set_xlabel('Number of Context Stations')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    ax.set_title(f'Runtime of AR NPs on HadISD (S={num_samples} NT={nt})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(runtime_file_name, dpi=300)
    # Plots memory
    memory_file_name = plot_name_folder + f'memory_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(context_sizes, memory[model_idx], label=model_name)
    ax.set_xlabel('Number of Context Stations')
    ax.set_ylabel('Memory Usage (MB)')
    ax.legend()
    ax.set_title(f'Memory Usage of AR NPs on HadISD (S={num_samples} NT={nt})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(memory_file_name, dpi=300)



# Plots a handful of kernels
def plot_ar_unrolls():
    # Hypers
    huge_grid_plots = False # Whether to plot enormous grid prediction - very slow computationally for AR
    order="random"
    #no_samples = [1, 2, 5, 10, 50, 100, 500, 1000]
    no_samples = [1, 2, 10, 50]
    folder_name = "experiments/plot_results/hadar/plots/"
    no_kernels = 5#20
    device="cuda"
    # End of hypers
    models = get_model_list()
    data, lat_mesh, lon_mesh, elev_np = get_had_testset_and_plot_stuff()

    batches_plot = []
    for i, batch in enumerate(data):
        batches_plot.append(batch)
        if i >= no_kernels: break

    for (model_yml, model_wab, model_name) in models:
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        model_folder = f"{folder_name}/{model_name}"
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        for sample in no_samples:
            def pred_fn_pred(model, batch, predict_without_yt_tnpa=True):
                return ar_predict(model, batch.xc, batch.yc, batch.xt, order, sample, device=device)
            plot_hadISD(
                model=model,
                batches=batches_plot,
                num_fig=len(batches_plot),
                name=model_folder+f"/ns_{sample}_od_{order}",
                pred_fn=pred_fn_pred,
                lat_mesh=lat_mesh,
                lon_mesh=lon_mesh,
                elev_np=elev_np,
                savefig=True, 
                logging=False,
                model_lbl=f"AR {model_name} (S={sample}) ",
                huge_grid_plots=huge_grid_plots
            )
                

# Loads hadISD set
def get_had_testset_and_plot_stuff():
    # Change these for correct machine / directory
    data_directory = "/scratch/pm846/TNP/data/data_processed/test"
    dem_path = "/scratch/pm846/TNP/data/elev_data/ETOPO_2022_v1_60s_N90W180_surface.nc"
    cache_dem_dir = "/scratch/pm846/TNP/data/elev_data/"
    num_grid_points_plot = 200
    # Normal hypers
    min_nc = 1
    max_nc = 2033
    nt = 250
    samples_per_epoch= 80_000
    batch_size = 32
    deterministic = True
    ordering_strategy = "random"
    num_val_workers = 1

    # Loads had dataset
    gen_test = HadISDDataGenerator(min_nc=min_nc, max_nc=max_nc, nt=nt, ordering_strategy=ordering_strategy,
        samples_per_epoch=samples_per_epoch, batch_size=batch_size, data_directory=data_directory)
    
    # Wraps data set in a proper torch set loader for less IO bottlenecking
    test_loader = torch.utils.data.DataLoader(
       gen_test,
        batch_size=None,
        num_workers=num_val_workers,
        worker_init_fn=(
            (
                adjust_num_batches
            )
            if num_val_workers > 0
            else None
        ),
        persistent_workers=True if num_val_workers > 0 else False,
        pin_memory=True,
    )

    # Loads elevation data from DEM file
    lat_mesh, lon_mesh, elev_np = get_cached_elevation_grid(gen_test.lat_range, gen_test.long_range,
        num_grid_points_plot, cache_dem_dir,
        dem_path)

    return test_loader, lat_mesh, lon_mesh, elev_np

def get_model_list():
    # List of models to compare
    tnp_plain = ('experiments/configs/hadISD/had_tnp_plain.yml',
        'pm846-university-of-cambridge/plain-tnp-had/model-o20d6s1q:v99', 'TNP-D')
    incTNP = ('experiments/configs/hadISD/had_incTNP.yml', 
        'pm846-university-of-cambridge/mask-tnp-had/model-9w1vbqjh:v99', 'incTNP')
    batchedTNP = ('experiments/configs/hadISD/had_incTNP_batched.yml',
        'pm846-university-of-cambridge/mask-batched-tnp-had/model-z5nlguxq:v99', 'incTNP-Batched')
    priorBatched = ('experiments/configs/hadISD/had_incTNP_priorbatched.yml',
        'pm846-university-of-cambridge/mask-priorbatched-tnp-had/model-83h4gpp2:v99', 'incTNP-Batched (Prior)')
    lbanp =('experiments/configs/hadISD/had_lbanp.yml',
        'pm846-university-of-cambridge/lbanp-had/model-zyzq4mno:v99', 'LBANP',)
    cnp = ('experiments/configs/hadISD/had_cnp.yml',
        'pm846-university-of-cambridge/cnp-had/model-suqmhf9v:v99', 'CNP')
    conv_cnp = ('experiments/configs/hadISD/had_convcnp.yml',
        'pm846-university-of-cambridge/convcnp-had/model-p4f775ey:v98', 'ConvCNP (50 x 50)')
    conv_cnp_100 = ('experiments/configs/hadISD/alt_variants/had_big_convcnp.yml',
        'pm846-university-of-cambridge/convcnp-had/model-ecytkrfq:v99', 'ConvCNP (100 x 100)')
    #models = [tnp_plain, incTNP, batchedTNP, priorBatched, lbanp, cnp, conv_cnp]
    #models = [batchedTNP, conv_cnp, cnp, incTNP, priorBatched, tnp_plain, lbanp]
    models = [tnp_plain, conv_cnp, batchedTNP, cnp]
    models = [priorBatched, cnp, conv_cnp]
    return models

# Compares NP models in AR mode on RBF set
def compare_had_models(base_out_txt_file: str, device: str = "cuda"):
    # Hypers to select - also look at dataset hypers
    ordering = "random"
    # End of hypers
    # Main loop - loads each model than compares writes performances to a text file
    models = get_model_list()
    data, lat_mesh, lon_mesh, elev_np = get_had_testset_and_plot_stuff()
    out_txt = ""
    for (model_yml, model_wab, model_name) in models:
        ll_list, rmse_list = [], []
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        for batch in tqdm(data, desc=f'{model_name} eval'):
            ll, rmse = ar_metrics(np_model=model, xc=batch.xc.to(device), yc=batch.yc.to(device),
                xt=batch.xt.to(device), yt=batch.yt.to(device), normalise=True, order=ordering)
            mean_ll = torch.mean(ll).item() # Goes from [m] to a float
            mean_rmse = torch.mean(rmse).item()
            ll_list.append(mean_ll)
            rmse_list.append(mean_rmse)
        ll_average = np.mean(ll_list)
        ll_std = np.std(ll_list) / np.sqrt(len(ll_list))

        rmse_average = np.mean(rmse_list)
        rmse_std = np.std(rmse_list) / np.sqrt(len(rmse_list))
        mod_sum = ("-" * 20) + f"\nModel: {model_name}\nMean LL: {ll_average} STD LL: {ll_std} Mean RMSE: {rmse_average} STD RMSE: {rmse_std}\n"
        print(mod_sum)
        out_txt += mod_sum
    with open(base_out_txt_file + f'_{ordering}.txt', 'w') as file:
        file.write(out_txt)


if __name__ == "__main__":
    #compare_had_models(base_out_txt_file="experiments/plot_results/hadar/ar_had_comp")
    #plot_ar_unrolls()
    #measure_perf_timings()
    measure_perf_timings_hadisd_plot() # for clearer plots