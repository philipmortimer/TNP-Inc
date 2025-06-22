# Measures sensitivity of TNP model to exchanging order of data based on eq 5 from https://proceedings.mlr.press/v253/mlodozeniec24a.html
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
import wandb
import os
from typing import Optional
from plot_adversarial_perms import get_model
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle
import random
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# Computes log joint variance of model - use Eq 5 but only for a fixed target and context set
@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]" , "perms_ctx: [K, nc]"
)
@torch.no_grad()
def m_var_fixed(tnp_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, perms_ctx: torch.Tensor, return_sample_index: Optional[int] = None):
    # perms_ctx = [K, nc] - K permutations for nc context points with their indices in the tensor
    K, nc = perms_ctx.shape
    _, _, dy = yc.shape
    m, _, dx = xc.shape
    nt = yt.shape[1]
    k_m = K * m

    assert return_sample_index == None or (return_sample_index >= 0 and return_sample_index < m), "Invalid return index"

    # Broadcasts context
    xc_broad = xc.unsqueeze(0).expand(K, -1, -1, -1) # [K, m, nc, dx]
    yc_broad = yc.unsqueeze(0).expand(K, -1, -1, -1) # [K, m, nc, dy]

    # Generates perm gather indices
    gather_x_idx = perms_ctx.view(K, 1, nc, 1).expand(-1, m, -1, dx)
    gather_y_idx = perms_ctx.view(K, 1, nc, 1).expand(-1, m, -1, dy)

    # Permutations
    xc_perm = torch.gather(xc_broad, 2, gather_x_idx)
    yc_perm = torch.gather(yc_broad, 2, gather_y_idx)

    # Broadcast target points to match shape
    xt_rep = xt.unsqueeze(0).expand(K, -1, -1, -1) # [K, m, nc, dx]
    yt_rep = yt.unsqueeze(0).expand(K, -1, -1, -1)  # [K, m, nc, dy]

    # Flattens data into a single batch
    xc_perm = xc_perm.reshape(k_m, nc, dx)
    yc_perm = yc_perm.reshape(k_m, nc, dy)
    xt_rep  = xt_rep.reshape(k_m, nt, dx)
    yt_rep  = yt_rep.reshape(k_m, nt, dy)

    # Creates a batch
    x = torch.cat((xc_perm, xt_rep), dim=1)
    y = torch.cat((yc_perm, yt_rep), dim=1)

    batch = Batch(x=x, y=y, xc=xc_perm, yc=yc_perm, xt=xt_rep, yt=yt_rep)
    # Note we choose not to use teacher forcing here currently
    log_probs = np_pred_fn(tnp_model, batch, predict_without_yt_tnpa=True).log_prob(yt_rep) # [K * m, nt, dy] 
    log_probs = log_probs.sum(dim=(-1, -2)).view(K, m) # sums over nt and dy [K, m]

    variance = log_probs.var(dim=0, unbiased=True) # Variance over K - this is Var_PI from eq 5 in paper (this gives [m])
    m_var_val = variance.mean().item() # Mean over m batches sampled from D^(n) - ie the monte-carlo approximation of the expectation
    # Also computes simplified version of eq 6 - the average NLL (a measure of model performance)
    mean = log_probs.mean(dim=0) # expectation over K - i.e. E_PI (shape is [m])
    m_mean_val = mean.mean().item() # Avergaes over the m batches sampled from D^(n)

    # Can also randomly return a sample
    rand_m_var = variance[return_sample_index] if return_sample_index is not None else None
    rand_m_mean = mean[return_sample_index] if return_sample_index is not None else None
    # Note - in this current implementation we only permutes the final full context. We don't account for incremental context points.
    # This may need reconsidering when we consider the masked variant with incremental learning. 
    # This func computes E_{(C, T) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{nt} log p_{\theta}(y_{t, j} | C_{PI}, x_{t, j})]]
    # when using a non-diagonal TNP. The inner expression changes when using different NPs (e.g. autoreg) and this needs to be considered also.
    return m_var_val, m_mean_val, rand_m_var, rand_m_mean

# Computes log joint variance of model. This is the full equation 5 from the paper - but more expensive
@check_shapes(
    "x: [m, n, dx]", "y: [m, n, dy]", "perms: [K, n]"
)
@torch.no_grad()
def m_var_autoreg(tnp_model, x: torch.Tensor, y: torch.Tensor, perms: torch.Tensor, return_sample_index: Optional[int] = None):
    # Computes the full equation 5 from the paper. Does this by incrementally expanding the context set and predicting only the next point
    K, n = perms.shape
    m = x.shape[0]

    assert return_sample_index == None or (return_sample_index >= 0 and return_sample_index < m), "Invalid return index"

    log_probs_list = []
    for k in range(K):
        x_perm = x[:, perms[k], :]
        y_perm = y[:, perms[k], :]
        log_prob_list = torch.zeros(m, device=x.device)
        # Note Appendix F.3.1 from paper about conditioning on no data (i.e. dont condition on empty context set)
        for i in range(1, n):
            xc = x_perm[:, :i, :]
            yc = y_perm[:, :i, :]
            xt = x_perm[:, i:i+1, :]
            yt = y_perm[:, i:i+1, :]
            log_p = tnp_model(xc, yc, xt).log_prob(yt) # [m, 1, dy]
            log_p = log_p.sum(dim=(-1, -2)) # sums out nt(=1) and dy giving shape [m] this gives us joint
            log_prob_list += log_p
        log_probs_list.append(log_prob_list)
    
    log_probs = torch.stack(log_probs_list, dim=0) # [K, m]
    variance = log_probs.var(dim=0, unbiased=True) # Variance over K - this is Var_PI from eq 5 in paper (this gives [m])
    m_var_val = variance.mean().item() # Mean over m batches sampled from D^(n) - ie the monte-carlo approximation of the expectation

    # Also computes eq 6 - the average NLL (a measure of model performance)
    mean = log_probs.mean(dim=0) # expectation over K - i.e. E_PI (shape is [m])
    m_mean_val = mean.mean().item() # Avergaes over the m batches sampled from D^(n)
    # Can also randomly return a sample
    rand_m_var = variance[return_sample_index] if return_sample_index is not None else None
    rand_m_mean = mean[return_sample_index] if return_sample_index is not None else None
    # Note - this function computes full equation 5 - incrementally building up the context set and predicting the next point.
    # This is slow and may look to optimise with KV.
    # Additionally, may wish to provide more control e.g. over batch size (do we actually want a batch size of 1 here?).
    # This func computes E_{(X, Y) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{n} log p_{\theta}(y_{j, PI} | x_{1:j, PI}, y_{1:j-1, PI})]]
    # Also want to consider implication of only predicting the next point. Does this make sense for non-diagonal TNPs where there may be an
    # advantage to modelling more targets together.
    return m_var_val, m_mean_val, rand_m_var, rand_m_mean


# Samples variance over trained models with different seeds
def exchange(models_with_different_seeds, data_loader, no_permutations, device, use_autoreg_eq, max_samples, seq_len, return_samples=None):
    assert return_samples == None or return_samples <= max_samples, "Cant return more samples than are computed"
    no_models = len(models_with_different_seeds)
    # Note - may want to consider diving by nt in future? (or even nt * dy)

    m_vars = []
    m_nlls = []
    m_var_nll_samples = []
    i = 0
    nc_prev, nt_prev = None, None
    for data in data_loader:
        # Ensures sequence length is correct and that nc and nt remain constant over samples.
        # This prevents greater variance for longer sequences (ie lacking comparison) but comes at cost of expressivity.
        # Could normalise variances or look at multiple sequence lengths.
        seq_len_data = data.x.shape[1]
        batch_size = data.xc.shape[0]
        assert seq_len_data == seq_len, f"Data sequence length {seq_len_data} does not match required sequence length {seq_len}."
        if nc_prev is not None and nt_prev is not None:
            assert data.xc.shape[1] == nc_prev, f"Context set size {data.xc.shape[1]} does not match previous size {nc_prev}."
            assert data.xt.shape[1] == nt_prev, f"Target set size {data.xt.shape[1]} does not match previous size {nt_prev}."
        nc_prev, nt_prev = data.xc.shape[1], data.xt.shape[1]

        if use_autoreg_eq:
            x, y = data.x, data.y
            x, y = x.to(device), y.to(device)
            n = x.shape[1]
            perms = torch.stack([torch.randperm(n, device=device) for _ in range(no_permutations)])
            #keys = torch.randn(no_permutations, n, device=device)
            #perms = keys.argsort(dim=-1) 
        else:
            xc, yc, xt, yt = data.xc, data.yc, data.xt, data.yt
            xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
            nc, nt = xc.shape[1], xt.shape[1]
            n = nc + nt
                
            perms = torch.stack([torch.randperm(nc, device=device) for _ in range(no_permutations)])
            #keys = torch.randn(no_permutations, nc, device=device)
            #perms = keys.argsort(dim=-1) 
        mods_out_mvar = []
        mods_out_mnll = []
        # Calculates if an individual sample is needed for plotting
        more_samples = return_samples is not None and len(m_var_nll_samples) < return_samples
        if more_samples: return_sample_index = random.randint(0, batch_size - 1)
            
        # Computes m_var for each model
        for model  in models_with_different_seeds:
            val = m_var_autoreg(model, x, y, perms, return_sample_index=return_sample_index) if use_autoreg_eq else m_var_fixed(model, xc, yc, xt, yt, perms, return_sample_index=return_sample_index)

            mods_out_mvar.append(val[0])
            mods_out_mnll.append(val[1])
            if more_samples: m_var_nll_samples.append((val[2], val[3]))
        m_vars.append(mods_out_mvar)
        m_nlls.append(mods_out_mnll)
        i += 1
        if max_samples is not None and  i >= max_samples: break

    assert i>=1, "No data batches were processed."
    assert return_samples is None or len(m_var_nll_samples) == return_samples, "Not enough return samples due to too small data loader"

    m_vars = np.array(m_vars)
    m_nlls = np.array(m_nlls)

    model_vars = m_vars.mean(axis=0) # Average for each model over the data batches
    model_nlls = m_nlls.mean(axis=0) # Average NLL over the data batches

    mean_m_vars = model_vars.mean() # Mean over the models
    mean_m_nlls = model_nlls.mean() # Mean NLL over the models

    # Can't do t test with single model
    if no_models == 1:
        return (mean_m_vars, None), (mean_m_nlls, None), m_var_nll_samples
    student_t_crit = stats.t.ppf(0.975, df=no_models - 1)
    sem_m_var = stats.sem(model_vars)
    sem_m_nll = stats.sem(model_nlls)
    half_w_m_var = student_t_crit * sem_m_var
    half_w_m_nll = student_t_crit * sem_m_nll
    return (mean_m_vars, half_w_m_var), (mean_m_nlls, half_w_m_nll), m_var_nll_samples

# Computes the exchangeability - this is the function to be called when computing exchangeability
def exchangeability_test(models, data, no_permutations=20, device='cuda', use_autoreg_eq=False, max_samples=200, seq_len=100, batch_size=16):
    assert no_permutations >= 2, "Must have at least 2 permutations to compute variance"
    data.batch_size=batch_size
    val_loader = torch.utils.data.DataLoader(
        data,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_val_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        pin_memory=True,
    )
    start_time = time.time()
    # Logs exchangeability
    (mean_m_var, half_w_m_var), (mean_m_nlls, half_w_m_nll), _ = exchange(models, val_loader, no_permutations, device, use_autoreg_eq, max_samples, seq_len)
    end_time = time.time()
    if half_w_m_var is None: half_w_m_var = 'N/A'
    if half_w_m_nll is None: half_w_m_nll = 'N/A'
    print("-----------")
    print(f"Exchangeability test time: {end_time - start_time:.4f} seconds")
    print(f"Exchangeability (eq 5): {mean_m_var} +/- {half_w_m_var}")
    print(f"LL (eq 6): {mean_m_nlls} +/- {half_w_m_nll}")
    print("-----------")

# Attempts to recreate something like figure 2
def plot_models_setup_rbf_same():
    # Loads each model
    tnp_plain = (get_model('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml', 'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200'), "TNP-D")
    inc_tnp = (get_model('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200'), "incTNP")
    tnp_ar = (get_model('experiments/configs/synthetic1dRBF/gp_tnpa_rangesame.yml', 'pm846-university-of-cambridge/tnpa-rbf-rangesame/model-6hwme8wi:v200'), "TNP-A")
    inc_tnp_batched=(get_model('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200'), "incTNP-Batched")
    models = [tnp_plain, inc_tnp, tnp_ar, inc_tnp_batched]
    #models = [inc_tnp_batched]

    # Data loader - RBF kernel in this case
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    nc, nt = 32, 64 
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 16_000
    batch_size = 4
    deterministic = True

    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=0.1,
        deterministic=True, kernel=kernels)
    
    return gen_val, models, nc, nt

# Attempts to recreate something like figure 2. Use plot_models_setup as helper for this func.
def plot_models(helper_tuple):
    # Colour pallete to use
    colours = cycle([
        "#0072B2", "#D55E00", "#E69F00", "#009E73",
        "#CC79A7", "#F0E442", "#56B4E9", "#000000"
        ])
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    (gen_val, models, nc, nt) = helper_tuple
    seq_len = nc + nt
    for (model, model_name), colour in zip(models, colours):
        model.eval()
        (mean_m_var, _,), (mean_m_nlls, _), m_var_nll_samples = exchange([model], gen_val, no_permutations=11, device='cuda', use_autoreg_eq=False, max_samples=10, seq_len=seq_len, return_samples=10)

        samples_m_var = [x[0].item() for x in m_var_nll_samples]
        samples_m_nll = [x[1].item() for x in m_var_nll_samples]

        # Plots small dots
        ax.scatter(
            samples_m_var,
            samples_m_nll,
            s=18,
            c=[colour],
            alpha=0.7,
            marker='o',
            edgecolors='none',
            zorder=2,
        )

        # Line to centroid dot
        for (sx, sy) in zip(samples_m_var, samples_m_nll):
            ax.plot([mean_m_var, sx], [mean_m_nlls, sy], lw=0.8, c=colour, alpha=0.35,zorder=1)

        # Plots large dot
        ax.scatter(
            mean_m_var,
            mean_m_nlls,
            s=100,
            c=[colour],
            alpha=0.95,
            marker='o',
            linewidth=0.6,
            edgecolors='k',
            label=model_name,
            zorder=3,
        )
        
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Joint Log-Likelihood Variance")
    ax.set_ylabel("Neg. Joint Log-Likelihood Mean (- Optimal)")

    # Tick params
    ax.tick_params(axis='both', which='major', length=4, width=0.8)

    ax.legend(
        frameon=False, ncol=2, handletextpad=0.4,
        columnspacing=0.8, loc="upper left"
    )

    ax.set_title("Implicit Bayesianness vs Performance")

    plt.savefig("experiments/plot_results/exchange/brunofig.png", bbox_inches="tight")



if __name__ == "__main__":
    plot_models(plot_models_setup_rbf_same())
    exit(0)
    # E.g. run with: python experiments/exchangeability.py --config experiments/configs/synthetic1d/gp_plain_tnp.yml
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
    nc, nt = 32, 64 
    batch_size = 16
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 16_000
    batch_size = 1024
    deterministic = True
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=0.1,
        deterministic=True, kernel=kernels)
    models = []
    useWandb = True # Defines if weights and biases model is to be used
    #wanddName = 'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200'
    wanddName = 'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200'
    if useWandb:
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
        models.append(tnp_model)
    else:
        model_arch.to('cuda')
        model_arch.eval()
        tnp_model=model_arch
        models.append(tnp_model)

    exchangeability_test(models, gen_val, no_permutations=20, device='cuda', use_autoreg_eq=False, max_samples = 100, seq_len = nc+nt, batch_size=16)
