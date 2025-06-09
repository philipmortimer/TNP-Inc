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

# Computes log joint variance of model - use Eq 5 but only for a fixed target and context set
@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]" , "perms_ctx: [K, nc]"
)
@torch.no_grad()
def m_var_fixed(tnp_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, perms_ctx: torch.Tensor):
    # perms_ctx = [K, nc] - K permutations for nc context points with their indices in the tensor
    K = perms_ctx.shape[0]
    m, nc = xc.shape[0], xc.shape[1]

    # Loops through each permutation - note tried stacking them to make faster but had weird bugs. This is also more readable.
    log_probs_list = []
    for k in range(K):
        xc_perm = xc[:, perms_ctx[k], :]
        yc_perm = yc[:, perms_ctx[k], :]
        log_p = tnp_model(xc_perm, yc_perm, xt).log_prob(yt) # [m, nt, dy]
        log_p = log_p.sum(dim=(-1, -2)) # nt and dy giving shape [m] this gives us joint
        log_probs_list.append(log_p)
    log_probs = torch.stack(log_probs_list, dim=0) # [K, m]
    variance = log_probs.var(dim=0, unbiased=True) # Variance over K - this is Var_PI from eq 5 in paper (this gives [m])
    m_var_val = variance.mean().item() # Mean over m batches sampled from D^(n) - ie the monte-carlo approximation of the expectation
    # Also computes simplified version of eq 6 - the average NLL (a measure of model performance)
    mean = log_probs.mean(dim=0).mean() # expectation over K - i.e. E_PI (shape is [m])
    m_mean_val = mean.mean().item() # Avergaes over the m batches sampled from D^(n)
    # Note - in this current implementation we only permutes the final full context. We don't account for incremental context points.
    # This may need reconsidering when we consider the masked variant with incremental learning. 
    # This func computes E_{(C, T) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{nt} log p_{\theta}(y_{t, j} | C_{PI}, x_{t, j})]]
    # when using a non-diagonal TNP. The inner expression changes when using different NPs (e.g. autoreg) and this needs to be considered also.
    return m_var_val, m_mean_val

# Computes log joint variance of model. This is the full equation 5 from the paper - but more expensive
@check_shapes(
    "x: [m, n, dx]", "y: [m, n, dy]", "perms: [K, n]"
)
@torch.no_grad()
def m_var_autoreg(tnp_model, x: torch.Tensor, y: torch.Tensor, perms: torch.Tensor):
    # Computes the full equation 5 from the paper. Does this by incrementally expanding the context set and predicting only the next point
    K, n = perms.shape
    m = x.shape[0]

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
    mean = log_probs.mean(dim=0).mean() # expectation over K - i.e. E_PI (shape is [m])
    m_mean_val = mean.mean().item() # Avergaes over the m batches sampled from D^(n)
    # Note - this function computes full equation 5 - incrementally building up the context set and predicting the next point.
    # This is slow and may look to optimise with KV.
    # Additionally, may wish to provide more control e.g. over batch size (do we actually want a batch size of 1 here?).
    # This func computes E_{(X, Y) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{n} log p_{\theta}(y_{j, PI} | x_{1:j, PI}, y_{1:j-1, PI})]]
    # Also want to consider implication of only predicting the next point. Does this make sense for non-diagonal TNPs where there may be an
    # advantage to modelling more targets together.
    return m_var_val, m_mean_val


# Samples variance over trained models with different seeds
def exchange(models_with_different_seeds, data_loader, no_permutations, device, use_autoreg_eq, max_samples, seq_len):
    no_models = len(models_with_different_seeds)

    m_vars = []
    m_nlls = []
    i = 0
    nc_prev, nt_prev = None, None
    for data in data_loader:
        # Ensures sequence length is correct and that nc and nt remain constant over samples.
        # This prevents greater variance for longer sequences (ie lacking comparison) but comes at cost of expressivity.
        # Could normalise variances or look at multiple sequence lengths.
        seq_len_data = data.x.shape[1]
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
        # Computes m_var for each model
        for model  in models_with_different_seeds:
            val = m_var_autoreg(model, x, y, perms) if use_autoreg_eq else m_var_fixed(model, xc, yc, xt, yt, perms)
            mods_out_mvar.append(val[0])
            mods_out_mnll.append(val[1])
        m_vars.append(mods_out_mvar)
        m_nlls.append(mods_out_mnll)
        i += 1
        if max_samples is not None and  i >= max_samples: break

    assert i>=1, "No data batches were processed."

    m_vars = np.array(m_vars)
    m_nlls = np.array(m_nlls)

    model_vars = m_vars.mean(axis=0) # Average for each model over the data batches
    model_nlls = m_nlls.mean(axis=0) # Average NLL over the data batches

    mean_m_vars = model_vars.mean() # Mean over the models
    mean_m_nlls = model_nlls.mean() # Mean NLL over the models

    # Can't do t test with single model
    if no_models == 1:
        return (mean_m_vars, None), (mean_m_nlls, None)
    student_t_crit = stats.t.ppf(0.975, df=no_models - 1)
    sem_m_var = stats.sem(model_vars)
    sem_m_nll = stats.sem(model_nlls)
    half_w_m_var = student_t_crit * sem_m_var
    half_w_m_nll = student_t_crit * sem_m_nll
    return (mean_m_vars, half_w_m_var), (mean_m_nlls, half_w_m_nll)

# Computes the exchangeability - this is the function to be called when computing exchangeability
def exchangeability_test(models, data, no_permutations=20, device='cuda', use_autoreg_eq=False, max_samples=200, seq_len=100, batch_size=16):
    assert no_permutations >= 2, "Must have at least 2 permutations to compute variance"
    data.batch_size=batch_size
    # TODO: fix dataloader to ensure that n is fixed for all batches (we MUST average over the same n each time)
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
    (mean_m_var, half_w_m_var), (mean_m_nlls, half_w_m_nll) = exchange(models, val_loader, no_permutations, device, use_autoreg_eq, max_samples, seq_len)
    end_time = time.time()
    if half_w_m_var is None: half_w_m_var = 'N/A'
    if half_w_m_nll is None: half_w_m_nll = 'N/A'
    print("-----------")
    print(f"Exchangeability test time: {end_time - start_time:.4f} seconds")
    print(f"Exchangeability (eq 5): {mean_m_var} +/- {half_w_m_var}")
    print(f"LL (eq 6): {mean_m_nlls} +/- {half_w_m_nll}")
    print("-----------")

if __name__ == "__main__":
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
    wanddName = 'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v0'
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
