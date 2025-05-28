# Measures sensitivity of TNP model to exchanging order of data based on eq 5 from https://proceedings.mlr.press/v253/mlodozeniec24a.html
import numpy as np
import torch
from scipy import stats
from check_shapes import check_shapes
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.lightning_utils import LitWrapper
import time

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
    # Note - in this current implementation we only permutes the final full context. We don't account for incremental context points.
    # This may need reconsidering when we consider the masked variant with incremental learning. 
    # This func computes E_{(C, T) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{nt} log p_{\theta}(y_{t, j} | C_{PI}, x_{t, j})]]
    # when using a non-diagonal TNP. The inner expression changes when using different NPs (e.g. autoreg) and this needs to be considered also.
    return m_var_val

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
    # Note - this function computes full equation 5 - incrementally building up the context set and predicting the next point.
    # This is slow and may look to optimise with KV.
    # Additionally, may wish to provide more control e.g. over batch size (do we actually want a batch size of 1 here?).
    # This func computes E_{(X, Y) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{n} log p_{\theta}(y_{j, PI} | x_{1:j, PI}, y_{1:j-1, PI})]]
    # Also want to consider implication of only predicting the next point. Does this make sense for non-diagonal TNPs where there may be an
    # advantage to modelling more targets together.
    return m_var_val


# Samples variance over trained models with different seeds
def exchange(models_with_different_seeds, data_loader, no_permutations, device, use_autoreg_eq):
    no_models = len(models_with_different_seeds)

    m_vars = []
    for data in data_loader:
        nc = data.xc.shape[1]
        n = nc + data.xt.shape[1]
        if use_autoreg_eq:
            x, y = data.x, data.y
            x, y = x.to(device), y.to(device)
            perms = torch.stack([torch.randperm(n, device=device) for _ in range(no_permutations)])
        else:
            xc, yc, xt, yt = data.xc, data.yc, data.xt, data.yt
            xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
            perms = torch.stack([torch.randperm(nc, device=device) for _ in range(no_permutations)])
        mods_out = []
        # Computes m_var for each model
        for model  in models_with_different_seeds:
            val = m_var_autoreg(model, x, y, perms) if use_autoreg_eq else m_var_fixed(model, xc, yc, xt, yt, perms)
            mods_out.append(val)
        print(mods_out)
        m_vars.append(mods_out)
        break

    m_vars = np.array(m_vars)
    m_vars = m_vars.mean(axis=0) # Average for each model over the data batches
    mean = m_vars.mean()
    # Can't do t test with single model
    if no_models == 1:
        return mean, None
    sem = stats.sem(m_vars)
    student_t_crit = stats.t.ppf(0.975, df=no_models - 1)
    half_w = student_t_crit * sem
    return mean, half_w

# Computes the exchangeability - this is the function to be called when computing exchangeability
def exchangeability_test(models, data, no_permutations=128, device='cuda', use_autoreg_eq=True):
    assert no_permutations >= 2, "Must have at least 2 permutations to compute variance"
    start_time = time.time()
    # Logs exchangeability
    mean, half_w = exchange(models, data, no_permutations, device, use_autoreg_eq)
    end_time = time.time()
    if half_w is None: half_w = 'N/A'
    print(f"Exchangeability test time: {end_time - start_time:.4f} seconds")
    print(f"Exchangeability: {mean} +/- {half_w}")

if __name__ == "__main__":
    # E.g. run with: python experiments/exchangeability.py --config experiments/configs/synthetic1d/gp_plain_tnp.yml
    experiment = initialize_experiment() # Gets config file
    model_arch = experiment.model # Gets type of model
    # Data generator
    gen_val = experiment.generators.val
    val_loader = torch.utils.data.DataLoader(
        gen_val,
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

    models = []
    use_weights = False # Defines if local checkpoints are to be used
    if use_weights:
        # Example usage - ammend with specific model paths and data etc
        model_paths = ["/scratch/pm846/TNP/checkpoints/epoch=499-step=500000.ckpt"]
        # Loads models
        for model_name in model_paths:
            # Takes in local cptk file path but will probably want to expand to support weights and biases model
            mod = LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                    model_name, model=model_arch
                )
            mod = mod.model
            mod.eval()
            models.append(mod)
    else:
        model_arch.to('cuda')
        model_arch.eval()
        models.append(model_arch)
    exchangeability_test(models, val_loader, no_permutations=20, device='cuda', use_autoreg_eq=False)
