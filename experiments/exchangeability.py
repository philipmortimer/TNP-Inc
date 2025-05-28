# Measures sensitivity of TNP model to exchanging order of data based on eq 5 from https://proceedings.mlr.press/v253/mlodozeniec24a.html
import numpy as np
import torch
from scipy import stats
from check_shapes import check_shapes
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.lightning_utils import LitWrapper

# Computes log joint variance of model
@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]"
)
@torch.no_grad()
def m_var(tnp_model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, perms_ctx: torch.Tensor):
    # perms_ctx = [K, nc] - K permutations for nc context points with their indices in the tensor
    perms_ctx = perms_ctx.to(xc.device)
    K = perms_ctx.shape[0]
    m, nc = xc.shape[0], xc.shape[1]

    # Loops through each permutation - note tried stacking them to make faster but had weird bugs. This is also more readable.
    log_probs_list = []
    for k in range(K):
        xc_perm = xc[:, perms_ctx[k], :]
        yc_perm = yc[:, perms_ctx[k], :]
        log_p = tnp_model(xc_perm, yc_perm, xt).log_prob(yt) # [m, nt, dy]
        log_p = log_p.sum(dim=(-1, -2)) # nt and dy giving shape [m]
        log_probs_list.append(log_p)
    log_probs = torch.stack(log_probs_list, dim=0) # [K, m]
    variance = log_probs.var(dim=0, unbiased=True) # Variance over K - this is Var_PI from eq 5 in paper (this gives [m])
    m_var_val = variance.mean().item() # Mean over m batches sampled from D^(n) - ie the monte-carlo approximation of the expectation
    # Note - in this current implementation we only permutes the final full context. We don't account for incremental context points.
    # This may need reconsidering when we consider the masked variant with incremental learning. 
    # This func computes E_{(C, T) ~ D^(n)}[Var_{PI}[sigma_{j=1}^{nt} log p_{\theta}(y_{t, j} | C_{PI}, x_{t, j})]]
    # when using a non-diagonal TNP. The inner expression changes when using different NPs (e.g. autoreg) and this needs to be considered also.
    return m_var_val


# Samples variance over trained models with different seeds
def exchange(models_with_different_seeds, data_loader, no_permutations, device):
    no_models = len(models_with_different_seeds)

    m_vars = []
    for data in data_loader:
        xc, yc, xt, yt = data.xc, data.yc, data.xt, data.yt
        xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
        nc = xc.shape[1]
        perms_ctx = torch.stack([torch.randperm(nc, device=device) for _ in range(no_permutations)])
        mods_out = []
        # Computes m_var for each model
        for model  in models_with_different_seeds:
            mods_out.append(m_var(model, xc, yc, xt, yt, perms_ctx))
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
def exchangeability_test(models, data, no_permutations=128, device='cuda'):
    assert no_permutations >= 2, "Must have at least 2 permutations to compute variance"
    # Logs exchangeability
    mean, half_w = exchange(models, data, no_permutations, device)
    if half_w is None: half_w = 'N/A'
    print(f"Exchangeability: {mean} +/- {half_w}")

if __name__ == "__main__":
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
    use_weights = True # Defines if local checkpoints are to be used
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
    exchangeability_test(models, val_loader, no_permutations=2, device='cuda')
