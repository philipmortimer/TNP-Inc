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
    nc, nt = 10, 20
    context_range = [[-2.0, 2.0]]
    target_range = [[-4.0, 4.0]]
    samples_per_epoch = 1
    batch_size = 1
    deterministic = True
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=nc, max_nc=nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=0.1,
        deterministic=True, kernel=kernels)
    data = next(iter(gen_val))
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
    print("starts")
    perms, log_p, (data_time, inference_time, total_time) = gather_rand_perms(models[0], data.xc, data.yc, data.xt, data.yt, 
        no_permutations=3000000, device='cuda', batch_size=2048)
    print(log_p)
    print(f"Data time: {data_time:.2f}s, Inference time: {inference_time:.2f}s, Total time: {total_time:.2f}s")