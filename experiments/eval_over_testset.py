# Core idea is to measure the peformance of models over a given test set (especially order senstivie models and batching strategies for that)
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
from typing import Optional, Union
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
import matplotlib.patheffects as pe
from tnp.data.base import Batch
from tnp.models.incTNPBatchedPrior import IncTNPBatchedPrior
from plot_adversarial_perms import get_model
from tqdm import tqdm


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

def shuffle_batch(model, batch, shuffle_strategy: str, device: str="cuda"):
    assert shuffle_strategy in {"random", "GreedyBestPrior", "GreedyWorstPrior", "GreedyMedianPrior"}, "Invalid context shuffle strategy"
    m, nc, dx = batch.xc.shape
    _, nt, dy = batch.yt.shape
    # Converts batch to cuda
    batch.xc, batch.yc, batch.xt, batch.yt, batch.x, batch.y = batch.xc.to(device), batch.yc.to(device), batch.xt.to(device), batch.yt.to(device), batch.x.to(device), batch.y.to(device)
    xc_new, yc_new = None, None
    if shuffle_strategy == "random":
        perms = torch.rand(m, nc, device=batch.xc.device).argsort(dim=1)
        perm_x = perms.unsqueeze(-1).expand(-1, -1, dx)
        perm_y = perms.unsqueeze(-1).expand(-1, -1, dy)
        xc_new = torch.gather(batch.xc, 1, perm_x) 
        yc_new = torch.gather(batch.yc, 1, perm_y)
    elif shuffle_strategy == "GreedyBestPrior":
        xc_new, yc_new = model.kv_cached_greedy_variance_ctx_builder(batch.xc, batch.yc, policy="best")
        #xc_new, yc_new = model.greedy_variance_ctx_builder(batch.xc, batch.yc, policy="best")
    elif shuffle_strategy == "GreedyWorstPrior":
        xc_new, yc_new = model.greedy_variance_ctx_builder(batch.xc, batch.yc, policy="worst")
    elif shuffle_strategy == "GreedyMedianPrior":
        xc_new, yc_new = model.greedy_variance_ctx_builder(batch.xc, batch.yc, policy="median")
    
    x = torch.cat((xc_new, batch.xt), dim=1)
    y = torch.cat((yc_new, batch.yt), dim=1)
    batch_new = Batch(xc=xc_new, yc=yc_new, xt=batch.xt, yt=batch.yt, y=y, x=x)
    return batch_new


# Evaluates a given models performance. Includes the option for a small number of defined strategies.
@torch.no_grad
def eval_model(model, test_set, shuffle_strategy):
    log_liks, rmse_vals, gt_log_liks = [], [], []
    for batch_test in tqdm(test_set, desc="Evaluating Model on One Train Set"):
        # Shuffles data using defined permute strategy
        batch = shuffle_batch(model, batch_test, shuffle_strategy)
        # Model LL and rmse
        m, nt, _ = batch.yt.shape
        # Gets predictive distribution from model
        pred_dist = np_pred_fn(model, batch, predict_without_yt_tnpa=True)
        ll = pred_dist.log_prob(batch.yt).sum() / (m * nt)
        rmse = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu()

        # GT LL - may need to wrap this in if statement in future datasets
        _, _, gt_loglik = batch_test.gt_pred(
            xc=batch.xc, yc=batch.yc, xt=batch.xt, yt=batch.yt
        )
        gt_loglik = gt_loglik.sum() / (m * nt)

        # Adds data
        log_liks.append(ll.item())
        rmse_vals.append(rmse.item())
        gt_log_liks.append(gt_loglik.item())

    # Gathers results
    results = {
        "mean_ll": np.mean(log_liks),
        "std_ll": np.std(log_liks),
        "mean_rmse": np.mean(rmse_vals),
        "std_rmse": np.std(rmse_vals),
        "mean_gt_ll": np.mean(gt_log_liks),
        "std_gt_ll": np.std(gt_log_liks),
    }
    return results

# Evaluates model performance over a number of passes of the test set (e.g. to account for noise etc)
def eval_model_over_permutations(model, test_set, no_reps: int = 1, shuffle_strategy: str = "random"):
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    mean_lls, std_lls, mean_rmse_vals, std_rmse_vals, mean_gt_lls, std_gt_lls = [], [], [], [], [], []
    for _ in tqdm(range(no_reps), desc="Evaluating model over multiple test sets"):
        result = eval_model(model, test_set, shuffle_strategy)
        mean_lls.append(result["mean_ll"])
        std_lls.append(result["std_ll"])
        mean_rmse_vals.append(result["mean_rmse"])
        std_rmse_vals.append(result["std_rmse"])
        mean_gt_lls.append(result["mean_gt_ll"])
        std_gt_lls.append(result["std_gt_ll"])
    # Gathers results
    results = {
        "num_params": num_params,
        "mean_lls": mean_lls,
        "std_lls": std_lls,
        "mean_rmse_vals": mean_rmse_vals,
        "std_rmse_vals": std_rmse_vals,
        "mean_gt_lls": mean_gt_lls,
        "std_gt_lls": std_gt_lls,
    }
    return results
        

# Gets rbf kernel with rangesame default test params used
def get_rbf_rangesame_test_set():
    # RBF kernel params
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    # Data generator params for test set
    min_nc = 1
    max_nc = 64
    nt= 128
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4096
    batch_size = 16
    noise_std = 0.1
    deterministic = True
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    test_set = [batch for batch in gen_test]
    return test_set, "RBF"

# Main function used to handle flow of evaluating model and plotting the results
def models_perf_main(model_list, test_data):
    test_set, test_data_name = test_data
    folder_name = "experiments/plot_results/eval_set/"
    txt_file_summary = f'Summary over eval data set {test_data_name}'
    for (yml_path, wandb_id, shuffle_strategy, model_name, special_args, no_reps) in model_list:
        model = get_model(yml_path, wandb_id, seed=False) # Loads model
        if special_args.startswith("TNPAR_"):
            model.num_samples = int(special_args.split("_")[1])
        results = eval_model_over_permutations(model, test_set, no_reps, shuffle_strategy)

        summary_block = f"""
        ----------------------------
        Model: {model_name}
        Params: {results["num_params"]}
        Mean_LLs: {results["mean_lls"]}
        Std_LLs: {results["std_lls"]}
        Mean_RMSEs: {results["mean_rmse_vals"]}
        Std_RMSEs: {results["std_rmse_vals"]}
        Mean_GT_LLs: {results["mean_gt_lls"]}
        Std_GT_LLs: {results["std_gt_lls"]}
        """
        txt_file_summary += summary_block
        print(summary_block)
    with open(folder_name + 'eval_summary.txt', 'w') as file_object:
        file_object.write(txt_file_summary)

# List of models to be tested, adjust this as required
def get_model_list():
    # Models available
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/plain-tnp-rbf-rangesame/model-7ib3k6ga:v200', 'random', "TNP-D", "",
        1)
    tnp_causal = ('experiments/configs/synthetic1dRBF/gp_causal_tnp_rangesame.yml', 
        'pm846-university-of-cambridge/mask-tnp-rbf-rangesame/model-vavo8sh2:v200', 'random', "IncTNP", "",
        1)
    tnp_causal_batched = ('experiments/configs/synthetic1dRBF/gp_batched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-batched-tnp-rbf-rangesame/model-xtnh0z37:v200', 'random', "IncTNP (Batched)", "",
        1)
    tnp_causal_batched_prior = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'random', "IncTNP-Prior (Batched)", "",
        1)
    # TNP Causal Batched Prior Greedy Strategies
    greedy_best_tnp_causal_batched_prior = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyBestPrior', 
        "IncTNP-Prior (Batched) - Best Greedy", "",
        1)
    greedy_worst_tnp_causal_batched_prior = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyWorstPrior', 
        "IncTNP-Prior (Batched) - Worst","",
        1) 
    greedy_median_tnp_causal_batched_prior = ('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 
        'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', 'GreedyMedianPrior', 
        "IncTNP-Prior (Batched) - Best Median", "",
        1)
    # TNP AR models
    ar_yml, ar_mod, name = 'experiments/configs/synthetic1dRBF/gp_tnpa_rangesame.yml', 'pm846-university-of-cambridge/tnpa-rbf-rangesame/model-wbgdzuz5:v200', "TNP-A"
    tnp_ar_5 = (ar_yml, ar_mod, 'random', name + " (5)", "TNPAR_5", 2)
    tnp_ar_50 = (ar_yml, ar_mod, 'random', name + " (50)", "TNPAR_50", 2)
    tnp_ar_100 = (ar_yml, ar_mod, 'random', name + " (100)", "TNPAR_100", 2)
    
    # Defines models to be used
    models = [tnp_plain, tnp_causal, tnp_causal_batched, tnp_causal_batched_prior, 
        greedy_best_tnp_causal_batched_prior, greedy_worst_tnp_causal_batched_prior, greedy_median_tnp_causal_batched_prior,
        tnp_ar_5, tnp_ar_50, tnp_ar_100]
    models = [greedy_best_tnp_causal_batched_prior]
    return models

if __name__ == "__main__":
    start_t = time.time()
    pl.seed_everything(1) #  Sets seed of randomness for reproducibility
    models_perf_main(get_model_list(), get_rbf_rangesame_test_set())
    print(f'Runtime: {time.time()-start_t:.2f}s')
    