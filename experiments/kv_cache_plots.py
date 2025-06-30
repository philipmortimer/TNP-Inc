# File that tests kv caching and plots it to show speedup
import torch
from plot_adversarial_perms import get_model
import torch.distributions as td
from tnp.networks.kv_cache import init_kv_cache, update_kv_cache
import numpy as np
import time

# Tests that KV caching works exactly the same as without
@torch.no_grad
def test_kv_cache():
    atol=1e-4 # Tolerance for close checks
    rtol = 1e-4
    device='cuda'
    # Fetches model
    model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', device=device)
    model.eval()
    # Generates random dataset
    N, m, nc, nt, dx, dy = 100, 16, 32, 128, 1, 1
    max_high = 2
    xcs = (torch.rand((N, m, nc, dx), device=device) * (2 * max_high)) - max_high
    ycs = (torch.rand((N, m, nc, dy), device=device) * (2 * max_high)) - max_high
    xts = (torch.rand((N, m, nt, dx), device=device) * (2 * max_high)) - max_high
    yts = (torch.rand((N, m, nt, dy), device=device) * (2 * max_high)) - max_high
    # Loops through data and asserts that with / without KV caching produce the same model prediction
    for i in range(N):
        # Initialises KV cache with start token
        start_token = model.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        kv_cache = init_kv_cache()
        model.encoder.update_context(start_token, kv_cache)
        xc, yc, xt, yt = xcs[i,:,:,:], ycs[i,:,:,:], xts[i,:,:,:], yts[i,:,:,:]
        # Number of context tokens conditioned on
        for ctx_toks in range(nc):
            xc_red, yc_red = xc[:, :ctx_toks, :], yc[:, :ctx_toks, :]
            xc_new, yc_new = xc[:, ctx_toks:ctx_toks+1, :], yc[:, ctx_toks:ctx_toks+1, :]
            # Non KV-cached
            pred_dist_non_cached = model.likelihood(model.decoder(model.encoder(xc=xc_red, yc=yc_red, xt=xt), xt))
            # KV cached
            zt = model.encoder._preprocess_targets(xt, dy)
            pred_dist_kv_cached = model.likelihood(model.decoder(model.encoder.query(zt, kv_cache), xt))
            new_zc = model.encoder._preprocess_context(xc_new, yc_new)
            model.encoder.update_context(new_zc, kv_cache)
            # Checks that distributions are same
            assert isinstance(pred_dist_non_cached, td.Normal) and isinstance(pred_dist_kv_cached, td.Normal), "Both should be normal predictions"
            assert torch.allclose(pred_dist_non_cached.mean, pred_dist_kv_cached.mean, atol=atol, rtol=rtol), "Dist means must be same"
            assert torch.allclose(pred_dist_non_cached.stddev, pred_dist_kv_cached.stddev, atol=atol, rtol=rtol), "Dist std must be same"#
    print("KV Cache tests all passed")

    
# Measures the conditioning time for the model
@torch.no_grad
def measure_condition_time_memory_kv():
    # Gets model
    device='cuda'
    model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', device=device)
    model.eval()
    # Dataset
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 2 # Number of runs to aggregate data over
    token_step = 50 # How many increments of tokens to go up in
    max_nc, dx, dy, m = 10_000, 1, 1, 1
    max_high = 2
    xcs = (torch.rand((1, max_nc, dx), device=device) * max_high * 2) - max_high
    ycs = (torch.rand((1, max_nc, dy), device=device) * max_high * 2) - max_high
    # Results structures
    context_sizes = np.arange(start=0, stop=max_nc, step=token_step, dtype=int)
    upper_ctxs = np.array([min(i + token_step, max_nc) for i in context_sizes])
    runtime = np.zeros((aggregate_over, len(context_sizes)))
    memory = np.zeros((aggregate_over, len(context_sizes)))
    ctx_inc = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for j in range(burn_in + aggregate_over):
        # Initailises  KV cache
        start_token = model.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        kv_cache = init_kv_cache()
        model.encoder.update_context(start_token, kv_cache)
        torch.cuda.reset_peak_memory_stats() #  Resets memory stats - we want cumulative memory
        # Adds context tokens n at a time
        ctx_inc = 0
        for lower_ctx, upper_ctx in zip(context_sizes, upper_ctxs):
            xc_new = xcs[:, lower_ctx:upper_ctx, :]
            yc_new = ycs[:, lower_ctx:upper_ctx, :]

            # Sets up measures
            torch.cuda.synchronize()
            starter.record()
            # Core update step
            with torch.no_grad():
                new_zc = model.encoder._preprocess_context(xc_new, yc_new)
                model.encoder.update_context(new_zc, kv_cache)
            # Measures time and memory
            ender.record()
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            update_time_ms = starter.elapsed_time(ender)

            # Writes measured results
            if j >= burn_in:
                write_j = j - burn_in
                runtime[write_j, ctx_inc] = update_time_ms
                memory[write_j, ctx_inc] = peak_memory_mb
            ctx_inc += 1
    # Averages runtime and memory
    runtime = np.mean(runtime, axis=0)
    memory = np.mean(memory, axis=0)

    # Writes results to file
    summary_block = f"""
    ----------------------------
    Cumulative Context Size: {upper_ctxs}
    Runtime Incremental (ms): {runtime}
    Memory Cumulative (Mb): {memory}
    """
    print(summary_block)
    with open('experiments/plot_results/kv/' + 'mem_run.txt', 'w') as file_object:
        file_object.write(summary_block)
    
            




if __name__ == "__main__":
    #test_kv_cache()
    measure_condition_time_memory_kv()
