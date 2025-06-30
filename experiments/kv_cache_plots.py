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
    atol=1e-5 # Tolerance for close checks
    rtol = 1e-4
    device='cuda'
    # Fetches model
    model = get_model('experiments/configs/synthetic1dRBF/gp_priorbatched_causal_tnp_rbf_rangesame.yml', 'pm846-university-of-cambridge/mask-priorbatched-tnp-rbf-rangesame/model-smgj3gn6:v200', device=device)
    model.eval()
    # Generates random dataset
    N, m, nc, nt, dx, dy = 100, 16, 32, 128, 1, 1
    xcs = torch.randn((N, m, nc, dx), device=device)
    ycs = torch.randn((N, m, nc, dy), device=device)
    xts = torch.randn((N, m, nt, dx), device=device)
    yts = torch.randn((N, m, nt, dy), device=device)
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
            assert torch.allclose(pred_dist_non_cached.stddev, pred_dist_kv_cached.stddev, atol=atol, rtol=rtol), "Dist std must be same"
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
    token_step = 1 # How many increments of tokens to go up in
    max_nc, dx, dy, m = 1000, 1, 1, 1
    xcs = torch.randn((1, max_nc, dx), device=device)
    ycs = torch.randn((1, max_nc, dy), device=device)
    # Results structures
    context_sizes = np.arange(start=0, stop=max_nc, step=token_step, dtype=int)
    no_ctx_points = len(context_sizes)
    runtime = np.zeros((aggregate_over, no_ctx_points))
    memory = np.zeros((aggregate_over, no_ctx_points))
    ctx_inc = 0
    for j in range(burn_in + aggregate_over):
        # Initailises  KV cache
        start_token = model.encoder.empty_token.expand(m, -1, -1) # Starts with empty token (prior condition)
        dz = start_token.shape[2]
        L = len(model.encoder.transformer_encoder.mhsa_layers)
        head_dim = int(round(model.encoder.transformer_encoder.mhsa_layers[0].attn.scale ** -2))
        kv_cache = init_kv_cache(L=L, m=m,
            k_dim=head_dim,
            v_dim=head_dim,
            max_len=max_nc + 1, no_heads=model.encoder.transformer_encoder.mhsa_layers[0].attn.num_heads, device=device,
            nc=max_nc, dz=dz)
        model.encoder.update_context(start_token, kv_cache)
        # Adds context tokens n at a time
        ctx_inc = 0
        for i in context_sizes:
            up_lim = context_sizes[i + 1] if i  + 1 < len(context_sizes) else context_sizes[-1]
            xc_new = xcs[:, i:up_lim, :]
            yc_new = ycs[:, i:up_lim, :]

            # Sets up measures
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            # Core update step
            with torch.no_grad():
                new_zc = model.encoder._preprocess_context(xc_new, yc_new)
                model.encoder.update_context(new_zc, kv_cache)
            # Measures time and memory
            torch.cuda.synchronize()
            end = time.perf_counter()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            update_time = (end - start) * 1000

            # Writes measured results
            if j >= burn_in:
                write_j = j - burn_in
                runtime[write_j, ctx_inc] = update_time
                memory[write_j, ctx_inc] = peak_memory_mb
            ctx_inc += 1
    # Averages runtime and memory
    runtime = np.mean(runtime, axis=0)
    memory = np.mean(memory, axis=0)

    print(runtime)
    print(memory)
            




if __name__ == "__main__":
    test_kv_cache()
    measure_condition_time_memory_kv()
