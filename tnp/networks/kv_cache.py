# Helper to make KV cache updates easy
from typing import Optional
import torch

def update_kv_cache(k_new, v_new, cache: Optional[dict], cache_id):
    if cache is None: return k_new, v_new # Training - in case of an empty cache, k and v are simply returned

    res = cache.get(cache_id, (None, None))
    k, v, toks_written = cache.get(cache_id, (None, None, None)) # Gets previously cached k and v values (or (None, None) if layer has not been cached yet)

    # Adds k and v to cache history
    toks_written_new = toks_written + k_new.shape[2]
    k[:, :, toks_written:toks_written+toks_written_new, :] = k_new
    v[:, :, toks_written:toks_written+toks_written_new, :] = v_new
    cache[cache_id] = (k, v, toks_written_new) # Updates cache
    return k[:, : , :toks_written_new, :], v[:, : , :toks_written_new, :]

def get_max_sized_mask(cache):
    if cache is None: return None
    return cache.get('context_big_mask', None)


# Initialises a KV cache with a max sequence length
def init_kv_cache(L, m, v_dim, k_dim, max_len, no_heads, device, nc, dz) -> dict:
    kv_cache = {} # Empty cache
    # Creates big mask for easy slicing
    mask_sa = torch.tril(torch.ones(nc + 1, nc + 1, dtype=torch.bool, device=device))
    mask_sa = mask_sa.unsqueeze(0).expand(m, -1, -1)
    kv_cache['context_big_mask'] = mask_sa
    for i in range(L):
        # Initialises per layer K and V cache
        self_attention_layer_tag = f"layer_{i}_sa"
        k_empty = torch.empty((m, no_heads, max_len, k_dim), device=device, dtype=torch.float32)
        v_empty = torch.empty((m, no_heads, max_len, v_dim), device=device, dtype=torch.float32)
        kv_cache[self_attention_layer_tag] = (k_empty, v_empty, 0)
        # Initialises context reps
        ctx_tag = f"context_layer_{i}"
        ctx_tensor = torch.empty((m, nc + 1, dz), device=device, dtype=torch.float32)
        kv_cache[ctx_tag] = (ctx_tensor, 0)
    return kv_cache
