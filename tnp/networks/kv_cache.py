# Helper to make KV cache updates easy
from typing import Optional
import torch

def update_kv_cache(k_new, v_new, cache: Optional[dict], cache_id):
    if cache is None: return k_new, v_new # Training - in case of an empty cache, k and v are simply returned

    res = cache.get(cache_id, (None, None))
    k, v, toks_written = cache.get(cache_id, (None, None, None)) # Gets previously cached k and v values (or (None, None) if layer has not been cached yet)

    # Adds k and v to cache history
    toks_written_new = toks_written + len(k_new)
    k[:, :, toks_written:toks_written+toks_written_new, :] = k_new
    v[:, :, toks_written:toks_written+toks_written_new, :] = v_new
    cache[cache_id] = (k, v, toks_written_new) # Updates cache
    return k[:, : , :toks_written_new, :], v[:, : , :toks_written_new, :]

# Initialises a KV cache with a max sequence length
def init_kv_cache(L, m, v_dim, k_dim, max_len, no_heads, device) -> dict:
    kv_cache = {} # Empty cache
    for i in range(L):
        self_attention_layer_tag = f"layer_{i}_sa"
        k_empty = torch.empty((m, no_heads, max_len, k_dim), device=device)
        v_empty = torch.empty((m, no_heads, max_len, v_dim), device=device)
        kv_cache[self_attention_layer_tag] = (k_empty, v_empty, 0)
    return kv_cache
