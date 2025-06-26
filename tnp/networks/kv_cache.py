# Helper to make KV cache updates easy
from typing import Optional
import torch

def update_kv_cache(k_new, v_new, cache: Optional[dict], cache_id):
    if cache is None: return k_new, v_new # Training - in case of an empty cache, k and v are simply returned

    k_curr, v_curr = cache.get(cache_id, (None, None)) # Gets previously cached k and v values (or (None, None) if layer has not been cached yet)
    
    if k_curr is None: 
        k_updated, v_updated = k_new, v_new # Layer has not yet been cached (first item in seq) so return k and v
    else:
        # Adds k and v to cache history
        k_updated = torch.cat((k_curr, k_new), dim=2) # [B, H, L, Dk]
        v_updated = torch.cat((v_curr, v_new), dim=2) # [B, H, L, Dv]
    cache[cache_id] = (k_updated, v_updated) # Updates cache
    return k_updated, v_updated

