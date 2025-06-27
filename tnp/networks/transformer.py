import copy
import warnings
from abc import ABC
from typing import Optional

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from .attention_layers import (
    MultiHeadCrossAttentionLayer,
    MultiHeadKRAttentionLayer,
    MultiHeadSelfAttentionLayer,
)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        num_layers: int,
    ):
        super().__init__()

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes("x: [m, n, d]", "mask: [m, n, n]", "return: [m, n, d]")
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for mhsa_layer in self.mhsa_layers:
            x = mhsa_layer(x, mask)

        return x


class TNPTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: Optional[MultiHeadSelfAttentionLayer] = None,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = (
            self.mhca_layers
            if mhsa_layer is None
            else _get_clones(mhsa_layer, num_layers)
        )

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc)
            elif isinstance(mhsa_layer, MultiHeadCrossAttentionLayer):
                xc = mhsa_layer(xc, xc)
            else:
                raise TypeError("Unknown layer type.")

            xt = mhca_layer(xt, xc)

        return xt

# Tnp Encoder that supports masked self attention and cross attention.
class TNPTransformerFullyMaskedEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: MultiHeadSelfAttentionLayer,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask_sa: [m, nc, nc]", "mask_ca: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask_sa: Optional[torch.Tensor] = None, mask_ca: Optional[torch.Tensor] = None,
        use_causal: bool = False,
    ) -> torch.Tensor:

        for i, (mhsa_layer, mhca_layer) in enumerate(zip(self.mhsa_layers, self.mhca_layers)):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc, mask=mask_sa, use_causal=use_causal)
            else:
                raise TypeError("Unknown layer type.")

            xt = mhca_layer(xt, xc, mask=mask_ca)

        return xt

    # Computes the MHSA representation with causal plus kv
    @check_shapes(
        "zc_new: [m, nc_new, dz]", "return: [L, m, nc_new, dz]"
    )
    def encode_context(self, zc_new: torch.Tensor, kv_cache: dict,
        use_causal: bool = False,) -> torch.Tensor:
        L = len(self.mhsa_layers)
        m, nc_new, dz = zc_new.shape
        ctx_vals = torch.empty((L, m, nc_new, dz), device=zc_new.device)
        for i, mhsa_layer in enumerate(self.mhsa_layers):
            self_attention_layer_tag = f"layer_{i}_sa" # Layer tag for KV
            zc_new = mhsa_layer(zc_new, kv_cache=kv_cache, kv_tag=self_attention_layer_tag, use_causal=use_causal)
            ctx_vals[i] = zc_new
        return ctx_vals

    # Query - runs MHCA pathway assuming MHSA attention has already been computed
    @check_shapes(
        "ctx: [L, m, nc, dz]", "zt: [m, nt, dz]", "return: [m, nt, dz]"
    )
    def query(self, ctx, zt) -> torch.Tensor:
        for i, mhca_layer in enumerate(self.mhca_layers):
            zt = mhca_layer(zt, ctx[i])
        return zt


# TNPTransformerEncoder but the mask is applied to the context (ie only the self attention)
class TNPTransformerMaskedEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhca_layer: MultiHeadCrossAttentionLayer,
        mhsa_layer: MultiHeadSelfAttentionLayer,
    ):
        super().__init__()

        self.mhca_layers = _get_clones(mhca_layer, num_layers)
        self.mhsa_layers = (
            self.mhca_layers
            if mhsa_layer is None
            else _get_clones(mhsa_layer, num_layers)
        )

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nc, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        for mhsa_layer, mhca_layer in zip(self.mhsa_layers, self.mhca_layers):
            if isinstance(mhsa_layer, MultiHeadSelfAttentionLayer):
                xc = mhsa_layer(xc, mask=mask)
            elif isinstance(mhsa_layer, MultiHeadCrossAttentionLayer):
                xc = mhsa_layer(xc, xc, mask=mask)
            else:
                raise TypeError("Unknown layer type.")

            xt = mhca_layer(xt, xc)

        return xt


class TNPKRTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        mhkr_layer: MultiHeadKRAttentionLayer,
    ):
        super().__init__()

        self.mhkr_layers = _get_clones(mhkr_layer, num_layers)

    @check_shapes(
        "xc: [m, nc, d]", "xt: [m, nt, d]", "mask: [m, nt, nc]", "return: [m, nt, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        for mhkr_layer in self.mhkr_layers:
            xt, xc = mhkr_layer(xt, xc)

        return xt


class BasePerceiverEncoder(nn.Module, ABC):
    def __init__(
        self,
        num_latents: int,
        mhsa_layer: MultiHeadSelfAttentionLayer,
        mhca_ctoq_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        """Base class for the Perceiver encoder.

        Args:
            num_latents (int): Number of latents.
            mhsa_layer (MultiHeadSelfAttentionLayer): MHSA layer between latents.
            mhca_ctoq_layer (MultiHeadCrossAttentionLayer): MHCA layer from context to latents.
            mhca_qtot_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to target.
            num_layers (int): Number of layers.
        """
        super().__init__()

        # Initialise latents.
        embed_dim = mhsa_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhsa_layers = _get_clones(mhsa_layer, num_layers)
        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class PerceiverEncoder(BasePerceiverEncoder):
    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for mhsa_layer, mhca_ctoq_layer, mhca_qtot_layer in zip(
            self.mhsa_layers, self.mhca_ctoq_layers, self.mhca_qtot_layers
        ):
            xq = mhca_ctoq_layer(xq, xc)
            xq = mhsa_layer(xq)
            xt = mhca_qtot_layer(xt, xq)

        return xt


class BaseISTEncoder(nn.Module, ABC):
    def __init__(
        self,
        num_latents: int,
        mhca_ctoq_layer: MultiHeadSelfAttentionLayer,
        mhca_qtoc_layer: MultiHeadCrossAttentionLayer,
        mhca_qtot_layer: MultiHeadCrossAttentionLayer,
        num_layers: int,
    ):
        """Base class for the IST encoder.

        Args:
            num_latents (int): Number of latents.
            mhca_ctoq_layer (MultiHeadSelfAttentionLayer): MHCA layer from context to latents.
            mhca_qtoc_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to context.
            mhca_qtot_layer (MultiHeadCrossAttentionLayer): MHCA layer from latents to target.
            num_layers (int): Number of layers.
        """
        super().__init__()

        embed_dim = mhca_ctoq_layer.embed_dim
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))

        self.mhca_ctoq_layers = _get_clones(mhca_ctoq_layer, num_layers)
        self.mhca_qtoc_layers = _get_clones(mhca_qtoc_layer, num_layers - 1)
        self.mhca_qtot_layers = _get_clones(mhca_qtot_layer, num_layers)


class ISTEncoder(BaseISTEncoder):
    @check_shapes(
        "xc: [m, nc, dx]", "xt: [m, nt, dx]", "mask: [m, nq, n]", "return: [m, nq, d]"
    )
    def forward(
        self, xc: torch.Tensor, xt: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            warnings.warn("mask is not currently being used.")

        xq = einops.repeat(self.latents, "l e -> m l e", m=xc.shape[0])
        for i, (mhca_ctoq_layer, mhca_qtot_layer) in enumerate(
            zip(self.mhca_ctoq_layers, self.mhca_qtot_layers)
        ):
            xq = mhca_ctoq_layer(xq, xc)
            xt = mhca_qtot_layer(xt, xq)

            if i < len(self.mhca_qtoc_layers):
                xc = self.mhca_qtoc_layers[i](xc, xq)

        return xt


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
