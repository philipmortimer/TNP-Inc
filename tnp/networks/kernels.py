from typing import Callable

import gpytorch
import torch


class GibbsKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        lengthscale_fn: Callable[[torch.Tensor], torch.Tensor],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lengthscale_fn = lengthscale_fn

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ):
        x1_lengthscale = self.lengthscale_fn(x1)
        x2_lengthscale = self.lengthscale_fn(x2)
        lengthscale = (x1_lengthscale**2 + x2_lengthscale**2) ** 0.5
        const = ((2 * x1_lengthscale * x2_lengthscale) / lengthscale**2) ** 0.5

        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or last_dim_is_batch
            or gpytorch.settings.trace_mode.on()
        ):
            x1_ = x1.div(lengthscale)
            x2_ = x2.div(lengthscale)
            return const * self.covar_dist(
                x1_,
                x2_,
                square_dist=True,
                diag=diag,
                dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf,
                postprocess=True,
                last_dim_is_batch=last_dim_is_batch,
                **params,
            )
        return const * gpytorch.functions.RBFCovariance.apply(
            x1,
            x2,
            lengthscale,
            lambda x1, x2: self.covar_dist(
                x1,
                x2,
                square_dist=True,
                diag=False,
                dist_postprocess_func=gpytorch.kernels.rbf_kernel.postprocess_rbf,
                postprocess=False,
                last_dim_is_batch=False,
                **params,
            ),
        )
