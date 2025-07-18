# A streamed GP model implemented based on https://arxiv.org/pdf/1705.07131 and https://github.com/thangbui/streaming_sparse_gp
from __future__ import annotations
from typing import Tuple, Optional
from check_shapes import check_shapes
from gpytorch.utils.cholesky import psd_safe_cholesky

import torch
from torch import nn
import numpy as np
import gpytorch
from gpytorch.constraints import Interval, GreaterThan


# Adds jitter to match gpflow func - helper
def _add_jitter(K: torch.Tensor, jitter: float) -> torch.Tensor:
    return K + jitter * torch.eye(K.size(0), dtype=K.dtype, device=K.device)


# Designed to mimic https://github.com/thangbui/streaming_sparse_gp/blob/master/code/osgpr.py OSGPR_VFE class but in PyTorch as closely as possible to use within NP env
class OSGPR_VFE(nn.Module):
    def __init__(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        kernel: gpytorch.kernels.Kernel,
        mu_old: torch.Tensor,
        Su_old: torch.Tensor,
        Kaa_old: torch.Tensor,
        Z_old: torch.Tensor,
        Z: torch.Tensor,
        external_noise_param=None, # Noise so that it can be used across optimisation steps
        mean_function: Optional[gpytorch.means.Mean] = None,
        jitter: float = 1e-4, # Small jitter by default
        device: str = "cuda",
        dtype: torch.dtype = torch.float64, # Data type using float 64 for numerical stability by default
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.X, self.Y = data # Unpacks the data
        self.X = self.X.to(self.device, dtype)
        self.Y = self.Y.to(self.device, dtype)
        if self.Y.ndim == 1:
            self.Y = self.Y.unsqueeze(-1) # Unsqueeze to [N, 1]

        # Sets up the GP - mean func defaults to zero mean
        self.kernel = kernel.to(self.device, dtype)
        self.mean_function = (
            mean_function.to(self.device, dtype)
            if mean_function is not None
            else gpytorch.means.ZeroMean().to(self.device, dtype)
        )

        # Uses external noise to be optimised if needed
        if external_noise_param is None:
            self.likelihood_variance = nn.Parameter(torch.tensor(1, dtype=dtype, device=self.device))
        else:
            self.likelihood_variance = external_noise_param

        self.inducing_variable = nn.Parameter(Z.to(self.device, dtype)) # Inducing points

        # Caches old data
        self.register_buffer("mu_old", mu_old.to(self.device, dtype), persistent=False)
        if self.mu_old.ndim == 1:
            self.mu_old = self.mu_old.unsqueeze(-1) # [Ma, 1]
        self.register_buffer("Su_old", Su_old.to(self.device, dtype), persistent=False)
        self.register_buffer("Kaa_old", Kaa_old.to(self.device, dtype), persistent=False)
        self.register_buffer("Z_old", Z_old.to(self.device, dtype), persistent=False)

        self.num_data = self.X.size(0)
        self.M_old = self.Z_old.size(0)
        self.jitter = jitter

    def _common_terms(self):
        self.kernel._disable_cache = True
        Mb = self.inducing_variable.size(0)
        sigma2 = self.likelihood_variance
        sigma = torch.sqrt(sigma2)
        jitter = self.jitter

        Kbf = self.kernel(self.inducing_variable, self.X).evaluate()
        Kbb = _add_jitter(self.kernel(self.inducing_variable).evaluate(), jitter)
        Kba = self.kernel(self.inducing_variable, self.Z_old).evaluate()
        Kaa_cur = _add_jitter(self.kernel(self.Z_old).evaluate(), jitter)
        Kaa = _add_jitter(self.Kaa_old, jitter)

        err = self.Y - self.mean_function(self.X)
        if err.ndim == 1:
            err = err.unsqueeze(-1)

        Sainv_ma = torch.linalg.solve(self.Su_old, self.mu_old) # [Ma, 1]
        Sinv_y = self.Y / sigma2  # [N, 1]
        c1 = Kbf @ Sinv_y
        c2 = Kba @ Sainv_ma
        c = c1 + c2 # [Mb, 1]

        #Lb = torch.linalg.cholesky(Kbb) # [Mb, Mb]
        Lb = psd_safe_cholesky(Kbb, jitter=jitter)
        Lbinv_c = torch.linalg.solve_triangular(Lb, c, upper=False)
        Lbinv_Kba = torch.linalg.solve_triangular(Lb, Kba, upper=False)
        Lbinv_Kbf = torch.linalg.solve_triangular(Lb, Kbf, upper=False) / sigma
        d1 = Lbinv_Kbf @ Lbinv_Kbf.T

        LSa = torch.linalg.cholesky(self.Su_old)
        Kab_Lbinv = Lbinv_Kba.T
        LSainv_Kab_Lbinv = torch.linalg.solve_triangular(LSa, Kab_Lbinv, upper=False)
        d2 = LSainv_Kab_Lbinv.T @ LSainv_Kab_Lbinv

        #La = torch.linalg.cholesky(Kaa)
        La = psd_safe_cholesky(Kaa, jitter=jitter)
        Lainv_Kab_Lbinv = torch.linalg.solve_triangular(La, Kab_Lbinv, upper=False)
        d3 = Lainv_Kab_Lbinv.T @ Lainv_Kab_Lbinv

        D = torch.eye(Mb, dtype=self.dtype, device=self.device) + d1 + d2 - d3
        D = _add_jitter(D, jitter)
        LD = psd_safe_cholesky(D, jitter=jitter)
        #LD = torch.linalg.cholesky(D)

        LDinv_Lbinv_c = torch.linalg.solve_triangular(LD, Lbinv_c, upper=False)

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_Lbinv_c, err, d1)


    # VFE bound
    def maximum_log_likelihood_objective(self) -> torch.Tensor:
        jitter = self.jitter
        sigma2 = self.likelihood_variance
        N = self.num_data

        # a is old inducing points, b is new and f is training points.
        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba, LDinv_Lbinv_c, err, Qff) = self._common_terms()

        #LSa = torch.linalg.cholesky(self.Su_old)
        LSa = psd_safe_cholesky(self.Su_old, jitter=jitter)
        Lainv_ma = torch.linalg.solve_triangular(LSa, self.mu_old, upper=False)

        # Constant term
        bound = -0.5 * N * torch.log(torch.tensor(2.0 * torch.pi, dtype=self.dtype, device=self.device))
        # Quadratic term
        bound += -0.5 * (err**2).sum() / sigma2
        bound += -0.5 * (Lainv_ma**2).sum()
        bound += 0.5 * (LDinv_Lbinv_c**2).sum()
        # Log det term
        bound += -0.5 * N * torch.log(sigma2)
        bound += -torch.log(torch.diag(LD)).sum()

        Kfdiag = self.kernel(self.X, diag=True)
        bound += -0.5 * Kfdiag.sum() / sigma2
        bound += 0.5 * torch.diag(Qff).sum()

        # Delta 2: a and b difference
        bound += torch.log(torch.diag(La)).sum()
        bound += -torch.log(torch.diag(LSa)).sum()

        Kaadiff = Kaa_cur - Lbinv_Kba.T @ Lbinv_Kba
        Sainv_Kaadiff = torch.linalg.solve(self.Su_old, Kaadiff)
        Kainv_Kaadiff = torch.linalg.solve(Kaa, Kaadiff)
        bound += -0.5 * (torch.diag(Sainv_Kaadiff) - torch.diag(Kainv_Kaadiff)).sum()

        return bound  # To be maxed wrt theta and Z

    
    @torch.no_grad()
    def predict_f(self, Xnew: torch.Tensor, full_cov: bool = False):
        Xnew = Xnew.to(self.device, self.dtype)
        jitter = self.jitter

        # a is old inducing points, b is new and f is training points. s is test points.
        Kbs = self.kernel(self.inducing_variable, Xnew).evaluate()


        (_Kbf, _Kba, _Kaa, _Kaa_cur, _La, Kbb, Lb, _D, LD, _Lbinv_Kba, LDinv_Lbinv_c, _err, _) = self._common_terms()

        Lbinv_Kbs = torch.linalg.solve_triangular(Lb, Kbs, upper=False)
        LDinv_Lbinv_Kbs = torch.linalg.solve_triangular(LD, Lbinv_Kbs, upper=False)

        mean = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_c # [Ns, 1]

        if full_cov:
            Kss = (
                self.kernel(Xnew).evaluate()
                + jitter * torch.eye(Xnew.size(0), dtype=self.dtype, device=self.device)
            )
            var1 = Kss
            var2 = - Lbinv_Kbs.T @ Lbinv_Kbs
            var3 = LDinv_Lbinv_Kbs.T @ LDinv_Lbinv_Kbs
            var = var1 + var2 + var3
        else:
            var1 = self.kernel(Xnew, diag=True)
            var2 = - (Lbinv_Kbs**2).sum(0)
            var3 = (LDinv_Lbinv_Kbs**2).sum(0)
            var = var1 + var2 + var3

        mf = self.mean_function(Xnew)
        if mf.ndim == 1: # Output shape is correct if needed
            mf = mf.unsqueeze(-1)

        return mean + mf, var


# Wrapper class to use with rbf kernel
class GPStreamSparseWrapperRBF(nn.Module):
    def __init__(
        self,
        num_inducing: int,
        lr: float = 0.0001,
        num_steps: int = 10,
        chunk_size: Optional[int] = None,
        jitter: float = 1e-3,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64, # Changed default to float64
    ):
        super().__init__()
        self.M = num_inducing
        self.lr = lr
        self.num_steps = num_steps
        self.chunk_size = chunk_size
        self.jitter = jitter
        self.device = device
        self.dtype = dtype

    def forward(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor) -> torch.distributions.MultivariateNormal:
        m, nc, dx = xc.shape
        chunks = self.chunk_size if self.chunk_size is not None else nc

        mean_list, covar_list = [], []
        for i in range(m):
            xci, yci, xti = xc[i], yc[i], xt[i]

            # Creates params for this stream
            kernel = gpytorch.kernels.RBFKernel().to(self.device, self.dtype)
            mean_function = gpytorch.means.ZeroMean().to(self.device, self.dtype)
            indices = torch.randperm(xci.shape[0])[:self.M] # Random initial inducing points
            inducing_variable = nn.Parameter(xci[indices].clone().detach())

            # Fresh prior state each time
            noise_param = nn.Parameter(torch.tensor(0.01, dtype=self.dtype, device=self.device))
            mu_old = torch.zeros(self.M, 1, device=self.device, dtype=self.dtype)
            Su_old = torch.eye(self.M, device=self.device, dtype=self.dtype)
            Kaa_old = kernel(inducing_variable).evaluate().detach()
            Z_old = inducing_variable.detach().clone()

            optimizer = torch.optim.Adam([
                {'params': kernel.parameters()},
                {'params': inducing_variable},
                {'params': noise_param},
            ], lr=self.lr)

            # Streams one chunk at a time
            lower = 0
            final_model_for_prediction = None
            while lower < nc:
                upper = min(lower + chunks, nc)
                xc_sub, yc_sub = xci[lower:upper], yci[lower:upper]

                # Optimisation for current chunk
                for learn_step in range(self.num_steps):
                    # Creates new model each time but with correct params to prevent graph issues
                    model_step = OSGPR_VFE(
                        data=(xc_sub, yc_sub), kernel=kernel, mu_old=mu_old, Su_old=Su_old,
                        Kaa_old=Kaa_old, Z_old=Z_old, Z=inducing_variable,
                        external_noise_param=noise_param,
                        mean_function=mean_function, jitter=self.jitter,
                        device=self.device, dtype=self.dtype
                    )
                    
                    optimizer.zero_grad()
                    loss = -model_step.maximum_log_likelihood_objective()
                    print(f'{loss} - {learn_step}/{self.num_steps}')
                    loss.backward()
                    optimizer.step()

                    #optimizer.param_groups.pop()

                # Updates the old states so next chunk can use posterior
                with torch.no_grad():
                    final_model_chunk = OSGPR_VFE(
                        data=(xc_sub, yc_sub), kernel=kernel, mu_old=mu_old, Su_old=Su_old,
                        Kaa_old=Kaa_old, Z_old=Z_old, Z=inducing_variable,
                        mean_function=mean_function, external_noise_param=noise_param, jitter=self.jitter,
                        device=self.device, dtype=self.dtype
                    )
                    mu_old, Su_old = final_model_chunk.predict_f(inducing_variable, full_cov=True)
                    Kaa_old = kernel(inducing_variable).evaluate().detach()
                    Z_old = inducing_variable.detach().clone()
                
                lower = upper
            
            # Final model prediction
            with torch.no_grad():
                mu, var_f = final_model_chunk.predict_f(xti, full_cov=True)
                var_y = var_f + final_model_chunk.likelihood_variance * torch.eye(xti.size(0), device=self.device, dtype=self.dtype)
                mean_list.append(mu.squeeze(-1))
                covar_list.append(var_y)
        mean_pred = torch.stack(mean_list, dim=0)
        covar_out = torch.stack(covar_list, dim=0)
        return torch.distributions.MultivariateNormal(loc=mean_pred, covariance_matrix=covar_out)
