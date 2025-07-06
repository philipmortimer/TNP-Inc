# A streamed Gaussian Process to compare to TNPs and provide a baseline
# Hyperparameters are updated in chunks and then exact GP conditioned on whole data is updated
import torch
from torch import nn
import gpytorch
from check_shapes import check_shapes
from typing import Callable, Optional, Literal


class ExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
    ):
        super().__init__(
            train_inputs=torch.zeros(1,1), # Dummy inputs
            train_targets=torch.zeros(1), # Dummy targets
            likelihood=likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# GP Stream class that handles logic of streaming context points to be used with np_pred_fun easily. Uses gaussian likelihood
class GPStream(nn.Module):
    def __init__(
        self,
        kernel_factory: Callable[[], gpytorch.kernels.Kernel], # GP Kernel (callable to create new one each time)
        lr: float = 0.05, # LR for grad updates
        n_steps: int = 10, # Number of grad steps per update
        chunk_size: int = 1, # Size of chunks to be streamed in to model
        train_strat: Literal["Expanding", "Sliding"] = "Expanding", # Whether to use an ever expanding window or a sliding one
        device: str = "cuda",
    ):
        super().__init__()
        self.lr = lr
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.device = device
        self.kernel_factory = kernel_factory
        self.strategy = train_strat

    # Streams data in chunks and updates the hypers of the gp for the given example. May want to alter this
    # eg do we want to see data update then continue or do we want to keep expanding the window (alleviates catastrophic forgetting but may fit too well)
    @check_shapes(
        "xc: [1, nc, dx]", "yc: [1, nc, 1]"
    )
    def _stream_through_gp(self, gp, likelihood, xc, yc):
        _, nc, dx = xc.shape
        assert nc >= self.chunk_size and nc % self.chunk_size == 0, f"Chunk size {self.chunk_size} should be smaller or equal to nc {nc} and divide perfectly"
        likelihood.train()
        gp.train()
        optimiser = torch.optim.Adam(gp.parameters(), lr=self.lr) # Adam optimiser chosen
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
        for i in range(0, nc, self.chunk_size):
            end = min(i + self.chunk_size, nc)
            cur_chunk_size = end - i

            if self.strategy == "Expanding":
                xc_sub = xc[:,:end,:].reshape(-1, dx) # [end, dx]
                yc_sub = yc[:,:end,:].reshape(-1) # [end]
            elif self.strategy == "Sliding":
                xc_sub = xc[:,i:end,:].reshape(-1, dx) # [cur_chunk_size, dx]
                yc_sub = yc[:,i:end,:].reshape(-1) # [cur_chunk_size]
            else: raise ValueError(f"Incorrect strategy: {self.strategy}")


            gp.set_train_data(inputs=xc_sub, targets=yc_sub, strict=False)
            # Performs update steps for hyper
            for _ in range(self.n_steps):
                optimiser.zero_grad()
                loss = -mll(gp(xc_sub), yc_sub)
                loss.backward()
                optimiser.step()
        likelihood.eval()
        gp.eval()

    
    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, 1]", "xt: [m, nt, dx]"
    )
    def forward(self, xc, yc, xt) -> gpytorch.distributions.MultitaskMultivariateNormal:
        xc, yc, xt = xc.to(self.device), yc.to(self.device), xt.to(self.device)
        m, nc, dx = xc.shape
        _, nt, _ = xt.shape
        # Each batch is given an independent GP
        means_list = []
        covars_list = []
        preds = []
        for i in range(m):
            # Creates new gp
            kernel = self.kernel_factory().to(self.device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            gp = ExactGP(likelihood=likelihood, kernel=kernel).to(self.device)

            self._stream_through_gp(gp, likelihood, xc[i:i+1,:,:], yc[i:i+1,:,:]) # Trains GP hypers
            xt_in = xt[i:i+1,:,:].reshape(nt,-1) # [nt, dx]
            with torch.no_grad():
                function_dist = gp(xt_in)
                pred_dist = likelihood(function_dist)
            means_list.append(pred_dist.mean)
            covars_list.append(pred_dist.covariance_matrix)
        batched_mean = torch.stack(means_list, dim=0)
        batched_covars = torch.stack(covars_list, dim=0)
        pred_dist = gpytorch.distributions.MultivariateNormal(batched_mean, batched_covars)
        return pred_dist


# RBF GP Stream
class GPStreamRBF(GPStream):
    def __init__(
        self,
        chunk_size: int = 1,
        lr: float = 0.05, # LR for grad updates
        n_steps: int = 10, # Number of grad steps per update
        train_strat: Literal["Expanding", "Sliding"] = "Expanding",
        device: str = "cuda",
    ):
        super().__init__(kernel_factory=(lambda: gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())),
            chunk_size=chunk_size, lr=lr, n_steps=n_steps, device=device, train_strat=train_strat)