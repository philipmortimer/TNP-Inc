# A streamed Gaussian Process to compare to TNPs and provide a baseline
# Hyperparameters are updated in chunks and then exact GP conditioned on whole data is updated
import torch
from torch import nn
import gpytorch
from check_shapes import check_shapes


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
        device: str = "cuda",
    ):
        super().__init__()
        self.lr = lr
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.device = device
        self.kernel_factory = kernel_factory

        self.gp = None
        self.likelihood = None
        self.kernel = None

    # Streams data in chunks and updates the hypers of the gp for the given example. May want to alter this
    # eg do we want to see data update then continue or do we want to keep expanding the window (alleviates catastrophic forgetting but may fit too well)
    @check_shapes(
        "xc: [1, nc, dx]", "yc: [1, nc, 1]"
    )
    def _stream_through_gp(self, xc, yc):
        _, nc, dx = xc.shape
        assert nc >= self.chunk_size and nc % self.chunk_size == 0, f"Chunk size {self.chunk_size} should be smaller or equal to nc {nc} and divide perfectly"
        self.likelihood.train()
        self.gp.train()
        optimiser = torch.optim.Adam(self.gp.parameters(), lr=self.lr) # Adam optimiser chosen
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
        for i in range(0, nc, self.chunk_size):
            end = min(i + self.chunk_size, nc)
            cur_chunk_size = end - i
            xc_sub = xc[:,i:end,:].reshape(-1, dx) # [cur_chunk_size, dx]
            yc_sub = yc[:,i:end,:].reshape(-1) # [cur_chunk_size]
            self.gp.set_train_data(inputs=xc_sub, targets=yc_sub, strict=False)
            # Performs update steps for hyper
            for _ in range(self.n_steps):
                optimiser.zero_grad()
                loss = -mll(self.gp(xc_sub), yc_sub)
                loss.backward()
                optimiser.step()
        self.likelihood.eval()
        self.gp.eval()

    
    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, 1]", "xt: [m, nt, dx]"
    )
    def forward(self, xc, yc, xt):
        xc, yc, xt = xc.to(self.device), yc.to(self.device), xt.to(self.device)
        m, nc, dx = xc.shape
        _, nt, _ = xt.shape
        # Each batch is given an independent GP
        preds = []
        for i in range(m):
            # Creates new gp
            self.kernel = self.kernel_factory().to(self.device)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.gp = ExactGP(likelihood=self.likelihood, kernel=self.kernel)

            self._stream_through_gp(xc[i:i+1,:,:], yc[i:i+1,:,:]) # Trains GP hypers
            xt_in = xt[i:i+1,:,:].reshape(nt,-1) # [nt, dx]
            with torch.no_grad():
                function_dist = self.gp(xt_in)
                pred_dist = self.likelihood(function_dist)
            mean = pred_dist.mean.view(nt, 1) # [nt, 1]
            cov_mat = pred_dist.covariance_matrix
            preds.append(gpytorch.distributions.MultivariateNormal(mean, cov_mat))
        pred_dist = gpytorch.distributions.MultitaskMultivariateNormal.from_batch(preds) # Combines dist from different batches
        return pred_dist


# RBF GP Stream
class GPStreamRBF(GPStream):
    def __init__(
        self,
        chunk_size: int = 1,
        lr: float = 0.05, # LR for grad updates
        n_steps: int = 10, # Number of grad steps per update
        device: str = "cuda",
    ):
        super().__init__(kernel_factory=(lambda: gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())),
            chunk_size=chunk_size, lr=lr, n_steps=n_steps, device=device)