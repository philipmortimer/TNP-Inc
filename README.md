# Transformer Neural Processes
A framework for implementing transformer neural processes (TNPs) in Python.

## Setting up the conda environment.
```
conda env create -f environment.yml
conda activate tnpp
pip install -e .
```

## Running experiments.
```
python experiments/lightning_train.py --config experiments/configs/models/tnp.yml --generator_config experiments/configs/generators/synthetic-1d.yml
```

## Constructing models.
How to construct a TNP from scratch:
```
import tnp

# First define the parameters.
dim_x = 1
dim_y = 1
embed_dim = 128
num_heads = 8
head_dim = 16
num_layers = 5

# Construct the encoder.
mhca_layer = tnp.networks.attention_layers.MultiHeadCrossAttentionLayer(
    embed_dim=embed_dim,
    num_heads=num_heads,
    head_dim=head_dim,
    feedforward_dim=embed_dim,
)
transformer_encoder = tnp.networks.transformer.TNPTransformerEncoder(
    mhca_layer=mhca_layer,
    num_layers=num_layers,
)
xy_encoder = tnp.networks.mlp.MLP(
    in_dim=1 + dim_x + dim_y,
    out_dim=embed_dim,
    num_layers=2,
    width=embed_dim,
)
tnp_encoder = tnp.models.tnp.TNPEncoder(
    transformer_encoder = transformer_encoder,
    xy_encoder=xy_encoder,
)

# Construct the decoder.
z_decoder = tnp.networks.mlp.MLP(
    in_dim=embed_dim,
    out_dim=2 * dim_y,
    num_layers=num_layers,
    width=embed_dim,
)
tnp_decoder = tnp.models.tnp.TNPDecoder(
    z_decoder=z_decoder,
)

# Construct the likelihood.
likelihood = tnp.likelihoods.gaussian.HeteroscedasticNormalLikelihood()

# Voila.
tnp = tnp.models.tnp.TNP(
    encoder=tnp_encoder,
    decoder=tnp_decoder,
    likelihood=likelihood,
)
```

## Loading data.
Data is loaded in batches, which follow the construction of `tnp.data.base.Batch`:
```
class Batch:
    # All observations of shape (m, n, d).
    x: torch.Tensor
    y: torch.Tensor

    # Context data of shape (m, nc, d).
    xc: torch.Tensor
    yc: torch.Tensor

    # Target data of shape (m, nt, d).
    xt: torch.Tensor
    yt: torch.Tensor
```
See `tnp.data` for more examples.
