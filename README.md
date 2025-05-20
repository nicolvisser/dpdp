# Duration-Penalized Dynamic Programming (DPDP)

This repo provides quick access to a DPDP quantization module via `torch.hub`:

This DPDP algorithm is described in [Word Segmentation on Discovered Phone Units with Dynamic Programming and Self-Supervised Scoring](https://arxiv.org/abs/2202.11929) and used in `Spoken Language Modeling with Duration-Penalized Self-Supervised Units` (arxiv link will be added soon).

```py
codebook = torch.load("your_codebook.pt") # (K, D)
features = torch.load("your_features.pt") # (T, D)

quantizer = torch.hub.load(
    "nicolvisser/dpdp",
    "dpdp_quantizer_from_codebook",
    codebook=codebook,
    lmbda=0, # <- control coarseness here
    num_neighbors=None,
    trust_repo=True,
    force_reload=True
)

quantized_features, indices = quantizer(features)
```
