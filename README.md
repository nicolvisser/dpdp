# Duration-Penalized Dynamic Programming (DPDP)

This repo provides quick access to a DPDP quantization module via `torch.hub`:

This DPDP algorithm is described in [Word Segmentation on Discovered Phone Units with Dynamic Programming and Self-Supervised Scoring](https://arxiv.org/abs/2202.11929).

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
### Notes

The `num_neighbors` argument restricts the quantization such that each feature can only map to the closest codebook entries to that feature.

For the dynamic programming implementation in this repo, using fewer neighbors does not speed up the search. If you want to speed up the quantization process, consider using the weighted finite-state transducer implementation in [DP-WFST](https://github.com/nicolvisser/dp-wfst/blob/main/dpwfst.py).

In the paper, [Spoken Language Modeling with Duration-Penalized Self-Supervised Units](https://arxiv.org/abs/2505.23494), we set `num_neighbors` to 5% of the codebook size and used [DP-WFST](https://github.com/nicolvisser/dp-wfst/blob/main/dpwfst.py).
