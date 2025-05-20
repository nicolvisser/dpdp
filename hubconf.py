dependencies = ["torch"]

from typing import Optional

import torch

from dpdp import DPDPQuantizer


def dpdp_quantizer_from_codebook(
    codebook: torch.Tensor, lmbda: float, num_neighbors: Optional[int] = None
):
    return DPDPQuantizer.from_codebook(codebook, lmbda, num_neighbors)


def dpdp_quantizer(
    codebook_size: int,
    codebook_dim: int,
    lmbda: float,
    num_neighbors: Optional[int] = None,
):
    return DPDPQuantizer(codebook_size, codebook_dim, lmbda, num_neighbors)
