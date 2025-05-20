from typing import Optional

import torch


class DPDPQuantizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        lmbda: float,
        num_neighbors: Optional[int] = None,
    ):
        super().__init__()
        self.codebook = torch.nn.Parameter(torch.randn(codebook_size, codebook_dim))
        self.lmbda = lmbda
        self.num_neighbors = num_neighbors

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        indices = dpdp(features, self.codebook, self.lmbda, self.num_neighbors)
        quantized_features = self.codebook[indices]
        quantized_features_st = features - (features - quantized_features).detach()
        return quantized_features_st, indices

    @classmethod
    def from_codebook(
        cls, codebook: torch.Tensor, lmbda: float, num_neighbors: Optional[int] = None
    ):
        quantizer = cls(codebook.shape[0], codebook.shape[1], lmbda, num_neighbors)
        quantizer.codebook.data.copy_(codebook)
        return quantizer


@torch.jit.script
def dpdp(
    features: torch.Tensor,
    codebook: torch.Tensor,
    lmbda: float,
    num_neighbors: Optional[int] = None,
):
    """
    Duration penalized dynamic programming for vector quantization as described in https://arxiv.org/abs/2202.11929

    Args:
        features: (T, D) tensor of features
        codebook: (K, D) tensor of codebook vectors
        lmbda: lambda parameter for duration penalty
        num_neighbors: number of neighbors in the codebook to consider for each feature vector. If None, all neighbors are considered.

    Note that in this implementation, there is no computational speedup for num_neighbors < K.
    If you want such a speedup, consider using the DP-WFST implementation.

    Returns:
        (T,) tensor of quantized feature vectors
    """
    if features.dim() != 2:
        raise NotImplementedError("Only works for 2D input")
    assert features.device == codebook.device

    T = features.shape[0]
    K = codebook.shape[0]

    distances = torch.cdist(x1=features, x2=codebook, p=2.0) ** 2

    if num_neighbors is not None:
        # Set the distances of furthest neighbors to infinity
        topk_indices = distances.topk(
            K - num_neighbors, dim=1, largest=True, sorted=True
        ).indices
        distances[torch.arange(distances.shape[0])[:, None], topk_indices] = float(
            "inf"
        )

    durations = torch.arange(1, T + 1, device=features.device, dtype=torch.float32)
    duration_penalties = lmbda * (-durations + 1)
    alphas = torch.zeros(
        T + 1, dtype=torch.float32, device=features.device
    )  # min cost up to time t
    betas = torch.zeros(
        T + 1, dtype=torch.int64, device=features.device
    )  # backpointers
    gammas = torch.zeros(
        T + 1, dtype=torch.int64, device=features.device
    )  # backpointers optimal unit
    units = torch.zeros(
        T, dtype=torch.int64, device=features.device
    )  # store final result here

    for t in range(1, T + 1):
        dist_slice_cumsum = torch.cumsum(distances[0:t].flip(dims=[-2]), dim=-2).flip(
            dims=[-2]
        )
        min_costs_without_dp, min_costs_without_dp_indices = torch.min(
            dist_slice_cumsum, dim=-1
        )
        costs_with_dp = (
            alphas[0:t] + min_costs_without_dp + duration_penalties[0:t].flip(dims=[-1])
        )
        min_cost_with_dp, min_cost_with_dp_indices = torch.min(costs_with_dp, dim=-1)
        alphas[t] = min_cost_with_dp
        betas[t] = min_cost_with_dp_indices
        gammas[t] = min_costs_without_dp_indices[min_cost_with_dp_indices]

    # backtracking
    index = T  # start at last index
    while index > 0:
        fill_down_to_index = betas[index]
        fill_value = gammas[index]
        units[fill_down_to_index:index] = fill_value
        index = fill_down_to_index

    return units
