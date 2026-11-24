from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .features import GraphSample


@dataclass(frozen=True)
class MPPConfig:
    n_fingerprints: int
    embedding_dim: int = 50
    hidden_layers: int = 6
    output_layers: int = 6
    radius: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 1 and name != "radius":
                raise ValueError(f"{name} must be positive.")
        if self.radius < 0:
            raise ValueError("radius must be non-negative.")


class MPPRegressor(nn.Module):
    def __init__(self, config: MPPConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        self.embed_fingerprint = nn.Embedding(config.n_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList(
            nn.Linear(dim, dim) for _ in range(config.hidden_layers)
        )
        self.W_output = nn.ModuleList(
            nn.Linear(dim, dim) for _ in range(config.output_layers)
        )
        self.W_property = nn.Linear(dim, 1)

    def forward(
        self,
        fingerprints: torch.Tensor,
        adjacency: torch.Tensor,
        molecule_sizes: list[int],
    ) -> torch.Tensor:
        if fingerprints.ndim != 1:
            raise ValueError("fingerprints must be a one-dimensional tensor.")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix.")
        if adjacency.shape[0] != fingerprints.shape[0]:
            raise ValueError("adjacency and fingerprints must have matching sizes.")
        if not molecule_sizes or any(size < 1 for size in molecule_sizes):
            raise ValueError("molecule_sizes must contain positive integers.")
        if sum(molecule_sizes) != fingerprints.shape[0]:
            raise ValueError("molecule_sizes must sum to the number of fingerprints.")

        vectors = self.embed_fingerprint(fingerprints)
        for layer in self.W_fingerprint:
            hidden = torch.relu(layer(vectors))
            vectors = F.normalize(hidden + adjacency @ hidden, p=2, dim=1)

        molecules = torch.stack(
            [part.sum(dim=0) for part in torch.split(vectors, molecule_sizes)]
        )
        for layer in self.W_output:
            molecules = torch.relu(layer(molecules))
        return self.W_property(molecules).squeeze(-1)


def collate_graphs(
    samples: list[GraphSample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor | None]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    fingerprints = torch.cat([sample.fingerprints for sample in samples]).to(device)
    adjacency = torch.block_diag(*[sample.adjacency for sample in samples]).to(device)
    sizes = [sample.n_atoms for sample in samples]

    labels: torch.Tensor | None = None
    if any(sample.pce is not None for sample in samples):
        if not all(sample.pce is not None for sample in samples):
            raise ValueError("A batch cannot mix labeled and unlabeled samples.")
        labels = torch.tensor(
            [float(sample.pce) for sample in samples],
            dtype=torch.float32,
            device=device,
        )
    return fingerprints, adjacency, sizes, labels
