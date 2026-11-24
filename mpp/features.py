from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from rdkit import Chem


UnknownPolicy = Literal["error", "zero"]


class UnknownFeatureError(ValueError):
    pass


@dataclass
class Vocabulary:
    atoms: dict[Any, int] = field(default_factory=dict)
    bonds: dict[Any, int] = field(default_factory=dict)
    fingerprints: dict[Any, int] = field(default_factory=dict)
    edges: dict[Any, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, dict[Any, int]]:
        return {
            "atoms": self.atoms,
            "bonds": self.bonds,
            "fingerprints": self.fingerprints,
            "edges": self.edges,
        }

    @classmethod
    def from_dict(cls, value: dict[str, dict[Any, int]]) -> "Vocabulary":
        required = {"atoms", "bonds", "fingerprints", "edges"}
        if set(value) != required:
            raise ValueError(f"Vocabulary keys must be {sorted(required)}.")
        return cls(**value)


@dataclass(frozen=True)
class GraphSample:
    fingerprints: torch.Tensor
    adjacency: torch.Tensor
    pce: float | None = None
    unknown_features: int = 0

    @property
    def n_atoms(self) -> int:
        return int(self.fingerprints.shape[0])


class GraphFeaturizer:
    def __init__(
        self,
        radius: int = 1,
        vocabulary: Vocabulary | None = None,
        max_atoms: int = 2048,
    ) -> None:
        if radius < 0:
            raise ValueError("radius must be non-negative.")
        if max_atoms < 1:
            raise ValueError("max_atoms must be positive.")
        self.radius = radius
        self.vocabulary = vocabulary or Vocabulary()
        self.max_atoms = max_atoms

    def fit(self, smiles_values: list[str]) -> "GraphFeaturizer":
        if not smiles_values:
            raise ValueError("At least one SMILES is required to fit the vocabulary.")
        for smiles in smiles_values:
            self._transform(smiles, mutate=True, unknown_policy="error")
        return self

    def transform(
        self,
        smiles: str,
        *,
        pce: float | None = None,
        unknown_policy: UnknownPolicy = "error",
    ) -> GraphSample:
        return self._transform(
            smiles,
            mutate=False,
            unknown_policy=unknown_policy,
            pce=pce,
        )

    def _transform(
        self,
        smiles: str,
        *,
        mutate: bool,
        unknown_policy: UnknownPolicy,
        pce: float | None = None,
    ) -> GraphSample:
        if unknown_policy not in {"error", "zero"}:
            raise ValueError("unknown_policy must be 'error' or 'zero'.")
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("SMILES must be a non-empty string.")
        smiles = smiles.strip()
        if "." in smiles:
            raise ValueError("Multi-fragment SMILES is not supported.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit could not parse the SMILES.")
        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() > self.max_atoms:
            raise ValueError(
                f"Molecule has {mol.GetNumAtoms()} atoms after adding hydrogens; "
                f"the configured limit is {self.max_atoms}."
            )

        unknown_count = 0

        def index(mapping: dict[Any, int], key: Any, kind: str) -> int:
            nonlocal unknown_count
            if key in mapping:
                return mapping[key]
            if mutate:
                mapping[key] = len(mapping)
                return mapping[key]
            if unknown_policy == "zero" and mapping:
                unknown_count += 1
                return 0
            raise UnknownFeatureError(f"Unknown {kind} encountered: {key!r}")

        atoms: list[Any] = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for atom in mol.GetAromaticAtoms():
            atoms[atom.GetIdx()] = (atoms[atom.GetIdx()], "aromatic")
        atom_ids = [index(self.vocabulary.atoms, atom, "atom") for atom in atoms]

        neighbors: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_id = index(self.vocabulary.bonds, str(bond.GetBondType()), "bond")
            neighbors[i].append((j, bond_id))
            neighbors[j].append((i, bond_id))

        if len(atom_ids) == 1 or self.radius == 0:
            nodes = [
                index(self.vocabulary.fingerprints, atom_id, "fingerprint")
                for atom_id in atom_ids
            ]
        else:
            nodes = list(atom_ids)
            current_edges: dict[int, list[tuple[int, int]]] = dict(neighbors)
            for _ in range(self.radius):
                updated_nodes: list[int] = []
                for i in range(len(atom_ids)):
                    local = current_edges.get(i, [])
                    neighborhood = tuple(sorted((nodes[j], edge) for j, edge in local))
                    fingerprint = (nodes[i], neighborhood)
                    updated_nodes.append(
                        index(
                            self.vocabulary.fingerprints,
                            fingerprint,
                            "fingerprint",
                        )
                    )

                updated_edges: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
                for i, local in current_edges.items():
                    for j, edge in local:
                        sides = tuple(sorted((nodes[i], nodes[j])))
                        edge_id = index(
                            self.vocabulary.edges,
                            (sides, edge),
                            "edge",
                        )
                        updated_edges[i].append((j, edge_id))
                nodes = updated_nodes
                current_edges = dict(updated_edges)

        adjacency = Chem.GetAdjacencyMatrix(mol).astype(np.float32, copy=False)
        fingerprints = torch.as_tensor(np.asarray(nodes), dtype=torch.long)
        adjacency_tensor = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency_tensor.shape != (fingerprints.numel(), fingerprints.numel()):
            raise RuntimeError("Fingerprint and adjacency dimensions do not match.")

        return GraphSample(
            fingerprints=fingerprints,
            adjacency=adjacency_tensor,
            pce=pce,
            unknown_features=unknown_count,
        )
