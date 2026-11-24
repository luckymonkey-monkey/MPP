from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .data import load_records
from .features import GraphFeaturizer, Vocabulary
from .model import MPPConfig, MPPRegressor


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def _state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary.")
    if "model_state" in payload:
        state = payload["model_state"]
        metadata = payload
    else:
        state = payload
        metadata = {}
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model state is empty.")
    if not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint parameter names must be strings.")
    return state, metadata


def infer_config(state: dict[str, torch.Tensor], radius: int = 1) -> MPPConfig:
    embedding = state.get("embed_fingerprint.weight")
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise ValueError("Checkpoint has no valid fingerprint embedding.")

    hidden_layers = len(
        {
            key.split(".")[1]
            for key in state
            if key.startswith("W_fingerprint.") and key.endswith(".weight")
        }
    )
    output_layers = len(
        {
            key.split(".")[1]
            for key in state
            if key.startswith("W_output.") and key.endswith(".weight")
        }
    )
    return MPPConfig(
        n_fingerprints=int(embedding.shape[0]),
        embedding_dim=int(embedding.shape[1]),
        hidden_layers=hidden_layers,
        output_layers=output_layers,
        radius=radius,
    )


def load_model_bundle(
    checkpoint_path: str | Path,
    reference_data: str | Path,
    device: torch.device,
) -> tuple[MPPRegressor, GraphFeaturizer, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    state, metadata = _state_dict(payload)

    if "config" in metadata:
        config = MPPConfig(**metadata["config"])
    else:
        config = infer_config(state)

    if "vocabulary" in metadata:
        vocabulary = Vocabulary.from_dict(metadata["vocabulary"])
        featurizer = GraphFeaturizer(radius=config.radius, vocabulary=vocabulary)
    else:
        records = load_records(reference_data)
        featurizer = GraphFeaturizer(radius=config.radius)
        featurizer.fit([record.smiles for record in records])

    actual = len(featurizer.vocabulary.fingerprints)
    if actual != config.n_fingerprints:
        raise ValueError(
            f"Vocabulary has {actual} fingerprints, but the checkpoint expects "
            f"{config.n_fingerprints}. Use the reference dataset paired with the checkpoint."
        )

    model = MPPRegressor(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, featurizer, metadata


def save_model_bundle(
    path: str | Path,
    model: MPPRegressor,
    featurizer: GraphFeaturizer,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "vocabulary": featurizer.vocabulary.as_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, destination)
