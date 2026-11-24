from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mpp.checkpoint import resolve_device, save_model_bundle
from mpp.data import load_records
from mpp.features import GraphFeaturizer, GraphSample
from mpp.model import MPPConfig, MPPRegressor, collate_graphs


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MPP graph regressor.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "smiles_pce.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "trained_mpp.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--output-layers", type=int, default=6)
    parser.add_argument("--radius", type=int, default=1)
    return parser.parse_args()


def batches(samples: list[GraphSample], size: int, rng: random.Random):
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    for start in range(0, len(indices), size):
        yield [samples[index] for index in indices[start : start + size]]


def validation_loss(
    model: MPPRegressor,
    samples: list[GraphSample],
    batch_size: int,
    device: torch.device,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = collate_graphs(samples[start : start + batch_size], device)
            prediction = model(batch[0], batch[1], batch[2])
            losses.append(float(F.mse_loss(prediction, batch[3]).item()))
    return float(np.mean(losses))


def main() -> int:
    args = parse_args()
    try:
        if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
            raise ValueError("epochs, batch-size, and patience must be positive.")
        if not 0.0 < args.learning_rate < 1.0:
            raise ValueError("learning-rate must be between 0 and 1.")
        if not 0.0 < args.validation_fraction < 0.5:
            raise ValueError("validation-fraction must be between 0 and 0.5.")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = resolve_device(args.device)

        records = load_records(args.data)
        if len(records) < 5:
            raise ValueError("Training requires at least five unique molecules.")

        featurizer = GraphFeaturizer(radius=args.radius)
        featurizer.fit([record.smiles for record in records])
        samples = [
            featurizer.transform(record.smiles, pce=record.pce)
            for record in records
        ]

        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(samples), generator=generator).tolist()
        validation_size = max(1, round(len(samples) * args.validation_fraction))
        validation = [samples[index] for index in order[:validation_size]]
        training = [samples[index] for index in order[validation_size:]]

        config = MPPConfig(
            n_fingerprints=len(featurizer.vocabulary.fingerprints),
            embedding_dim=args.embedding_dim,
            hidden_layers=args.hidden_layers,
            output_layers=args.output_layers,
            radius=args.radius,
        )
        model = MPPRegressor(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        rng = random.Random(args.seed)

        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        final_epoch = 0

        for epoch in range(1, args.epochs + 1):
            final_epoch = epoch
            model.train()
            train_losses: list[float] = []
            for group in batches(training, args.batch_size, rng):
                batch = collate_graphs(group, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch[0], batch[1], batch[2])
                loss = F.mse_loss(prediction, batch[3])
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced a non-finite loss.")
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            current_loss = validation_loss(model, validation, args.batch_size, device)
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch == 1 or epoch % 25 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_mse": round(float(np.mean(train_losses)), 6),
                            "validation_mse": round(current_loss, 6),
                        }
                    )
                )
            if stale_epochs >= args.patience:
                break

        if best_state is None:
            raise RuntimeError("No valid checkpoint was produced.")
        model.load_state_dict(best_state)
        save_model_bundle(
            args.output,
            model,
            featurizer,
            metadata={
                "training_rows": len(training),
                "validation_rows": len(validation),
                "seed": args.seed,
                "best_validation_mse": best_loss,
                "epochs_completed": final_epoch,
            },
        )
        print(f"Saved checkpoint: {args.output}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
