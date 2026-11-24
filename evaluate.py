from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mpp.checkpoint import load_model_bundle, resolve_device
from mpp.data import load_records
from mpp.model import collate_graphs


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MPP checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Labeled evaluation dataset.")
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=ROOT / "data" / "smiles_pce.txt",
        help="Dataset used to reconstruct a legacy checkpoint vocabulary.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "mpp_model.pth",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unknown-policy", choices=("error", "zero"), default="error")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive.")
        records = load_records(args.data)
        device = resolve_device(args.device)
        model, featurizer, _ = load_model_bundle(
            args.checkpoint,
            args.reference_data,
            device,
        )
        samples = [
            featurizer.transform(
                record.smiles,
                pce=record.pce,
                unknown_policy=args.unknown_policy,
            )
            for record in records
        ]

        predicted: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(samples), args.batch_size):
                batch = collate_graphs(samples[start : start + args.batch_size], device)
                predicted.extend(model(batch[0], batch[1], batch[2]).cpu().tolist())

        y_true = np.asarray([record.pce for record in records], dtype=np.float64)
        y_pred = np.asarray(predicted, dtype=np.float64)
        residual = y_pred - y_true
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        denominator = float(np.sum(np.square(y_true - y_true.mean())))
        r2 = None if denominator == 0.0 else float(1.0 - np.sum(np.square(residual)) / denominator)
        report = {
            "n": len(records),
            "mae_percentage_points": round(mae, 6),
            "rmse_percentage_points": round(rmse, 6),
            "r2": None if r2 is None else round(r2, 6),
            "unknown_features": int(sum(sample.unknown_features for sample in samples)),
        }
        print(json.dumps(report, indent=2))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
