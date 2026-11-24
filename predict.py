from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from mpp.checkpoint import load_model_bundle, resolve_device
from mpp.features import UnknownFeatureError
from mpp.model import collate_graphs


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "mpp_model.pth"
DEFAULT_DATA = ROOT / "data" / "smiles_pce.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict photothermal conversion efficiency.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", help="One SMILES string.")
    source.add_argument("--input", type=Path, help="Text file containing one SMILES per line.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument(
        "--unknown-policy",
        choices=("error", "zero"),
        default="error",
        help="How to handle fingerprints absent from the checkpoint vocabulary.",
    )
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> list[str]:
    if args.smiles is not None:
        return [args.smiles.strip()]
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    values = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value and not value.startswith("#")]
    if not values:
        raise ValueError(f"Input file contains no SMILES: {args.input}")
    return values


def main() -> int:
    args = parse_args()
    try:
        smiles_values = load_inputs(args)
        device = resolve_device(args.device)
        model, featurizer, _ = load_model_bundle(
            args.checkpoint,
            args.reference_data,
            device,
        )
        samples = [
            featurizer.transform(smiles, unknown_policy=args.unknown_policy)
            for smiles in smiles_values
        ]
        batch = collate_graphs(samples, device)
        with torch.inference_mode():
            predictions = model(batch[0], batch[1], batch[2]).cpu().tolist()

        rows = [
            {
                "smiles": smiles,
                "predicted_pce_percent": round(float(prediction), 6),
                "unknown_features": sample.unknown_features,
            }
            for smiles, prediction, sample in zip(smiles_values, predictions, samples)
        ]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(rows[0] if len(rows) == 1 else rows, indent=2))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, UnknownFeatureError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
