from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from mpp.data import load_records


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate the five regression baselines.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "smiles_pce.txt")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Use the original 10,000-tree settings.")
    return parser.parse_args()


def ascii_matrix(smiles_values: list[str]) -> np.ndarray:
    if not smiles_values:
        raise ValueError("At least one SMILES is required.")
    width = max(map(len, smiles_values))
    matrix = np.full((len(smiles_values), width), 30, dtype=np.float64)
    for row, smiles in enumerate(smiles_values):
        encoded = np.fromiter((ord(character) for character in smiles), dtype=np.float64)
        matrix[row, : encoded.size] = encoded
    return matrix


def main() -> int:
    args = parse_args()
    try:
        records = load_records(args.data)
        if not 2 <= args.folds <= len(records):
            raise ValueError("folds must be between 2 and the number of records.")
        x = ascii_matrix([record.smiles for record in records])
        y = np.asarray([record.pce for record in records], dtype=np.float64)
        estimators = 10_000 if args.full else 500
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=estimators,
                random_state=args.seed,
                n_jobs=-1,
            ),
            "support_vector": SVR(),
            "decision_tree": DecisionTreeRegressor(max_depth=2_000, random_state=args.seed),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=estimators,
                learning_rate=0.1,
                random_state=args.seed,
            ),
            "knn": KNeighborsRegressor(n_neighbors=min(50, len(records) - 1), weights="distance"),
        }
        folds = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        report: dict[str, dict[str, float]] = {}
        for name, model in models.items():
            prediction = cross_val_predict(model, x, y, cv=folds, n_jobs=1)
            report[name] = {
                "mae_percentage_points": round(float(mean_absolute_error(y, prediction)), 6),
                "r2": round(float(r2_score(y, prediction)), 6),
            }
        print(json.dumps(report, indent=2))
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
