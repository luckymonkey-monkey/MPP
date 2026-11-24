from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from rdkit import Chem


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    pce: float
    line_number: int


def load_records(
    path: str | Path,
    *,
    deduplicate: bool = True,
    validate_smiles: bool = True,
) -> list[MoleculeRecord]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source}")

    records: list[MoleculeRecord] = []
    seen: dict[str, MoleculeRecord] = {}

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"{source}:{line_number}: expected '<SMILES> <PCE>'."
                )

            smiles, label = parts
            if not smiles:
                raise ValueError(f"{source}:{line_number}: SMILES is empty.")
            if "." in smiles:
                raise ValueError(
                    f"{source}:{line_number}: multi-fragment SMILES is not supported."
                )

            try:
                pce = float(label)
            except ValueError as exc:
                raise ValueError(
                    f"{source}:{line_number}: PCE must be numeric, got {label!r}."
                ) from exc

            if not isfinite(pce):
                raise ValueError(f"{source}:{line_number}: PCE must be finite.")
            if not 0.0 <= pce <= 100.0:
                raise ValueError(
                    f"{source}:{line_number}: PCE must be between 0 and 100."
                )
            if validate_smiles and Chem.MolFromSmiles(smiles) is None:
                raise ValueError(f"{source}:{line_number}: RDKit rejected the SMILES.")

            record = MoleculeRecord(smiles=smiles, pce=pce, line_number=line_number)
            previous = seen.get(smiles)
            if previous is not None:
                if previous.pce != pce:
                    raise ValueError(
                        f"{source}:{line_number}: duplicate SMILES has conflicting PCE "
                        f"values {previous.pce} and {pce}."
                    )
                if deduplicate:
                    continue

            seen[smiles] = record
            records.append(record)

    if not records:
        raise ValueError(f"Dataset is empty: {source}")
    return records
