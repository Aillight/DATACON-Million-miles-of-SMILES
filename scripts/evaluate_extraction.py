from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SMILES_COLUMNS = ("canonical_smiles", "smiles", "SMILES")
DEFAULT_PROPERTY_COLUMNS = ("property_name", "property", "endpoint")
DEFAULT_VALUE_COLUMNS = ("value", "pMIC", "pmic", "property_value", "activity_value", "label")
MISSING_LABEL = "<missing>"


@dataclass(frozen=True)
class ExtractionRecord:
    smiles: str
    property_name: str
    value: str


@dataclass(frozen=True)
class EvaluationResult:
    gold_records: int
    generated_records: int
    gold_keys: int
    generated_keys: int
    shared_keys: int
    missing_keys: int
    extra_keys: int
    exact_value_matches: int
    value_accuracy_on_shared_keys: float
    key_precision: float
    key_recall: float
    key_f1: float
    macro_f1: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated ChemX extraction CSV against gold CSV.")
    parser.add_argument("generated_csv", type=Path)
    parser.add_argument("gold_csv", type=Path)
    parser.add_argument("--smiles-column")
    parser.add_argument("--property-column")
    parser.add_argument("--value-column")
    parser.add_argument("--value-decimals", type=int, default=3)
    parser.add_argument("--no-canonicalize", action="store_true")
    parser.add_argument("--mismatches-output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    gold_records = load_records(
        args.gold_csv,
        smiles_column=args.smiles_column,
        property_column=args.property_column,
        value_column=args.value_column,
        value_decimals=args.value_decimals,
        canonicalize=not args.no_canonicalize,
    )
    generated_records = load_records(
        args.generated_csv,
        smiles_column=args.smiles_column,
        property_column=args.property_column,
        value_column=args.value_column,
        value_decimals=args.value_decimals,
        canonicalize=not args.no_canonicalize,
    )

    gold_index = build_index(gold_records)
    generated_index = build_index(generated_records)
    result = evaluate_indexes(gold_index, generated_index, len(gold_records), len(generated_records))

    if args.mismatches_output:
        write_mismatches(args.mismatches_output, gold_index, generated_index)

    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print_result(result)


def load_records(
    path: Path,
    smiles_column: str | None = None,
    property_column: str | None = None,
    value_column: str | None = None,
    value_decimals: int = 3,
    canonicalize: bool = True,
) -> list[ExtractionRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        fields = reader.fieldnames
        smiles_field = smiles_column or pick_column(fields, DEFAULT_SMILES_COLUMNS, "SMILES")
        property_field = property_column or pick_column(fields, DEFAULT_PROPERTY_COLUMNS, "property name")
        value_field = value_column or pick_column(fields, DEFAULT_VALUE_COLUMNS, "value")

        records: list[ExtractionRecord] = []
        for row_number, row in enumerate(reader, start=2):
            raw_smiles = normalize_text(row.get(smiles_field, ""))
            raw_value = normalize_value(row.get(value_field, ""), value_decimals)
            if not raw_smiles or not raw_value:
                continue

            smiles = canonicalize_smiles(raw_smiles) if canonicalize else raw_smiles
            if not smiles:
                raise ValueError(f"Invalid SMILES in {path} at row {row_number}: {raw_smiles}")

            property_name = normalize_property(row.get(property_field, "") or "pMIC")
            records.append(ExtractionRecord(smiles=smiles, property_name=property_name, value=raw_value))

    return records


def build_index(records: Iterable[ExtractionRecord]) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for record in records:
        index[(record.smiles, record.property_name)] = record.value
    return index


def evaluate_indexes(
    gold_index: dict[tuple[str, str], str],
    generated_index: dict[tuple[str, str], str],
    gold_record_count: int,
    generated_record_count: int,
) -> EvaluationResult:
    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for macro-F1 evaluation.") from exc

    gold_keys = set(gold_index)
    generated_keys = set(generated_index)
    shared_keys = gold_keys & generated_keys
    all_keys = sorted(gold_keys | generated_keys)

    y_true = [gold_index.get(key, MISSING_LABEL) for key in all_keys]
    y_pred = [generated_index.get(key, MISSING_LABEL) for key in all_keys]
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    exact_matches = sum(1 for key in shared_keys if gold_index[key] == generated_index[key])
    key_precision = safe_divide(len(shared_keys), len(generated_keys))
    key_recall = safe_divide(len(shared_keys), len(gold_keys))
    key_f1 = safe_divide(2 * key_precision * key_recall, key_precision + key_recall)

    return EvaluationResult(
        gold_records=gold_record_count,
        generated_records=generated_record_count,
        gold_keys=len(gold_keys),
        generated_keys=len(generated_keys),
        shared_keys=len(shared_keys),
        missing_keys=len(gold_keys - generated_keys),
        extra_keys=len(generated_keys - gold_keys),
        exact_value_matches=exact_matches,
        value_accuracy_on_shared_keys=safe_divide(exact_matches, len(shared_keys)),
        key_precision=key_precision,
        key_recall=key_recall,
        key_f1=key_f1,
        macro_f1=macro_f1,
    )


def write_mismatches(
    path: Path,
    gold_index: dict[tuple[str, str], str],
    generated_index: dict[tuple[str, str], str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(gold_index) | set(generated_index))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["canonical_smiles", "property_name", "gold_value", "generated_value", "status"],
        )
        writer.writeheader()
        for smiles, property_name in keys:
            key = (smiles, property_name)
            gold_value = gold_index.get(key, "")
            generated_value = generated_index.get(key, "")
            if not generated_value:
                status = "missing"
            elif not gold_value:
                status = "extra"
            elif gold_value == generated_value:
                status = "match"
            else:
                status = "mismatch"
            writer.writerow(
                {
                    "canonical_smiles": smiles,
                    "property_name": property_name,
                    "gold_value": gold_value,
                    "generated_value": generated_value,
                    "status": status,
                }
            )


def print_result(result: EvaluationResult) -> None:
    print(f"gold_records={result.gold_records}")
    print(f"generated_records={result.generated_records}")
    print(f"gold_keys={result.gold_keys}")
    print(f"generated_keys={result.generated_keys}")
    print(f"shared_keys={result.shared_keys}")
    print(f"missing_keys={result.missing_keys}")
    print(f"extra_keys={result.extra_keys}")
    print(f"exact_value_matches={result.exact_value_matches}")
    print(f"value_accuracy_on_shared_keys={result.value_accuracy_on_shared_keys:.6f}")
    print(f"key_precision={result.key_precision:.6f}")
    print(f"key_recall={result.key_recall:.6f}")
    print(f"key_f1={result.key_f1:.6f}")
    print(f"macro_f1={result.macro_f1:.6f}")


def pick_column(fields: list[str], candidates: Iterable[str], label: str) -> str:
    normalized = {normalize_column_name(field): field for field in fields}
    for candidate in candidates:
        field = normalized.get(normalize_column_name(candidate))
        if field:
            return field
    raise ValueError(f"Could not find {label} column. Available columns: {', '.join(fields)}")


def canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for canonical SMILES evaluation.") from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True)


def normalize_column_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def normalize_property(value: str) -> str:
    text = normalize_text(value)
    return text or "pMIC"


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def normalize_value(value: object, decimals: int) -> str:
    text = normalize_text(value).replace(",", ".")
    if not text:
        return ""
    try:
        return f"{float(text):.{decimals}f}"
    except ValueError:
        return text


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
