from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DOMAIN = "Oxazolidinones"
DEFAULT_OUTPUT_DIR = Path("data/test_set")

SMILES_COLUMNS = (
    "canonical_smiles",
    "can_smiles",
    "smiles",
    "compound_smiles",
    "molecule_smiles",
)
PROPERTY_COLUMNS = ("property_name", "property", "target_type", "endpoint", "target", "assay")
VALUE_COLUMNS = ("pmic", "pMIC", "target_value", "value", "property_value", "activity_value", "label", "y")
UNIT_COLUMNS = ("unit", "units", "target_units", "property_unit", "activity_unit")
ID_COLUMNS = ("id", "example_id", "compound_id", "molecule_id", "article_id", "paper_id")
DOMAIN_COLUMNS = ("domain", "subset", "benchmark", "task", "collection", "family")
ARTICLE_HINTS = ("pdf", "article", "paper", "document", "full_text", "text")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize the benchmark test set.")
    parser.add_argument("--dataset")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-articles", action="store_true")
    args = parser.parse_args()

    rows, manifest = load_rows(
        dataset_name=args.dataset or f"ai-chem/{args.domain}",
        domain=args.domain,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    article_dir = output_dir / "articles"
    if not args.skip_articles:
        article_dir.mkdir(parents=True, exist_ok=True)

    raw_examples_path = output_dir / "raw_examples.jsonl"
    write_jsonl(raw_examples_path, rows)

    article_manifest: list[dict[str, Any]] = []
    if not args.skip_articles:
        for index, row in enumerate(rows):
            article_manifest.extend(save_article_artifacts(row, article_dir, index))

    article_references_path = output_dir / "article_references.csv"
    write_article_references(article_references_path, rows)

    gold_records = []
    for index, row in enumerate(rows):
        gold_records.extend(extract_gold_records(row, source_id=f"row-{index:05d}"))

    gold_path = output_dir / "gold.csv"
    write_gold_csv(gold_path, gold_records)

    manifest.update(
        {
            "output_dir": str(output_dir),
            "raw_examples": str(raw_examples_path),
            "article_references": str(article_references_path),
            "gold_csv": str(gold_path),
            "gold_record_count": len(gold_records),
            "article_artifacts": article_manifest,
        }
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows={len(rows)}")
    print(f"gold_records={len(gold_records)}")
    print(f"raw_examples={raw_examples_path}")
    print(f"gold_csv={gold_path}")
    print(f"manifest={manifest_path}")


def load_rows(
    dataset_name: str,
    domain: str,
    split: str,
    trust_remote_code: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset

    manifest: dict[str, Any] = {
        "dataset": dataset_name,
        "domain": domain,
        "requested_split": split,
        "config": None,
        "loaded_splits": [],
        "domain_filter_applied": False,
    }

    config_names: list[str] = []
    try:
        config_names = get_dataset_config_names(dataset_name, trust_remote_code=trust_remote_code)
    except Exception as exc:
        manifest["config_lookup_warning"] = str(exc)

    config_name = find_matching_config(config_names, domain)
    manifest["available_configs"] = config_names
    manifest["config"] = config_name

    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if config_name:
        load_kwargs["name"] = config_name

    try:
        loaded = load_dataset(dataset_name, split=split, **load_kwargs)
    except Exception as exc:
        manifest["direct_split_warning"] = str(exc)
        loaded = load_dataset(dataset_name, **load_kwargs)
    datasets = select_splits(loaded, split)
    manifest["loaded_splits"] = [name for name, _ in datasets]

    rows: list[dict[str, Any]] = []
    for split_name, dataset in datasets:
        for row in dataset:
            row_dict = dict(row)
            row_dict["_benchmark_split"] = split_name
            rows.append(row_dict)

    if not config_name:
        filtered_rows = [row for row in rows if row_matches_domain(row, domain)]
        if filtered_rows:
            rows = filtered_rows
            manifest["domain_filter_applied"] = True
        else:
            manifest["domain_filter_warning"] = (
                "No domain column matched; kept all loaded rows. Check manifest schema before evaluation."
            )

    manifest["row_count"] = len(rows)
    manifest["columns"] = sorted({key for row in rows for key in row.keys()})
    return rows, manifest


def select_splits(loaded: Any, split: str) -> list[tuple[str, Any]]:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError("datasets is required to load the benchmark dataset.") from exc

    if isinstance(loaded, Dataset):
        return [(split or "data", loaded)]
    if isinstance(loaded, DatasetDict):
        if split in loaded:
            return [(split, loaded[split])]
        if "validation" in loaded and split == "test":
            return [("validation", loaded["validation"])]
        return [(name, dataset) for name, dataset in loaded.items()]
    raise TypeError(f"Unsupported datasets object: {type(loaded)!r}")


def find_matching_config(config_names: Iterable[str], domain: str) -> str | None:
    normalized_domain = normalize_token(domain)
    for config in config_names:
        if normalize_token(config) == normalized_domain:
            return config
    for config in config_names:
        if normalized_domain in normalize_token(config):
            return config
    return None


def row_matches_domain(row: dict[str, Any], domain: str) -> bool:
    normalized_domain = normalize_token(domain)
    normalized_domain_columns = {normalize_token(column) for column in DOMAIN_COLUMNS}
    for key, value in row.items():
        if normalize_token(key) not in normalized_domain_columns:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if normalized_domain in normalize_token(str(item)):
                return True
    return False


def extract_gold_records(row: dict[str, Any], source_id: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    maybe_add_gold_record(row, source_id, records)

    for key, value in row.items():
        if not isinstance(value, list):
            continue
        if not any(hint in key.lower() for hint in ("gold", "target", "label", "table", "record")):
            continue
        for item_index, item in enumerate(value):
            if isinstance(item, dict):
                maybe_add_gold_record(item, f"{source_id}:{key}:{item_index}", records)

    return records


def maybe_add_gold_record(source: dict[str, Any], source_id: str, records: list[dict[str, str]]) -> None:
    smiles_key = find_column(source, SMILES_COLUMNS)
    value_key = find_column(source, VALUE_COLUMNS)
    if not smiles_key or not value_key:
        return

    property_key = find_column(source, PROPERTY_COLUMNS)
    unit_key = find_column(source, UNIT_COLUMNS)
    id_key = find_column(source, ID_COLUMNS)

    smiles = stringify(source.get(smiles_key))
    if not smiles:
        return

    property_name = stringify(source.get(property_key)) if property_key else "pMIC"
    if not property_name and value_key.lower() == "pmic":
        property_name = "pMIC"

    value = stringify(source.get(value_key))
    if not value:
        return
    unit = stringify(source.get(unit_key)) if unit_key else ""
    normalized_property, normalized_value, normalized_unit = normalize_benchmark_value(
        property_name=property_name,
        value=value,
        unit=unit,
        smiles=smiles,
    )

    records.append(
        {
            "source_id": source_id,
            "compound_id": stringify(source.get(id_key)) if id_key else "",
            "smiles": smiles,
            "canonical_smiles": canonicalize_smiles(smiles),
            "property_name": normalized_property,
            "value": normalized_value,
            "unit": normalized_unit,
        }
    )


def save_article_artifacts(row: dict[str, Any], article_dir: Path, row_index: int) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    row_id = safe_filename(first_present_value(row, ID_COLUMNS) or f"row-{row_index:05d}")

    for key, value in row.items():
        lower_key = key.lower()
        if not any(hint in lower_key for hint in ARTICLE_HINTS):
            continue

        target_base = article_dir / f"{row_id}_{safe_filename(key)}"
        artifact = save_single_artifact(value, target_base)
        if artifact:
            artifact["field"] = key
            artifact["source_id"] = row_id
            saved.append(artifact)

    return saved


def save_single_artifact(value: Any, target_base: Path) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            return write_bytes(value["bytes"], target_base)
        path_value = value.get("path")
        if path_value:
            copied = copy_path_artifact(Path(path_value), target_base)
            if copied:
                return copied
        return None

    if isinstance(value, bytes):
        return write_bytes(value, target_base)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith(("http://", "https://")):
            suffix = ".pdf" if ".pdf" in stripped.lower() else ".download"
            target = target_base.with_suffix(suffix)
            urllib.request.urlretrieve(stripped, target)
            return {"path": str(target), "kind": "url", "url": stripped}
        path_artifact = copy_path_artifact(Path(stripped), target_base)
        if path_artifact:
            return path_artifact
        if len(stripped) > 500:
            target = target_base.with_suffix(".txt")
            target.write_text(stripped, encoding="utf-8")
            return {"path": str(target), "kind": "text"}

    return None


def copy_path_artifact(path: Path, target_base: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix or ".bin"
    target = target_base.with_suffix(suffix)
    shutil.copyfile(path, target)
    return {"path": str(target), "kind": "file", "source_path": str(path)}


def write_bytes(payload: bytes, target_base: Path) -> dict[str, Any]:
    suffix = ".pdf" if payload.startswith(b"%PDF") else ".bin"
    target = target_base.with_suffix(suffix)
    target.write_bytes(payload)
    return {
        "path": str(target),
        "kind": "bytes",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_gold_csv(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = ["source_id", "compound_id", "smiles", "canonical_smiles", "property_name", "value", "unit"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_article_references(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["pdf", "doi", "title", "publisher", "year", "access"]
    seen: set[tuple[str, str, str]] = set()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {field: stringify(row.get(field)) for field in fieldnames}
            key = (record["pdf"], record["doi"], record["title"])
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(record)


def normalize_benchmark_value(property_name: str, value: str, unit: str, smiles: str) -> tuple[str, str, str]:
    normalized_property = property_name or "pMIC"
    numeric_value = parse_number(value)
    if numeric_value is None:
        return normalized_property, value, unit

    if normalized_property.lower() == "pmic":
        return "pMIC", format_number(numeric_value), "pMIC"

    if normalized_property.lower() == "mic":
        molar_value = mic_to_molar(numeric_value, unit, smiles)
        if molar_value and molar_value > 0:
            return "pMIC", format_number(-math.log10(molar_value)), "pMIC"

    return normalized_property, format_number(numeric_value), unit


def mic_to_molar(value: float, unit: str, smiles: str) -> float | None:
    normalized_unit = unit.replace("μ", "u").replace("µ", "u").lower().strip()
    if normalized_unit in {"mol/l", "m", "m/l"}:
        return value
    if normalized_unit in {"mmol/l", "mm", "mm/l", "mm/liter", "mm/litre", "mm/l"} or normalized_unit == "mm/l":
        return value * 1e-3
    if normalized_unit in {"umol/l", "um", "um/l", "um/liter", "um/litre"}:
        return value * 1e-6
    if normalized_unit in {"ug/ml", "ug/mL".lower(), "mcg/ml"}:
        molecular_weight = molecular_weight_from_smiles(smiles)
        if not molecular_weight:
            return None
        grams_per_liter = value * 1e-3
        return grams_per_liter / molecular_weight
    return None


def molecular_weight_from_smiles(smiles: str) -> float | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return float(Descriptors.MolWt(molecule))


def canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError:
        return smiles

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    return Chem.MolToSmiles(molecule, canonical=True)


def find_column(source: dict[str, Any], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_token(key): key for key in source.keys()}
    for candidate in candidates:
        key = normalized.get(normalize_token(candidate))
        if key:
            return key
    return None


def first_present_value(source: dict[str, Any], candidates: Iterable[str]) -> str | None:
    key = find_column(source, candidates)
    if not key:
        return None
    value = stringify(source.get(key))
    return value or None


def normalize_token(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).strip()


def parse_number(value: str) -> float | None:
    text = value.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: float) -> str:
    return f"{value:.3f}"


def safe_filename(value: Any) -> str:
    text = stringify(value) or "artifact"
    cleaned = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in text)
    return cleaned[:80].strip("_") or "artifact"


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return str(value)


if __name__ == "__main__":
    main()
