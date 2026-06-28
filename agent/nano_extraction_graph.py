from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.extraction_graph import (
    DEFAULT_HUGGINGFACE_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    ExtractionPromptContext,
    append_unique,
    build_openrouter_headers,
    call_openrouter_json_completion,
    extract_first_json_object,
    extract_huggingface_message_text,
    resolve_openrouter_token,
    resolve_huggingface_token,
    safe_exception_message,
    strip_markdown_json_fence,
)


MAX_NANO_EXTRACTION_ATTEMPTS = 3
PERIODIC_TABLE = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}
SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")
FORMULA_CANDIDATE_RE = re.compile(r"\b(?:[A-Z][a-z]?\s*\d*(?:\.\d+)?\s*){2,}\b")
CATALYST_LABEL_RE = re.compile(r"[-/@]|SBA|CNT|CNF|MOF|ZIF|SiO2|Al2O3|TiO2", re.IGNORECASE)
ALLOWED_NANO_PROPERTY_TOKENS = (
    "size",
    "diameter",
    "radius",
    "length",
    "width",
    "hydrodynamic",
    "zeta",
    "km",
    "vmax",
    "kcat",
    "activity",
    "lod",
    "limit",
    "range",
    "ic50",
    "viability",
    "cytotoxicity",
    "ph",
    "temperature",
    "recovery",
    "rsd",
    "surface area",
    "bet",
    "pore",
    "rate",
    "velocity",
    "yield",
    "conversion",
    "selectivity",
)


class ExtractedNanozymeProperty(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    material_id: str = Field(min_length=1)
    material_name: str = Field(min_length=1)
    material_formula: str = Field(min_length=1)
    property_name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    assay: str = ""
    condition: str = ""


class NanozymeExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rows: list[ExtractedNanozymeProperty] = Field(default_factory=list)


class ValidatedNanozymeProperty(ExtractedNanozymeProperty):
    normalized_formula: str = Field(min_length=1)


class NanoExtractionGraphState(TypedDict, total=False):
    chunk_text: str
    extracted_objects: list[dict[str, Any]]
    validated_objects: list[dict[str, Any]]
    rejected_objects: list[dict[str, Any]]
    error_log: list[str]
    warnings: list[str]
    valid: bool
    attempts: int
    max_attempts: int
    status: Literal["pending", "valid", "invalid", "failed"]


NanoExtractorFn = Callable[
    [ExtractionPromptContext],
    NanozymeExtractionBatch | list[ExtractedNanozymeProperty | dict[str, Any]] | dict[str, Any],
]


@dataclass(frozen=True)
class FormulaValidation:
    normalized_formula: str
    errors: list[str]


def build_nanozyme_extraction_prompt(context: ExtractionPromptContext) -> str:
    error_block = "\n".join(f"- {error}" for error in context.error_log) or "- none"
    return (
        "Extract nanozyme and nanoparticle material property data from the article chunk.\n"
        "Return only structured rows matching the schema: material_id, material_name, "
        "material_formula, property_name, value, unit, assay, condition.\n"
        "Extract scalar material or assay properties such as particle diameter, nanoparticle size, "
        "Km, Vmax, limit of detection, linear range endpoints, optimum pH, optimum temperature, "
        "recovery, and RSD.\n"
        "For catalyst and nanocatalyst review articles, also extract process metrics such as product yield, "
        "conversion, and selectivity when they are tied to a material/catalyst.\n"
        "Do not extract ordinary reagents, solvents, vendor names, figure numbers, references, or citations.\n"
        "Use the material formula for the active nanozyme material, for example Mn3O4, Fe3O4, V2O5, Cu1.8S.\n"
        "If a range is reported, return separate rows for lower and upper endpoints with property_name "
        "ending in '_lower' and '_upper'.\n"
        "If previous validation errors are listed, fix impossible formulas or impossible physical values.\n\n"
        f"Attempt: {context.attempt}\n"
        f"Previous validation errors:\n{error_block}\n\n"
        f"Article chunk:\n{context.chunk_text}"
    )


def build_nanozyme_extraction_graph(extractor: NanoExtractorFn):
    graph = StateGraph(NanoExtractionGraphState)
    graph.add_node("extractor", lambda state: nano_extractor_node(state, extractor))
    graph.add_node("critic", nano_critic_node)
    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry_nano,
        {
            "retry": "extractor",
            "done": END,
        },
    )
    return graph.compile()


def run_nanozyme_extraction(
    chunk_text: str,
    extractor: NanoExtractorFn,
    max_attempts: int = MAX_NANO_EXTRACTION_ATTEMPTS,
) -> NanoExtractionGraphState:
    graph = build_nanozyme_extraction_graph(extractor)
    return graph.invoke(
        {
            "chunk_text": chunk_text,
            "extracted_objects": [],
            "validated_objects": [],
            "rejected_objects": [],
            "error_log": [],
            "warnings": [],
            "valid": False,
            "attempts": 0,
            "max_attempts": max_attempts,
            "status": "pending",
        }
    )


def make_huggingface_nanozyme_extractor(
    model: str = DEFAULT_HUGGINGFACE_MODEL,
    client: Any | None = None,
    max_output_tokens: int = 2000,
    token: str | None = None,
) -> NanoExtractorFn:
    resolved_token = resolve_huggingface_token(token)
    if client is None:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=model, token=resolved_token)

    def extractor(context: ExtractionPromptContext) -> dict[str, Any]:
        response = call_huggingface_nanozyme_completion(
            client=client,
            model=model,
            context=context,
            max_output_tokens=max_output_tokens,
            token=resolved_token,
        )
        text = extract_huggingface_message_text(response)
        return parse_nanozyme_batch_json(text)

    return extractor


def make_openrouter_nanozyme_extractor(
    model: str = DEFAULT_OPENROUTER_MODEL,
    client: Any | None = None,
    max_output_tokens: int = 2000,
    token: str | None = None,
) -> NanoExtractorFn:
    resolved_token = resolve_openrouter_token(token)
    if client is None:
        if not resolved_token:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter nanozyme extraction.")
        from openai import OpenAI

        client = OpenAI(
            api_key=resolved_token,
            base_url=DEFAULT_OPENROUTER_BASE_URL,
            default_headers=build_openrouter_headers(),
        )

    def extractor(context: ExtractionPromptContext) -> dict[str, Any]:
        schema = NanozymeExtractionBatch.model_json_schema()
        text = call_openrouter_json_completion(
            client=client,
            model=model,
            prompt=build_huggingface_nanozyme_prompt(context, schema),
            schema=schema,
            schema_name="nanozyme_extraction_batch",
            max_output_tokens=max_output_tokens,
            token=resolved_token,
        )
        return parse_nanozyme_batch_json(text)

    return extractor


def call_huggingface_nanozyme_completion(
    client: Any,
    model: str,
    context: ExtractionPromptContext,
    max_output_tokens: int,
    token: str | None = None,
) -> Any:
    schema = NanozymeExtractionBatch.model_json_schema()
    messages = [
        {
            "role": "system",
            "content": (
                "You extract nanozyme material data. Return one JSON object only. "
                "The JSON must match the provided schema and contain a rows array. "
                "Use an empty rows array when the chunk has no extractable records."
            ),
        },
        {"role": "user", "content": build_huggingface_nanozyme_prompt(context, schema)},
    ]
    kwargs = {
        "messages": messages,
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        return client.chat_completion(**kwargs)
    except Exception as first_exc:
        kwargs.pop("response_format", None)
        try:
            return client.chat_completion(**kwargs)
        except Exception as second_exc:
            raise RuntimeError(
                "Hugging Face nanozyme extraction request failed: "
                f"{type(second_exc).__name__}: {safe_exception_message(second_exc, token)}"
            ) from first_exc


def build_huggingface_nanozyme_prompt(context: ExtractionPromptContext, schema: dict[str, Any]) -> str:
    return (
        f"{build_nanozyme_extraction_prompt(context)}\n\n"
        "Return a single valid JSON object with this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Do not include markdown fences, commentary, citations, or explanatory text."
    )


def nano_extractor_node(state: NanoExtractionGraphState, extractor: NanoExtractorFn) -> NanoExtractionGraphState:
    attempt = int(state.get("attempts", 0)) + 1
    context = ExtractionPromptContext(
        chunk_text=str(state.get("chunk_text", "")),
        error_log=list(state.get("error_log", [])),
        attempt=attempt,
    )
    try:
        batch, rejected_objects = normalize_nano_extractor_output_detailed(extractor(context))
    except (TypeError, ValidationError, ValueError, RuntimeError) as exc:
        return {
            "extracted_objects": [],
            "rejected_objects": list(state.get("rejected_objects", [])),
            "attempts": attempt,
            "valid": False,
            "status": "failed",
            "error_log": append_unique(state.get("error_log", []), f"extractor output validation failed: {exc}"),
        }

    return {
        "extracted_objects": [row.model_dump() for row in batch.rows],
        "rejected_objects": list(state.get("rejected_objects", [])) + rejected_objects,
        "attempts": attempt,
        "valid": False,
        "status": "pending",
    }


def nano_critic_node(state: NanoExtractionGraphState) -> NanoExtractionGraphState:
    if state.get("status") == "failed":
        return {
            "validated_objects": [],
            "error_log": list(state.get("error_log", [])),
            "warnings": list(state.get("warnings", [])),
            "valid": False,
            "status": "failed",
        }

    rows = [ExtractedNanozymeProperty.model_validate(row) for row in state.get("extracted_objects", [])]
    validated_rows, errors, warnings, rejected_rows = validate_and_normalize_nanozyme_rows_detailed(rows)
    valid = not errors
    return {
        "validated_objects": [row.model_dump() for row in validated_rows],
        "rejected_objects": list(state.get("rejected_objects", [])) + rejected_rows,
        "error_log": append_unique(state.get("error_log", []), *errors),
        "warnings": append_unique(state.get("warnings", []), *warnings),
        "valid": valid,
        "status": "valid" if valid else "invalid",
    }


def should_retry_nano(state: NanoExtractionGraphState) -> Literal["retry", "done"]:
    if state.get("valid", False):
        return "done"
    if int(state.get("attempts", 0)) >= int(state.get("max_attempts", MAX_NANO_EXTRACTION_ATTEMPTS)):
        return "done"
    return "retry"


def normalize_nano_extractor_output(
    output: NanozymeExtractionBatch | list[ExtractedNanozymeProperty | dict[str, Any]] | dict[str, Any],
) -> NanozymeExtractionBatch:
    batch, _rejected = normalize_nano_extractor_output_detailed(output)
    return batch


def normalize_nano_extractor_output_detailed(
    output: NanozymeExtractionBatch | list[ExtractedNanozymeProperty | dict[str, Any]] | dict[str, Any],
) -> tuple[NanozymeExtractionBatch, list[dict[str, Any]]]:
    if isinstance(output, NanozymeExtractionBatch):
        return output, []
    if isinstance(output, list):
        return build_nano_batch_from_rows(output)
    if isinstance(output, dict):
        payload = normalize_nano_payload(output)
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("Nanozyme extractor JSON payload must contain a rows array.")
        return build_nano_batch_from_rows(rows)
    raise TypeError(f"Unsupported nano extractor output type: {type(output)!r}")


def build_nano_batch_from_rows(
    rows: list[ExtractedNanozymeProperty | dict[str, Any]],
) -> tuple[NanozymeExtractionBatch, list[dict[str, Any]]]:
    valid_rows: list[ExtractedNanozymeProperty] = []
    rejected_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        try:
            valid_rows.append(ExtractedNanozymeProperty.model_validate(row))
        except ValidationError as exc:
            rejected_rows.append(build_rejected_nano_payload(row, index=index, reason=format_validation_error(exc)))
    return NanozymeExtractionBatch(rows=valid_rows), rejected_rows


def parse_nanozyme_batch_json(text: str) -> dict[str, Any]:
    stripped = strip_markdown_json_fence(text.strip())
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = json.loads(extract_first_json_object(stripped))

    if isinstance(payload, list):
        return normalize_nano_payload({"rows": payload})
    if isinstance(payload, dict):
        return normalize_nano_payload(payload)
    raise ValueError("Nanozyme extractor JSON payload must be an object or list.")


def normalize_nano_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return payload

    normalized_rows: list[Any] = []
    for row in rows:
        normalized_rows.append(normalize_nano_row(row) if isinstance(row, dict) else row)

    normalized = dict(payload)
    normalized["rows"] = normalized_rows
    return normalized


def normalize_nano_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "material_id": pick_first(row, "material_id", "compound_id", "id") or "material",
        "material_name": pick_first(row, "material_name", "material", "name") or "",
        "material_formula": pick_first(row, "material_formula", "formula", "composition") or "",
        "property_name": pick_first(row, "property_name", "property", "endpoint") or "",
        "value": parse_numeric_value(row.get("value")),
        "unit": pick_first(row, "unit", "units") or "",
        "assay": pick_first(row, "assay", "method", "measurement") or "",
        "condition": pick_first(row, "condition", "conditions") or "",
    }
    if not normalized["material_formula"]:
        normalized["material_formula"] = first_formula_candidate(normalized["material_name"])
    if not normalized["material_name"]:
        normalized["material_name"] = normalized["material_formula"]
    return normalized


def validate_and_normalize_nanozyme_rows(
    rows: list[ExtractedNanozymeProperty],
) -> tuple[list[ValidatedNanozymeProperty], list[str], list[str]]:
    validated, errors, warnings, _rejected = validate_and_normalize_nanozyme_rows_detailed(rows)
    return validated, errors, warnings


def validate_and_normalize_nanozyme_rows_detailed(
    rows: list[ExtractedNanozymeProperty],
) -> tuple[list[ValidatedNanozymeProperty], list[str], list[str], list[dict[str, Any]]]:
    validated: list[ValidatedNanozymeProperty] = []
    errors: list[str] = []
    warnings: list[str] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str, str]] = set()

    for index, row in enumerate(rows, start=1):
        formula_validation = validate_material_formula(row.material_formula)
        canonical_property, property_errors = validate_nanozyme_property_name(row.property_name)
        row_errors = [
            f"row {index} material_id={row.material_id}: {error}" for error in formula_validation.errors
        ]
        row_errors.extend(
            f"row {index} material_id={row.material_id}: {error}" for error in sanity_check_nanozyme_value(row)
        )
        if property_errors:
            reason = "; ".join(
                f"row {index} material_id={row.material_id}: {error}" for error in property_errors
            )
            warnings.append(reason)
            rejected.append(build_rejected_nano_row(row, index=index, reason=reason))
            continue
        if row_errors:
            errors.extend(row_errors)
            rejected.append(build_rejected_nano_row(row, index=index, reason="; ".join(row_errors)))
            continue

        key = (
            formula_validation.normalized_formula,
            normalize_text(canonical_property).lower(),
            normalize_text(row.unit).lower(),
            normalize_text(row.assay).lower(),
            normalize_text(row.condition).lower(),
        )
        if key in seen_keys:
            reason = (
                f"row {index} material_id={row.material_id}: duplicate {canonical_property} for "
                f"{formula_validation.normalized_formula} skipped"
            )
            warnings.append(reason)
            rejected.append(build_rejected_nano_row(row, index=index, reason=reason))
            continue

        seen_keys.add(key)
        row_payload = row.model_dump()
        row_payload["property_name"] = canonical_property
        validated.append(ValidatedNanozymeProperty(**row_payload, normalized_formula=formula_validation.normalized_formula))

    return validated, errors, warnings, rejected


def validate_nanozyme_property_name(property_name: str) -> tuple[str, list[str]]:
    text = normalize_text(property_name)
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not normalized:
        return text, ["property_name is empty"]
    if not any(token in normalized for token in ALLOWED_NANO_PROPERTY_TOKENS):
        return text, [f"property_name {property_name!r} is not an allowed nanozyme endpoint"]
    return canonical_nanozyme_property_name(text), []


def canonical_nanozyme_property_name(property_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", property_name.lower()).strip()
    suffix = ""
    if normalized.endswith(" lower"):
        normalized = normalized[: -len(" lower")].strip()
        suffix = "_lower"
    elif normalized.endswith(" upper"):
        normalized = normalized[: -len(" upper")].strip()
        suffix = "_upper"

    if "vmax" in normalized:
        return "Vmax" + suffix
    if "kcat" in normalized:
        return "kcat" + suffix
    if re.search(r"\bkm\b", normalized):
        return "Km" + suffix
    if "zeta" in normalized:
        return "zeta potential" + suffix
    if "ic50" in normalized:
        return "IC50" + suffix
    if "viability" in normalized:
        return "cell viability" + suffix
    if "cytotoxic" in normalized:
        return "cytotoxicity" + suffix
    if "diameter" in normalized:
        return "particle diameter" + suffix
    if "hydrodynamic" in normalized:
        return "hydrodynamic size" + suffix
    if "size" in normalized:
        return "particle size" + suffix
    if "surface area" in normalized or "bet" in normalized:
        return "BET surface area" + suffix
    if "limit" in normalized or "lod" in normalized:
        return "limit of detection" + suffix
    if "range" in normalized:
        return "linear range" + suffix
    if "ph" in normalized:
        return "optimum pH" + suffix
    if "temperature" in normalized or "temp" in normalized:
        return "optimum temperature" + suffix
    if "rsd" in normalized:
        return "RSD" + suffix
    if "activity" in normalized:
        return "activity" + suffix
    if "yield" in normalized:
        return property_name.strip()
    if "conversion" in normalized:
        return property_name.strip()
    if "selectivity" in normalized:
        return property_name.strip()
    return property_name.strip()


def build_rejected_nano_row(row: ExtractedNanozymeProperty, index: int, reason: str) -> dict[str, Any]:
    payload = row.model_dump()
    payload.update(
        {
            "row_index": index,
            "reason": reason,
            "validator": "nano_critic",
        }
    )
    return payload


def build_rejected_nano_payload(row: Any, index: int, reason: str) -> dict[str, Any]:
    payload = normalize_nano_row(row) if isinstance(row, dict) else {}
    payload.update(
        {
            "row_index": index,
            "reason": f"row {index}: {reason}",
            "validator": "nano_extractor_schema",
        }
    )
    return payload


def format_validation_error(error: ValidationError) -> str:
    messages: list[str] = []
    for item in error.errors():
        field = ".".join(str(part) for part in item.get("loc", [])) or "row"
        messages.append(f"{field}: {item.get('msg', 'invalid value')}")
    return "; ".join(messages)


def validate_material_formula(formula: str) -> FormulaValidation:
    normalized = clean_formula(formula)
    if not normalized:
        return FormulaValidation(normalized_formula="", errors=["material_formula is empty"])

    pymatgen_validation = validate_formula_with_pymatgen(normalized)
    if pymatgen_validation is not None:
        return pymatgen_validation

    tokens = list(FORMULA_TOKEN_RE.finditer(normalized))
    if not tokens:
        return FormulaValidation(normalized_formula=normalized, errors=[f"formula {formula!r} has no elements"])

    errors: list[str] = []
    for token in tokens:
        element = token.group(1)
        count = token.group(2)
        if element not in PERIODIC_TABLE:
            errors.append(f"unknown element {element!r} in formula {formula!r}")
        if count and float(count) <= 0:
            errors.append(f"non-positive element count {count!r} in formula {formula!r}")

    remainder = FORMULA_TOKEN_RE.sub("", normalized)
    remainder = re.sub(r"\d+(?:\.\d+)?", "", remainder)
    remainder = re.sub(r"[()\[\]{}.,;:+\-_/@·*]", "", remainder)
    if remainder:
        errors.append(f"unparsed formula fragment {remainder!r} in formula {formula!r}")

    if errors and looks_like_catalyst_label(normalized):
        return FormulaValidation(normalized_formula=normalized, errors=[])

    return FormulaValidation(normalized_formula=normalized, errors=errors)


def looks_like_catalyst_label(normalized_formula: str) -> bool:
    if not CATALYST_LABEL_RE.search(normalized_formula):
        return False
    known_elements = {
        match.group(1)
        for match in FORMULA_TOKEN_RE.finditer(normalized_formula)
        if match.group(1) in PERIODIC_TABLE
    }
    return bool(known_elements)


def validate_formula_with_pymatgen(formula: str) -> FormulaValidation | None:
    try:
        from pymatgen.core import Composition
    except ImportError:
        return None

    try:
        composition = Composition(formula)
    except Exception as exc:
        return FormulaValidation(normalized_formula=formula, errors=[f"pymatgen rejected formula {formula!r}: {exc}"])
    if not composition.elements:
        return FormulaValidation(normalized_formula=formula, errors=[f"formula {formula!r} has no elements"])
    return FormulaValidation(normalized_formula=composition.reduced_formula, errors=[])


def sanity_check_nanozyme_value(row: ExtractedNanozymeProperty) -> list[str]:
    value = float(row.value)
    property_name = normalize_text(row.property_name).lower()
    unit = normalize_text(row.unit).lower()
    errors: list[str] = []

    if not math.isfinite(value):
        return [f"{row.property_name} value must be finite"]

    if "ph" in property_name:
        if value < 0 or value > 14:
            errors.append(f"pH value {value:g} is outside 0-14")
        return errors

    if any(token in property_name for token in ("temperature", "temp")):
        celsius_value = value - 273.15 if unit in {"k", "kelvin"} else value
        if celsius_value < -273.15:
            errors.append(f"temperature value {value:g} {row.unit} is physically impossible")
        return errors

    if any(token in property_name for token in ("size", "diameter", "radius", "length", "width")):
        size_nm = convert_size_to_nm(value, unit)
        if size_nm is None:
            if value <= 0:
                errors.append(f"{row.property_name} value must be positive")
        elif size_nm < 0.1:
            errors.append(f"{row.property_name} value {value:g} {row.unit} is below 0.1 nm")
        elif size_nm > 1_000_000:
            errors.append(f"{row.property_name} value {value:g} {row.unit} is unrealistically large")
        return errors

    if any(token in property_name for token in ("km", "vmax", "kcat", "lod", "limit", "range", "rate", "velocity")):
        if value <= 0:
            errors.append(f"{row.property_name} value must be positive")
        return errors

    if "recovery" in property_name and (value < 0 or value > 200):
        errors.append(f"recovery value {value:g} is outside 0-200%")
    if "rsd" in property_name and (value < 0 or value > 100):
        errors.append(f"RSD value {value:g} is outside 0-100%")
    if value < 0:
        errors.append(f"{row.property_name} value must not be negative")
    return errors


def convert_size_to_nm(value: float, unit: str) -> float | None:
    compact_unit = re.sub(r"\s+", "", unit.lower())
    if compact_unit in {"nm", "nanometer", "nanometers"}:
        return value
    if compact_unit in {"um", "µm", "μm", "micrometer", "micrometers"}:
        return value * 1000
    if compact_unit in {"m", "meter", "meters"}:
        return value * 1_000_000_000
    if compact_unit in {"angstrom", "angstroms", "a", "å"}:
        return value * 0.1
    return None


def clean_formula(value: str) -> str:
    text = normalize_text(value).translate(SUBSCRIPT_TRANSLATION)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9)])\d*[+-]+$", "", text)
    return text


def first_formula_candidate(value: str) -> str:
    for match in FORMULA_CANDIDATE_RE.finditer(value):
        candidate = clean_formula(match.group(0))
        if validate_material_formula(candidate).errors:
            continue
        return candidate
    return ""


def pick_first(row: dict[str, Any], *keys: str) -> str:
    normalized_keys = {normalize_key(key): key for key in row.keys()}
    for key in keys:
        source_key = normalized_keys.get(normalize_key(key))
        if source_key is None:
            continue
        value = row.get(source_key)
        if value is None:
            continue
        text = normalize_text(value)
        if text:
            return text
    return ""


def parse_numeric_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_text(value).replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not match:
        return float("nan")
    return float(match.group(0))


def normalize_key(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
