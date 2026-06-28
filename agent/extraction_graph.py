from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError


MAX_EXTRACTION_ATTEMPTS = 3
DEFAULT_HUGGINGFACE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"


class ExtractedProperty(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    compound_id: str = Field(min_length=1)
    smiles: str = Field(min_length=1)
    property_name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)


class ExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rows: list[ExtractedProperty] = Field(default_factory=list)


class ValidatedProperty(ExtractedProperty):
    canonical_smiles: str = Field(min_length=1)


class ExtractionPromptContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_text: str
    error_log: list[str] = Field(default_factory=list)
    attempt: int = Field(ge=1)


class ExtractionGraphState(TypedDict, total=False):
    chunk_text: str
    extracted_objects: list[dict[str, Any]]
    validated_objects: list[dict[str, Any]]
    error_log: list[str]
    warnings: list[str]
    valid: bool
    attempts: int
    max_attempts: int
    status: Literal["pending", "valid", "invalid", "failed"]


ExtractorFn = Callable[[ExtractionPromptContext], ExtractionBatch | list[ExtractedProperty] | dict[str, Any]]


def build_extraction_prompt(context: ExtractionPromptContext) -> str:
    error_block = "\n".join(f"- {error}" for error in context.error_log) or "- none"
    return (
        "Extract antibacterial small-molecule activity data from the article chunk.\n"
        "Return only structured rows matching the schema: compound_id, smiles, "
        "property_name, value, unit.\n"
        "Use pMIC as property_name when the source reports normalized pMIC values.\n"
        "If previous RDKit validation errors are listed, fix invalid valence, ring, "
        "or atom syntax issues before returning rows.\n\n"
        f"Attempt: {context.attempt}\n"
        f"Previous validation errors:\n{error_block}\n\n"
        f"Article chunk:\n{context.chunk_text}"
    )


def build_small_molecule_extraction_graph(extractor: ExtractorFn):
    graph = StateGraph(ExtractionGraphState)
    graph.add_node("extractor", lambda state: extractor_node(state, extractor))
    graph.add_node("critic", critic_node)
    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry": "extractor",
            "done": END,
        },
    )
    return graph.compile()


def run_small_molecule_extraction(
    chunk_text: str,
    extractor: ExtractorFn,
    max_attempts: int = MAX_EXTRACTION_ATTEMPTS,
) -> ExtractionGraphState:
    graph = build_small_molecule_extraction_graph(extractor)
    result = graph.invoke(
        {
            "chunk_text": chunk_text,
            "extracted_objects": [],
            "validated_objects": [],
            "error_log": [],
            "warnings": [],
            "valid": False,
            "attempts": 0,
            "max_attempts": max_attempts,
            "status": "pending",
        }
    )
    return result


def make_openai_structured_extractor(
    model: str = "gpt-4.1-mini",
    client: Any | None = None,
    max_output_tokens: int = 2000,
    token: str | None = None,
) -> ExtractorFn:
    if client is None:
        from openai import OpenAI

        resolved_token = resolve_openai_token(token)
        client = OpenAI(api_key=resolved_token) if resolved_token else OpenAI()

    def extractor(context: ExtractionPromptContext) -> ExtractionBatch:
        response = client.responses.parse(
            model=model,
            input=build_extraction_prompt(context),
            text_format=ExtractionBatch,
            max_output_tokens=max_output_tokens,
            temperature=0,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError("OpenAI structured response did not include output_parsed.")
        return ExtractionBatch.model_validate(parsed)

    return extractor


def make_openrouter_structured_extractor(
    model: str = DEFAULT_OPENROUTER_MODEL,
    client: Any | None = None,
    max_output_tokens: int = 2000,
    token: str | None = None,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
) -> ExtractorFn:
    resolved_token = resolve_openrouter_token(token)
    if client is None:
        if not resolved_token:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter extraction.")
        from openai import OpenAI

        client = OpenAI(
            api_key=resolved_token,
            base_url=base_url,
            default_headers=build_openrouter_headers(),
        )

    def extractor(context: ExtractionPromptContext) -> ExtractionBatch:
        schema = ExtractionBatch.model_json_schema()
        text = call_openrouter_json_completion(
            client=client,
            model=model,
            prompt=build_huggingface_extraction_prompt(context, schema),
            schema=schema,
            schema_name="extraction_batch",
            max_output_tokens=max_output_tokens,
            token=resolved_token,
        )
        payload = parse_extraction_batch_json(text)
        return ExtractionBatch.model_validate(payload)

    return extractor


def make_huggingface_structured_extractor(
    model: str = DEFAULT_HUGGINGFACE_MODEL,
    client: Any | None = None,
    max_output_tokens: int = 2000,
    token: str | None = None,
) -> ExtractorFn:
    resolved_token = resolve_huggingface_token(token)
    if client is None:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=model, token=resolved_token)

    def extractor(context: ExtractionPromptContext) -> ExtractionBatch:
        response = call_huggingface_chat_completion(
            client=client,
            model=model,
            context=context,
            max_output_tokens=max_output_tokens,
            token=resolved_token,
        )
        text = extract_huggingface_message_text(response)
        payload = parse_extraction_batch_json(text)
        return ExtractionBatch.model_validate(payload)

    return extractor


def call_huggingface_chat_completion(
    client: Any,
    model: str,
    context: ExtractionPromptContext,
    max_output_tokens: int,
    token: str | None = None,
) -> Any:
    schema = ExtractionBatch.model_json_schema()
    messages = [
        {
            "role": "system",
            "content": (
                "You extract chemical activity data. Return one JSON object only. "
                "The JSON must match the provided schema and contain a rows array. "
                "Use an empty rows array when the chunk has no extractable records."
            ),
        },
        {"role": "user", "content": build_huggingface_extraction_prompt(context, schema)},
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
                "Hugging Face extraction request failed: "
                f"{type(second_exc).__name__}: {safe_exception_message(second_exc, token)}"
            ) from first_exc


def build_huggingface_extraction_prompt(context: ExtractionPromptContext, schema: dict[str, Any]) -> str:
    return (
        f"{build_extraction_prompt(context)}\n\n"
        "Return a single valid JSON object with this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Do not include markdown fences, commentary, citations, or explanatory text."
    )


def call_openrouter_json_completion(
    client: Any,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    max_output_tokens: int,
    token: str | None = None,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "Return one valid JSON object only. Do not include markdown fences or commentary.",
        },
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "messages": messages,
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as first_exc:
        kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as second_exc:
            raise RuntimeError(
                "OpenRouter extraction request failed: "
                f"{type(second_exc).__name__}: {safe_exception_message(second_exc, token)}"
            ) from first_exc
    return extract_chat_message_text(response)


def build_openrouter_headers() -> dict[str, str]:
    load_local_env()
    headers: dict[str, str] = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME") or os.getenv("OPENROUTER_APP_TITLE")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def extract_chat_message_text(response: Any) -> str:
    choices = get_response_value(response, "choices")
    if choices:
        choice = choices[0]
        message = get_response_value(choice, "message")
        content = get_response_value(message, "content") if message is not None else None
        if content:
            return str(content)
        text = get_response_value(choice, "text")
        if text:
            return str(text)
    if isinstance(response, str):
        return response
    raise ValueError("Chat completion response did not contain message content.")


def extract_huggingface_message_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    choices = get_response_value(response, "choices")
    if choices:
        choice = choices[0]
        message = get_response_value(choice, "message")
        content = get_response_value(message, "content") if message is not None else None
        if content:
            return str(content)
        text = get_response_value(choice, "text")
        if text:
            return str(text)

    generated_text = get_response_value(response, "generated_text")
    if generated_text:
        return str(generated_text)

    raise ValueError("Hugging Face response did not contain message content.")


def parse_extraction_batch_json(text: str) -> dict[str, Any]:
    stripped = strip_markdown_json_fence(text.strip())
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = json.loads(extract_first_json_object(stripped))

    if isinstance(payload, list):
        return normalize_extraction_payload({"rows": payload})
    if isinstance(payload, dict):
        return normalize_extraction_payload(payload)
    raise ValueError("Extractor JSON payload must be an object or list.")


def normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return payload

    normalized_rows: list[Any] = []
    for row in rows:
        if isinstance(row, dict):
            row = dict(row)
            property_name = str(row.get("property_name") or "").strip()
            unit = str(row.get("unit") or "").strip()
            if property_name.lower() == "pmic" and not unit:
                row["unit"] = "pMIC"
        normalized_rows.append(row)

    normalized = dict(payload)
    normalized["rows"] = normalized_rows
    return normalized


def strip_markdown_json_fence(text: str) -> str:
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return fence.group(1).strip() if fence else text


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in extractor response.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unterminated JSON object in extractor response.")


def get_response_value(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def resolve_huggingface_token(token: str | None = None) -> str | None:
    return resolve_token_from_env(("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"), token=token)


def resolve_openai_token(token: str | None = None) -> str | None:
    return resolve_token_from_env(("OPENAI_API_KEY",), token=token)


def resolve_openrouter_token(token: str | None = None) -> str | None:
    return resolve_token_from_env(("OPENROUTER_API_KEY", "OPENROUTER_TOKEN"), token=token)


def resolve_token_from_env(names: tuple[str, ...], token: str | None = None) -> str | None:
    if token:
        return token

    load_local_env()
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    streamlit_value = resolve_streamlit_secret(names)
    if streamlit_value:
        return streamlit_value

    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in names:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except OSError:
                    continue
                if value:
                    return str(value)
    except OSError:
        return None
    return None


@lru_cache(maxsize=1)
def load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)


def resolve_streamlit_secret(names: tuple[str, ...]) -> str | None:
    try:
        import streamlit as st
    except Exception:
        return None

    try:
        secrets = st.secrets
        for name in names:
            value = secrets.get(name)
            if value:
                return str(value)
        api_section = secrets.get("api") or secrets.get("llm")
        if isinstance(api_section, dict):
            for name in names:
                value = api_section.get(name)
                if value:
                    return str(value)
    except Exception:
        return None
    return None


def safe_exception_message(exc: Exception, *secrets: str | None) -> str:
    message = str(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "<redacted>")
    return message


def extractor_node(state: ExtractionGraphState, extractor: ExtractorFn) -> ExtractionGraphState:
    attempt = int(state.get("attempts", 0)) + 1
    context = ExtractionPromptContext(
        chunk_text=str(state.get("chunk_text", "")),
        error_log=list(state.get("error_log", [])),
        attempt=attempt,
    )

    try:
        batch = normalize_extractor_output(extractor(context))
    except (TypeError, ValidationError, ValueError, RuntimeError) as exc:
        return {
            "extracted_objects": [],
            "attempts": attempt,
            "valid": False,
            "status": "failed",
            "error_log": append_unique(state.get("error_log", []), f"extractor output validation failed: {exc}"),
        }

    return {
        "extracted_objects": [row.model_dump() for row in batch.rows],
        "attempts": attempt,
        "valid": False,
        "status": "pending",
    }


def critic_node(state: ExtractionGraphState) -> ExtractionGraphState:
    if state.get("status") == "failed":
        return {
            "validated_objects": [],
            "error_log": list(state.get("error_log", [])),
            "warnings": list(state.get("warnings", [])),
            "valid": False,
            "status": "failed",
        }

    extracted_objects = state.get("extracted_objects", [])
    rows = [ExtractedProperty.model_validate(row) for row in extracted_objects]
    validated_rows, errors, warnings = validate_and_canonicalize_rows(rows)

    valid = not errors
    error_log = append_unique(state.get("error_log", []), *errors)
    warning_log = append_unique(state.get("warnings", []), *warnings)
    return {
        "validated_objects": [row.model_dump() for row in validated_rows],
        "error_log": error_log,
        "warnings": warning_log,
        "valid": valid,
        "status": "valid" if valid else "invalid",
    }


def should_retry(state: ExtractionGraphState) -> Literal["retry", "done"]:
    if state.get("valid", False):
        return "done"
    if int(state.get("attempts", 0)) >= int(state.get("max_attempts", MAX_EXTRACTION_ATTEMPTS)):
        return "done"
    return "retry"


def normalize_extractor_output(output: ExtractionBatch | list[ExtractedProperty] | dict[str, Any]) -> ExtractionBatch:
    if isinstance(output, ExtractionBatch):
        return output
    if isinstance(output, list):
        return ExtractionBatch(rows=[ExtractedProperty.model_validate(row) for row in output])
    if isinstance(output, dict):
        return ExtractionBatch.model_validate(output)
    raise TypeError(f"Unsupported extractor output type: {type(output)!r}")


def validate_and_canonicalize_rows(
    rows: list[ExtractedProperty],
) -> tuple[list[ValidatedProperty], list[str], list[str]]:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError as exc:
        raise RuntimeError("RDKit is required for SMILES validation.") from exc
    RDLogger.DisableLog("rdApp.error")

    validated: list[ValidatedProperty] = []
    errors: list[str] = []
    warnings: list[str] = []
    seen_keys: set[tuple[str, str]] = set()

    for index, row in enumerate(rows, start=1):
        molecule = Chem.MolFromSmiles(row.smiles)
        if molecule is None:
            errors.append(
                f"row {index} compound_id={row.compound_id}: RDKit could not parse SMILES {row.smiles!r}"
            )
            continue

        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        key = (canonical_smiles, row.property_name)
        if key in seen_keys:
            warnings.append(
                f"row {index} compound_id={row.compound_id}: duplicate {row.property_name} for {canonical_smiles} skipped"
            )
            continue

        seen_keys.add(key)
        validated.append(ValidatedProperty(**row.model_dump(), canonical_smiles=canonical_smiles))

    return validated, errors, warnings


def append_unique(values: list[str] | tuple[str, ...], *new_values: str) -> list[str]:
    result = list(values)
    seen = set(result)
    for value in new_values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
