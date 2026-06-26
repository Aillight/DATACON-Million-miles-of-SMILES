from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedTable:
    index: int
    markdown: str
    page: int | None = None
    flavor: str | None = None


@dataclass
class ParsedDocument:
    source: str
    parser: str
    markdown: str
    tables: list[ParsedTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_pdf_to_markdown(pdf_path: str | Path, extract_tables: bool = True) -> ParsedDocument:
    path = Path(pdf_path)
    warnings: list[str] = []

    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.name}")

    markdown = _extract_document_markdown(path, warnings)
    tables = _extract_tables(path, warnings) if extract_tables else []
    combined_markdown = _merge_markdown(markdown, tables)

    parser_name = "docling+camelot" if tables else "docling"
    return ParsedDocument(
        source=path.name,
        parser=parser_name,
        markdown=combined_markdown,
        tables=tables,
        warnings=warnings,
    )


def _extract_document_markdown(path: Path, warnings: list[str]) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        warnings.append("Docling is not installed; document text extraction was skipped.")
        return ""

    try:
        result = DocumentConverter().convert(str(path))
        document = result.document
        if hasattr(document, "export_to_markdown"):
            return document.export_to_markdown()
        if hasattr(document, "export_to_text"):
            return document.export_to_text()
        warnings.append("Docling returned a document without a supported export method.")
    except Exception as exc:
        warnings.append(f"Docling failed to parse PDF: {exc}")

    return ""


def _extract_tables(path: Path, warnings: list[str]) -> list[ParsedTable]:
    try:
        import camelot
    except ImportError:
        warnings.append("Camelot is not installed; table extraction was skipped.")
        return []

    tables: list[ParsedTable] = []
    for flavor in ("lattice", "stream"):
        try:
            parsed_tables = camelot.read_pdf(str(path), pages="all", flavor=flavor)
        except Exception as exc:
            warnings.append(f"Camelot {flavor} mode failed: {exc}")
            continue

        for index, table in enumerate(parsed_tables, start=1):
            markdown = _rows_to_markdown(table.df.values.tolist())
            if markdown.strip():
                page = _safe_int(getattr(table, "page", None))
                tables.append(ParsedTable(index=index, markdown=markdown, page=page, flavor=flavor))

        if tables:
            break

    return tables


def _merge_markdown(markdown: str, tables: list[ParsedTable]) -> str:
    chunks = [markdown.strip()] if markdown.strip() else []
    for table in tables:
        title = f"## Table {table.index}"
        if table.page:
            title += f" (page {table.page})"
        chunks.append(f"{title}\n\n{table.markdown}")
    return "\n\n".join(chunks)


def _rows_to_markdown(rows: list[list[Any]]) -> str:
    cleaned_rows = [[_clean_cell(cell) for cell in row] for row in rows if any(_clean_cell(cell) for cell in row)]
    if not cleaned_rows:
        return ""

    width = max(len(row) for row in cleaned_rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in cleaned_rows]
    header = normalized_rows[0]
    body = normalized_rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _clean_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
