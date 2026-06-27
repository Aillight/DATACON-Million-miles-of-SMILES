from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_TARGET_SECTIONS = (
    "experimental",
    "experimental section",
    "methods",
    "materials and methods",
    "results",
    "results and discussion",
    "discussion",
)


@dataclass(frozen=True)
class ArticleSection:
    title: str
    body: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextChunk:
    index: int
    section_title: str
    text: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_article_text(
    text: str,
    target_sections: list[str] | tuple[str, ...] | None = None,
    max_chars: int = 6000,
    overlap_chars: int = 400,
) -> list[TextChunk]:
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")

    sections = split_article_sections(text)
    selected_sections = select_sections(sections, target_sections or DEFAULT_TARGET_SECTIONS)
    if not selected_sections and text.strip():
        selected_sections = [ArticleSection(title="Document", body=text.strip())]

    chunks: list[TextChunk] = []
    for section in selected_sections:
        section_text = _with_section_title(section)
        chunks.extend(_chunk_section(section.title, section_text, len(chunks), max_chars, overlap_chars))

    return chunks


def split_article_sections(text: str) -> list[ArticleSection]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []

    matches = list(_iter_section_heading_matches(normalized_text))
    if not matches:
        return [ArticleSection(title="Document", body=normalized_text)]

    sections: list[ArticleSection] = []
    preamble = normalized_text[: matches[0].start()].strip()
    if preamble:
        sections.append(ArticleSection(title="Preamble", body=preamble))

    for index, match in enumerate(matches):
        title = _clean_heading(match.group("title"))
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        body = normalized_text[body_start:body_end].strip()
        if body:
            sections.append(ArticleSection(title=title, body=body))

    return sections


def select_sections(
    sections: list[ArticleSection],
    target_sections: list[str] | tuple[str, ...],
) -> list[ArticleSection]:
    targets = [_normalize_title(section) for section in target_sections]
    if not targets:
        return sections

    selected: list[ArticleSection] = []
    for section in sections:
        normalized_title = _normalize_title(section.title)
        if any(target in normalized_title or normalized_title in target for target in targets):
            selected.append(section)
    return selected


def _chunk_section(
    section_title: str,
    section_text: str,
    start_index: int,
    max_chars: int,
    overlap_chars: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    cursor = 0
    index = start_index

    while cursor < len(section_text):
        hard_end = min(cursor + max_chars, len(section_text))
        end = _find_soft_boundary(section_text, cursor, hard_end)
        chunk_text = section_text[cursor:end].strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    index=index,
                    section_title=section_title,
                    text=chunk_text,
                    start_char=cursor,
                    end_char=end,
                )
            )
            index += 1

        if end >= len(section_text):
            break
        next_cursor = max(0, end - overlap_chars)
        cursor = next_cursor if next_cursor > cursor else end

    return chunks


def _iter_section_heading_matches(text: str):
    heading_pattern = re.compile(
        r"(?im)^(?P<prefix>#{1,6}\s+|\d+(?:\.\d+)*\.?\s+)?"
        r"(?P<title>abstract|introduction|background|related work|methods|materials and methods|"
        r"experimental|experimental section|results|results and discussion|discussion|conclusion|"
        r"conclusions|references|acknowledgements|appendix)\s*$"
    )
    yield from heading_pattern.finditer(text)


def _find_soft_boundary(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)

    window = text[start:hard_end]
    for separator in ("\n\n", ". ", "; ", "\n"):
        offset = window.rfind(separator)
        if offset >= int(len(window) * 0.5):
            return start + offset + len(separator)
    return hard_end


def _with_section_title(section: ArticleSection) -> str:
    return f"## {section.title}\n\n{section.body}".strip()


def _clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("#", "")).strip()


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
