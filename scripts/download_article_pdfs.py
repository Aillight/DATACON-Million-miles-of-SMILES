from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("data/articles")
DEFAULT_USER_AGENT = "DataConChemXBot/0.1 (local research prototype; mailto:example@example.com)"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass(frozen=True)
class ArticleReference:
    pdf_id: str
    doi: str
    title: str = ""
    access: str = ""
    source_csv: str = ""
    row_count: int = 0

    @property
    def key(self) -> str:
        return self.doi.lower() or self.pdf_id.lower()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DownloadResult:
    article: ArticleReference
    status: str
    path: str = ""
    resolver: str = ""
    url: str = ""
    warnings: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "article": self.article.to_dict(),
            "status": self.status,
            "path": self.path,
            "resolver": self.resolver,
            "url": self.url,
            "warnings": self.warnings,
            "candidates": self.candidates,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve DOI rows to open-access article PDFs.")
    parser.add_argument("--csv", action="append", required=True, type=Path, help="Input ChemX CSV path.")
    parser.add_argument("--domain", help="Output subdirectory name. Defaults to input CSV stem.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--access-only", action="store_true", help="Only process rows with access=1.")
    parser.add_argument("--limit", type=int, help="Maximum number of unique articles to process.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    references = read_article_references(args.csv, access_only=args.access_only)
    if args.limit is not None:
        references = references[: args.limit]

    domain = args.domain or infer_domain(args.csv)
    output_dir = args.output_dir / safe_filename(domain)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[DownloadResult] = []
    for index, article in enumerate(references, start=1):
        print(f"[{index}/{len(references)}] {article.doi or article.pdf_id}", flush=True)
        if args.dry_run:
            results.append(DownloadResult(article=article, status="dry-run"))
            continue

        result = download_article_pdf(article, output_dir=output_dir, timeout=args.timeout)
        results.append(result)
        print(f"  {result.status}: {result.path or result.url or '; '.join(result.warnings)}", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    downloaded = sum(1 for result in results if result.status == "downloaded")
    print(f"articles={len(references)}")
    print(f"downloaded={downloaded}")
    print(f"manifest={manifest_path}")


def read_article_references(csv_paths: list[Path], access_only: bool = False) -> list[ArticleReference]:
    grouped: dict[str, dict[str, Any]] = {}

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for row in reader:
                if access_only and normalize_access(row.get("access")) != "1":
                    continue

                doi = normalize_text(row.get("doi"))
                pdf_id = normalize_text(row.get("pdf"))
                if not doi and not pdf_id:
                    continue

                key = (doi or pdf_id).lower()
                entry = grouped.setdefault(
                    key,
                    {
                        "pdf_id": pdf_id,
                        "doi": doi,
                        "title": normalize_text(row.get("title")),
                        "access": normalize_access(row.get("access")),
                        "source_csv": str(csv_path),
                        "row_count": 0,
                    },
                )
                entry["row_count"] += 1
                if not entry["pdf_id"] and pdf_id:
                    entry["pdf_id"] = pdf_id
                if not entry["title"] and row.get("title"):
                    entry["title"] = normalize_text(row.get("title"))

    return [
        ArticleReference(
            pdf_id=entry["pdf_id"],
            doi=entry["doi"],
            title=entry["title"],
            access=entry["access"],
            source_csv=entry["source_csv"],
            row_count=entry["row_count"],
        )
        for entry in sorted(grouped.values(), key=lambda item: (item["doi"].lower(), item["pdf_id"].lower()))
    ]


def download_article_pdf(article: ArticleReference, output_dir: Path, timeout: float = 30.0) -> DownloadResult:
    if not article.doi:
        return DownloadResult(article=article, status="skipped", warnings=["Missing DOI."])

    target_path = output_dir / f"{safe_filename(article.pdf_id or article.doi)}.pdf"
    warnings: list[str] = []
    if target_path.exists() and target_path.stat().st_size > 0 and file_has_pdf_header(target_path):
        return DownloadResult(article=article, status="exists", path=str(target_path))
    if target_path.exists():
        warnings.append(f"Existing file is not a valid PDF and will be retried: {target_path}")

    candidates: list[tuple[str, str]] = []
    for resolver in (resolve_with_openalex, resolve_with_ncbi_oa, resolve_with_crossref):
        try:
            candidates.extend(resolver(article.doi, timeout=timeout))
        except Exception as exc:
            warnings.append(f"{resolver.__name__} failed: {exc}")

    seen_urls: set[str] = set()
    ordered_candidates = []
    for resolver_name, url in candidates:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        ordered_candidates.append((resolver_name, url))

    for resolver_name, url in ordered_candidates:
        try:
            payload, content_type = fetch_url_bytes(url, timeout=timeout)
        except Exception as exc:
            warnings.append(f"download failed from {resolver_name}: {exc}")
            continue

        pdf_payload = extract_pdf_payload(payload, content_type, url)
        if pdf_payload is None:
            warnings.append(f"candidate was not a PDF from {resolver_name}: {url}")
            continue

        target_path.write_bytes(pdf_payload)
        return DownloadResult(
            article=article,
            status="downloaded",
            path=str(target_path),
            resolver=resolver_name,
            url=url,
            warnings=warnings,
            candidates=[candidate_url for _, candidate_url in ordered_candidates],
        )

    return DownloadResult(
        article=article,
        status="not-found",
        warnings=warnings or ["No PDF candidates found."],
        candidates=[candidate_url for _, candidate_url in ordered_candidates],
    )


def resolve_with_openalex(doi: str, timeout: float = 30.0) -> list[tuple[str, str]]:
    quoted_id = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    url = f"https://api.openalex.org/works/{quoted_id}"
    payload, _ = fetch_url_bytes(url, timeout=timeout, accept="application/json")
    data = json.loads(payload.decode("utf-8"))

    urls: list[tuple[str, str]] = []
    for location in openalex_locations(data):
        pdf_url = location.get("pdf_url")
        if isinstance(pdf_url, str) and pdf_url:
            urls.append(("openalex", pdf_url))
    return urls


def openalex_locations(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ("best_oa_location", "primary_location"):
        value = data.get(key)
        if isinstance(value, dict):
            yield value
    locations = data.get("locations")
    if isinstance(locations, list):
        for value in locations:
            if isinstance(value, dict):
                yield value


def resolve_with_crossref(doi: str, timeout: float = 30.0) -> list[tuple[str, str]]:
    quoted_doi = urllib.parse.quote(doi, safe="")
    url = f"https://api.crossref.org/works/{quoted_doi}"
    payload, _ = fetch_url_bytes(url, timeout=timeout, accept="application/json")
    data = json.loads(payload.decode("utf-8"))
    links = data.get("message", {}).get("link", [])

    urls: list[tuple[str, str]] = []
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            link_url = link.get("URL")
            content_type = normalize_text(link.get("content-type")).lower()
            if isinstance(link_url, str) and ("pdf" in content_type or link_url.lower().endswith(".pdf")):
                urls.append(("crossref", link_url))
    return urls


def resolve_with_ncbi_oa(doi: str, timeout: float = 30.0) -> list[tuple[str, str]]:
    pmcid = resolve_pmcid(doi, timeout=timeout)
    if not pmcid:
        return []

    oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={urllib.parse.quote(pmcid)}"
    payload, _ = fetch_url_bytes(oa_url, timeout=timeout, accept="application/xml,text/xml,*/*")
    root = ET.fromstring(payload)

    urls: list[tuple[str, str]] = []
    for link in root.findall(".//link"):
        href = link.attrib.get("href", "")
        if href:
            urls.append(("ncbi-oa", normalize_ncbi_href(href)))
    return urls


def resolve_pmcid(doi: str, timeout: float = 30.0) -> str:
    query = urllib.parse.urlencode({"ids": doi, "format": "json", "tool": "datacon-chemx-local"})
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?{query}"
    payload, _ = fetch_url_bytes(url, timeout=timeout, accept="application/json,*/*")
    data = json.loads(payload.decode("utf-8"))
    records = data.get("records", [])
    if not isinstance(records, list) or not records:
        return ""
    pmcid = records[0].get("pmcid", "")
    return str(pmcid).strip()


def normalize_ncbi_href(href: str) -> str:
    if href.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        return href.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/", 1)
    return href


def fetch_url_bytes(url: str, timeout: float = 30.0, accept: str = "application/pdf,*/*") -> tuple[bytes, str]:
    try:
        return fetch_url_bytes_once(url, timeout=timeout, accept=accept, user_agent=DEFAULT_USER_AGENT)
    except urllib.error.HTTPError as exc:
        if exc.code != 403 or accept == "application/json":
            raise
        return fetch_url_bytes_once(url, timeout=timeout, accept=accept, user_agent=BROWSER_USER_AGENT)


def fetch_url_bytes_once(url: str, timeout: float, accept: str, user_agent: str) -> tuple[bytes, str]:
    headers = {
        "Accept": accept,
        "User-Agent": user_agent,
    }
    referer = infer_referer(url)
    if referer:
        headers["Referer"] = referer

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read()
    return payload, content_type


def infer_referer(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/"


def extract_pdf_payload(payload: bytes, content_type: str, url: str) -> bytes | None:
    if looks_like_pdf(payload, content_type, url):
        return payload
    if looks_like_tgz(payload, content_type, url):
        return extract_pdf_from_tgz(payload)
    return None


def looks_like_pdf(payload: bytes, content_type: str, url: str) -> bool:
    return payload.startswith(b"%PDF-")


def looks_like_tgz(payload: bytes, content_type: str, url: str) -> bool:
    return (
        payload.startswith(b"\x1f\x8b")
        or "gzip" in content_type.lower()
        or url.lower().endswith((".tar.gz", ".tgz"))
    )


def extract_pdf_from_tgz(payload: bytes) -> bytes | None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            pdf_payload = extracted.read()
            if pdf_payload.startswith(b"%PDF-"):
                return pdf_payload
    return None


def file_has_pdf_header(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def infer_domain(csv_paths: list[Path]) -> str:
    if len(csv_paths) == 1:
        return csv_paths[0].stem
    return "articles"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_access(value: Any) -> str:
    text = normalize_text(value)
    if text in {"1", "1.0", "true", "True", "yes", "open"}:
        return "1"
    if text in {"0", "0.0", "false", "False", "no", "closed"}:
        return "0"
    return text


def safe_filename(value: str) -> str:
    text = value.strip() or "article"
    cleaned = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in text)
    return cleaned[:120].strip("._") or "article"


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP error: {exc.code} {exc.reason}") from exc
