from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MICRO_UNIT_RE = r"(?:u|micro|\\u00b5|\\u03bc)"
SCALE_LABEL_RE = re.compile(
    rf"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>nm|nanometers?|{MICRO_UNIT_RE}m|micrometers?|mm|cm|m)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImagePanel:
    index: int
    bbox: tuple[int, int, int, int]
    kind: str = "unknown"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleBarDetection:
    bbox: tuple[int, int, int, int]
    length_px: float
    label_nm: float | None = None
    nm_per_px: float | None = None
    polarity: str = "dark"
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParticleMeasurement:
    contour_id: int
    bbox: tuple[int, int, int, int]
    area_px: float
    equivalent_diameter_px: float
    equivalent_diameter_nm: float | None = None
    circularity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParticleSummary:
    count: int
    mean_diameter_nm: float | None = None
    median_diameter_nm: float | None = None
    std_diameter_nm: float | None = None
    min_diameter_nm: float | None = None
    max_diameter_nm: float | None = None
    mean_diameter_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisionAnalysisResult:
    source: str
    image_width: int
    image_height: int
    panels: list[ImagePanel] = field(default_factory=list)
    scale_bar: ScaleBarDetection | None = None
    particles: list[ParticleMeasurement] = field(default_factory=list)
    particle_summary: ParticleSummary = field(default_factory=lambda: ParticleSummary(count=0))
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["panels"] = [panel.to_dict() for panel in self.panels]
        payload["scale_bar"] = self.scale_bar.to_dict() if self.scale_bar else None
        payload["particles"] = [particle.to_dict() for particle in self.particles]
        payload["particle_summary"] = self.particle_summary.to_dict()
        return payload


def analyze_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    scale_label: str | None = None,
    dpi: int = 200,
    max_pages: int | None = None,
    crop_panels: bool = False,
) -> list[VisionAnalysisResult]:
    output_path = Path(output_dir)
    page_images = render_pdf_pages(pdf_path=pdf_path, output_dir=output_path / "pages", dpi=dpi, max_pages=max_pages)
    if not crop_panels:
        return [analyze_image_file(image_path, scale_label=scale_label) for image_path in page_images]

    results: list[VisionAnalysisResult] = []
    for page_index, image_path in enumerate(page_images, start=1):
        panel_dir = output_path / "panels" / f"page_{page_index:03d}"
        results.extend(analyze_image_panels(image_path, output_dir=panel_dir, scale_label=scale_label))
    return results


def render_pdf_pages(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    max_pages: int | None = None,
) -> list[Path]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required to render PDF pages for CV analysis.") from exc

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []
    scale = dpi / 72.0

    pdf = pdfium.PdfDocument(str(path))
    page_count = len(pdf)
    limit = min(page_count, max_pages) if max_pages is not None else page_count
    for page_index in range(limit):
        page = pdf[page_index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image_path = output_path / f"page_{page_index + 1:03d}.png"
        image.save(image_path)
        rendered_paths.append(image_path)

    return rendered_paths


def analyze_image_file(
    image_path: str | Path,
    scale_label: str | None = None,
    min_particle_area_px: float = 20.0,
) -> VisionAnalysisResult:
    cv2, np = require_cv()
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    label_nm = parse_scale_label_to_nm(scale_label) if scale_label else None
    scale_bar = detect_scale_bar(image, label_nm=label_nm)
    nm_per_px = scale_bar.nm_per_px if scale_bar else None
    particles = segment_particles(image, nm_per_px=nm_per_px, min_area_px=min_particle_area_px)
    panels = detect_image_panels(image)
    warnings: list[str] = []
    if scale_label and scale_bar is None:
        warnings.append("Scale label was provided, but no scale bar was detected.")
    if scale_bar and scale_bar.nm_per_px is None:
        warnings.append("Scale bar was detected without a numeric scale label; particle diameters are in pixels only.")

    height, width = image.shape[:2]
    return VisionAnalysisResult(
        source=str(path),
        image_width=width,
        image_height=height,
        panels=panels,
        scale_bar=scale_bar,
        particles=particles,
        particle_summary=summarize_particle_measurements(particles),
        warnings=warnings,
    )


def analyze_image_panels(
    image_path: str | Path,
    output_dir: str | Path,
    scale_label: str | None = None,
    min_particle_area_px: float = 20.0,
    min_area_fraction: float = 0.02,
) -> list[VisionAnalysisResult]:
    cv2, np = require_cv()
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    panels = detect_image_panels(image, min_area_fraction=min_area_fraction)
    if not panels:
        return [analyze_image_file(path, scale_label=scale_label, min_particle_area_px=min_particle_area_px)]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[VisionAnalysisResult] = []
    for panel in panels:
        x, y, w, h = panel.bbox
        crop = image[y : y + h, x : x + w]
        crop_path = output_path / f"{path.stem}_panel_{panel.index:03d}_{panel.kind}.png"
        cv2.imwrite(str(crop_path), crop)
        label_nm = parse_scale_label_to_nm(scale_label) if scale_label else None
        scale_hint = detect_scale_bar(crop, label_nm=label_nm)
        if panel.kind == "plot_or_histogram" and scale_hint is None:
            results.append(
                VisionAnalysisResult(
                    source=str(crop_path),
                    image_width=int(w),
                    image_height=int(h),
                    panels=[panel],
                    warnings=["Plot or histogram panel detected; route this crop to VLM/plot reader."],
                )
            )
            continue

        result = analyze_image_file(
            crop_path,
            scale_label=scale_label,
            min_particle_area_px=min_particle_area_px,
        )
        results.append(
            VisionAnalysisResult(
                source=result.source,
                image_width=result.image_width,
                image_height=result.image_height,
                panels=[panel],
                scale_bar=result.scale_bar,
                particles=result.particles,
                particle_summary=result.particle_summary,
                warnings=result.warnings,
            )
        )
    return results


def detect_scale_bar(image: Any, label_nm: float | None = None) -> ScaleBarDetection | None:
    cv2, np = require_cv()
    gray = to_gray(image)
    height, width = gray.shape[:2]
    candidates: list[ScaleBarDetection] = []

    for polarity, mask in (
        ("dark", cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1]),
        ("light", cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)[1]),
    ):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not looks_like_scale_bar(x, y, w, h, width, height):
                continue
            confidence = scale_bar_confidence(x, y, w, h, width, height)
            nm_per_px = label_nm / float(w) if label_nm and w > 0 else None
            candidates.append(
                ScaleBarDetection(
                    bbox=(int(x), int(y), int(w), int(h)),
                    length_px=float(w),
                    label_nm=label_nm,
                    nm_per_px=nm_per_px,
                    polarity=polarity,
                    confidence=confidence,
                )
            )

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.confidence, candidate.length_px))


def segment_particles(
    image: Any,
    nm_per_px: float | None = None,
    min_area_px: float = 20.0,
    max_area_fraction: float = 0.10,
) -> list[ParticleMeasurement]:
    cv2, np = require_cv()
    gray = to_gray(image)
    height, width = gray.shape[:2]
    mask = foreground_particle_mask(gray)
    mask = remove_scale_bar_region(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = float(width * height) * max_area_fraction
    particles: list[ParticleMeasurement] = []
    for contour_id, contour in enumerate(contours, start=1):
        area = float(cv2.contourArea(contour))
        if area < min_area_px or area > max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        x, y, w, h = cv2.boundingRect(contour)
        equivalent_diameter_px = float(math.sqrt(4.0 * area / math.pi))
        equivalent_diameter_nm = equivalent_diameter_px * nm_per_px if nm_per_px else None
        particles.append(
            ParticleMeasurement(
                contour_id=contour_id,
                bbox=(int(x), int(y), int(w), int(h)),
                area_px=area,
                equivalent_diameter_px=equivalent_diameter_px,
                equivalent_diameter_nm=equivalent_diameter_nm,
                circularity=circularity,
            )
        )

    return sorted(particles, key=lambda particle: (particle.bbox[1], particle.bbox[0]))


def detect_image_panels(image: Any, min_area_fraction: float = 0.02) -> list[ImagePanel]:
    cv2, np = require_cv()
    gray = to_gray(image)
    height, width = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = float(width * height) * min_area_fraction
    panels: list[ImagePanel] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < min_area or w < 40 or h < 40:
            continue
        kind = classify_panel(gray[y : y + h, x : x + w])
        panels.append(
            ImagePanel(
                index=len(panels) + 1,
                bbox=(int(x), int(y), int(w), int(h)),
                kind=kind,
                confidence=min(1.0, area / float(width * height)),
            )
        )

    return sorted(panels, key=lambda panel: (panel.bbox[1], panel.bbox[0]))


def summarize_particle_measurements(particles: list[ParticleMeasurement]) -> ParticleSummary:
    if not particles:
        return ParticleSummary(count=0)

    px_values = [particle.equivalent_diameter_px for particle in particles]
    nm_values = [particle.equivalent_diameter_nm for particle in particles if particle.equivalent_diameter_nm is not None]
    return ParticleSummary(
        count=len(particles),
        mean_diameter_nm=mean(nm_values),
        median_diameter_nm=median(nm_values),
        std_diameter_nm=std(nm_values),
        min_diameter_nm=min(nm_values) if nm_values else None,
        max_diameter_nm=max(nm_values) if nm_values else None,
        mean_diameter_px=mean(px_values),
    )


def parse_scale_label_to_nm(label: str | None) -> float | None:
    if not label:
        return None
    normalized = label.replace("\u00b5", "u").replace("\u03bc", "u")
    match = SCALE_LABEL_RE.search(normalized)
    if not match:
        return None

    value = float(match.group("value").replace(",", "."))
    unit = match.group("unit").lower().replace("\u00b5", "u").replace("\u03bc", "u")
    if unit.startswith("nm") or unit.startswith("nanometer"):
        return value
    if unit in {"um", "micro", "micrometer", "micrometers"} or unit.startswith("micro"):
        return value * 1_000.0
    if unit == "mm":
        return value * 1_000_000.0
    if unit == "cm":
        return value * 10_000_000.0
    if unit == "m":
        return value * 1_000_000_000.0
    return None


def write_vision_results_json(path: str | Path, results: list[VisionAnalysisResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_particle_summary_csv(path: str | Path, results: list[VisionAnalysisResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source",
            "count",
            "mean_diameter_nm",
            "median_diameter_nm",
            "std_diameter_nm",
            "min_diameter_nm",
            "max_diameter_nm",
            "mean_diameter_px",
            "scale_length_px",
            "scale_label_nm",
            "nm_per_px",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            scale = result.scale_bar
            summary = result.particle_summary
            writer.writerow(
                {
                    "source": result.source,
                    "count": summary.count,
                    "mean_diameter_nm": format_optional_float(summary.mean_diameter_nm),
                    "median_diameter_nm": format_optional_float(summary.median_diameter_nm),
                    "std_diameter_nm": format_optional_float(summary.std_diameter_nm),
                    "min_diameter_nm": format_optional_float(summary.min_diameter_nm),
                    "max_diameter_nm": format_optional_float(summary.max_diameter_nm),
                    "mean_diameter_px": format_optional_float(summary.mean_diameter_px),
                    "scale_length_px": format_optional_float(scale.length_px if scale else None),
                    "scale_label_nm": format_optional_float(scale.label_nm if scale else None),
                    "nm_per_px": format_optional_float(scale.nm_per_px if scale else None),
                }
            )


def require_cv():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and numpy are required for CV recognition.") from exc
    return cv2, np


def to_gray(image: Any) -> Any:
    cv2, np = require_cv()
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def looks_like_scale_bar(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> bool:
    if w < max(20, int(image_width * 0.04)):
        return False
    if h < 2 or h > max(12, int(image_height * 0.04)):
        return False
    if w / max(h, 1) < 6:
        return False
    if y < int(image_height * 0.45):
        return False
    return True


def scale_bar_confidence(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> float:
    aspect_score = min(1.0, (w / max(h, 1)) / 20.0)
    bottom_score = min(1.0, max(0.0, (y / image_height - 0.45) / 0.45))
    width_score = min(1.0, w / max(1.0, image_width * 0.25))
    return round(0.45 * aspect_score + 0.35 * bottom_score + 0.20 * width_score, 3)


def foreground_particle_mask(gray: Any) -> Any:
    cv2, np = require_cv()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, light_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return choose_particle_mask(dark_mask, light_mask)


def choose_particle_mask(dark_mask: Any, light_mask: Any) -> Any:
    cv2, np = require_cv()
    dark_score = particle_mask_score(dark_mask)
    light_score = particle_mask_score(light_mask)
    return dark_mask if dark_score >= light_score else light_mask


def particle_mask_score(mask: Any) -> float:
    cv2, np = require_cv()
    foreground_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    if foreground_fraction <= 0.001 or foreground_fraction >= 0.75:
        return 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    fraction_score = 1.0 - abs(foreground_fraction - 0.15)
    return max(0.0, fraction_score) + min(1.0, contour_count / 50.0)


def remove_scale_bar_region(mask: Any) -> Any:
    cv2, np = require_cv()
    cleaned = mask.copy()
    height, width = cleaned.shape[:2]
    bottom_band = cleaned[int(height * 0.75) :, :]
    contours, _ = cv2.findContours(bottom_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if looks_like_scale_bar(x, y + int(height * 0.75), w, h, width, height):
            cv2.rectangle(cleaned, (x, y + int(height * 0.75)), (x + w, y + int(height * 0.75) + h), 0, -1)
    return cleaned


def classify_panel(gray_panel: Any) -> str:
    cv2, np = require_cv()
    height, width = gray_panel.shape[:2]
    margin_y = max(2, int(height * 0.04))
    margin_x = max(2, int(width * 0.04))
    inner = gray_panel[margin_y : height - margin_y, margin_x : width - margin_x]
    if inner.size == 0:
        inner = gray_panel

    edges = cv2.Canny(inner, 50, 150)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    axis_score = np.count_nonzero(horizontal) + np.count_nonzero(vertical)
    texture_score = np.count_nonzero(edges)
    if axis_score > max(120, texture_score * 0.80):
        return "plot_or_histogram"
    if texture_score > max(200, inner.size * 0.01):
        return "microscopy"
    return "unknown"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    if avg is None:
        return None
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"
