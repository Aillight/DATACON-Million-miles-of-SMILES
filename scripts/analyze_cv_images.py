from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.vision.cv_recognition import (
    analyze_image_file,
    analyze_image_panels,
    analyze_pdf_pages,
    write_particle_summary_csv,
    write_vision_results_json,
)


DEFAULT_OUTPUT_DIR = Path("outputs/vision")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CV recognition on a PDF page render or image file.")
    parser.add_argument("input", type=Path, help="PDF, PNG, JPG, or TIFF input.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale-label", help="Manual scale label near the scale bar, for example '100 nm'.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--min-particle-area-px", type=float, default=20.0)
    parser.add_argument(
        "--crop-panels",
        action="store_true",
        help="Detect visual panels first and run CV on saved panel crops instead of full PDF pages.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.input.suffix.lower()
    if suffix == ".pdf":
        results = analyze_pdf_pages(
            pdf_path=args.input,
            output_dir=args.output_dir,
            scale_label=args.scale_label,
            dpi=args.dpi,
            max_pages=args.max_pages,
            crop_panels=args.crop_panels,
        )
    elif args.crop_panels:
        results = analyze_image_panels(
            args.input,
            output_dir=args.output_dir / "panels",
            scale_label=args.scale_label,
            min_particle_area_px=args.min_particle_area_px,
        )
    else:
        results = [
            analyze_image_file(
                args.input,
                scale_label=args.scale_label,
                min_particle_area_px=args.min_particle_area_px,
            )
        ]

    json_path = args.output_dir / "vision_results.json"
    csv_path = args.output_dir / "particle_summary.csv"
    write_vision_results_json(json_path, results)
    write_particle_summary_csv(csv_path, results)

    total_particles = sum(result.particle_summary.count for result in results)
    detected_scales = sum(1 for result in results if result.scale_bar is not None)
    print(f"inputs={len(results)}")
    print(f"scale_bars={detected_scales}")
    print(f"particles={total_particles}")
    print(f"json={json_path}")
    print(f"summary_csv={csv_path}")


if __name__ == "__main__":
    main()
