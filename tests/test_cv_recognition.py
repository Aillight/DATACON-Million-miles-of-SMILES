import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.vision.cv_recognition import (
    analyze_image_file,
    analyze_image_panels,
    detect_scale_bar,
    parse_scale_label_to_nm,
    segment_particles,
    summarize_particle_measurements,
)


class CvRecognitionTests(unittest.TestCase):
    def test_parse_scale_label_to_nm(self) -> None:
        self.assertEqual(100.0, parse_scale_label_to_nm("100 nm"))
        self.assertEqual(500.0, parse_scale_label_to_nm("0.5 um"))
        self.assertEqual(500.0, parse_scale_label_to_nm("0.5 µm"))
        self.assertEqual(2_000_000.0, parse_scale_label_to_nm("2 mm"))
        self.assertIsNone(parse_scale_label_to_nm("scale bar"))

    def test_detect_scale_bar(self) -> None:
        image = synthetic_microscopy_image()

        scale = detect_scale_bar(image, label_nm=100.0)

        self.assertIsNotNone(scale)
        self.assertAlmostEqual(100.0, scale.length_px, delta=5.0)
        self.assertAlmostEqual(1.0, scale.nm_per_px or 0, delta=0.1)

    def test_segment_particles_with_scale(self) -> None:
        image = synthetic_microscopy_image()
        scale = detect_scale_bar(image, label_nm=100.0)

        particles = segment_particles(image, nm_per_px=scale.nm_per_px if scale else None, min_area_px=50)
        summary = summarize_particle_measurements(particles)

        self.assertGreaterEqual(summary.count, 3)
        self.assertGreater(summary.mean_diameter_nm or 0, 15.0)
        self.assertLess(summary.mean_diameter_nm or 999, 35.0)

    def test_analyze_image_file_writes_expected_result_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "micrograph.png"
            cv2.imwrite(str(image_path), synthetic_microscopy_image())

            result = analyze_image_file(image_path, scale_label="100 nm", min_particle_area_px=50)

        self.assertEqual(str(image_path), result.source)
        self.assertIsNotNone(result.scale_bar)
        self.assertGreaterEqual(result.particle_summary.count, 3)
        self.assertIn("particles", result.to_dict())

    def test_cli_analyzes_image_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "micrograph.png"
            output_dir = temp_path / "out"
            cv2.imwrite(str(image_path), synthetic_microscopy_image())

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_cv_images.py",
                    str(image_path),
                    "--scale-label",
                    "100 nm",
                    "--output-dir",
                    str(output_dir),
                    "--min-particle-area-px",
                    "50",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads((output_dir / "vision_results.json").read_text(encoding="utf-8"))
            self.assertIn("particles=", completed.stdout)
            self.assertGreaterEqual(payload[0]["particle_summary"]["count"], 3)
            self.assertTrue((output_dir / "particle_summary.csv").exists())

    def test_analyze_image_panels_runs_cv_on_detected_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "page.png"
            output_dir = temp_path / "panels"
            cv2.imwrite(str(image_path), synthetic_page_with_micrograph_panel())

            results = analyze_image_panels(
                image_path,
                output_dir=output_dir,
                scale_label="100 nm",
                min_particle_area_px=50,
                min_area_fraction=0.01,
            )

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any(result.particle_summary.count >= 3 for result in results))


def synthetic_microscopy_image() -> np.ndarray:
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    for center, radius in [((70, 80), 10), ((125, 75), 12), ((180, 95), 11), ((90, 145), 9)]:
        cv2.circle(image, center, radius, (20, 20, 20), -1)
    cv2.rectangle(image, (195, 210), (295, 214), (15, 15, 15), -1)
    return image


def synthetic_page_with_micrograph_panel() -> np.ndarray:
    page = np.full((420, 520, 3), 255, dtype=np.uint8)
    panel = synthetic_microscopy_image()
    page[70:310, 90:410] = panel
    cv2.rectangle(page, (85, 65), (415, 315), (30, 30, 30), 2)
    cv2.putText(page, "Fig. 1A", (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    return page


if __name__ == "__main__":
    unittest.main()
