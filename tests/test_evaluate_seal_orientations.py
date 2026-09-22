from pathlib import Path

from PIL import Image

from scripts.evaluate_seal_orientations import (
    create_oriented_image,
    summarize_orientation_records,
)


def test_create_oriented_image_rotates_clockwise_and_counter_clockwise(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGB", (2, 3))
    image.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
    )
    image.save(source)

    cw_path = create_oriented_image(source, "cw90", tmp_path)
    ccw_path = create_oriented_image(source, "ccw90", tmp_path)

    with Image.open(cw_path) as cw:
        assert cw.size == (3, 2)
        assert list(cw.getdata()) == [
            (255, 0, 255),
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 255),
            (255, 255, 0),
            (0, 255, 0),
        ]
    with Image.open(ccw_path) as ccw:
        assert ccw.size == (3, 2)
        assert list(ccw.getdata()) == [
            (0, 255, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 0),
            (0, 0, 255),
            (255, 0, 255),
        ]


def test_orientation_summary_reports_rescues_regressions_and_combined_accuracy() -> None:
    records = []
    exact_by_file = {
        "already.jpg": {"original", "ccw90"},
        "rescued.jpg": {"cw90"},
        "missed.jpg": set(),
    }
    for file_name, exact_orientations in exact_by_file.items():
        for orientation in ("original", "cw90", "ccw90"):
            exact = orientation in exact_orientations
            records.append(
                {
                    "file_name": file_name,
                    "orientation": orientation,
                    "api_success": True,
                    "expected": "EXPECTED",
                    "observed_exact": exact,
                    "candidate_contains": exact,
                    "outcome": "exact" if exact else "wrong",
                    "service_elapsed_ms": 100.0,
                }
            )

    result = summarize_orientation_records(records)

    assert result["by_orientation"]["original"]["top1_exact"] == 1
    assert result["by_orientation"]["cw90"]["top1_exact"] == 1
    assert result["comparison"]["cw90"]["rescued_vs_original"] == ["rescued.jpg"]
    assert result["comparison"]["cw90"]["regressed_vs_original"] == ["already.jpg"]
    assert result["comparison"]["cw90"]["original_or_rotated_exact"] == 2
    assert result["combined"]["any_orientation_exact"] == 2
    assert result["combined"]["any_orientation_accuracy"] == 0.6667
    assert result["combined"]["still_not_exact"] == ["missed.jpg"]
