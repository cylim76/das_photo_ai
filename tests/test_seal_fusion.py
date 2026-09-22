from app.pipelines.common import ExtractionCandidate
from app.pipelines.seal_fusion import fuse_seal_candidates


def candidate(value: str, confidence: float, score: float) -> ExtractionCandidate:
    return ExtractionCandidate(
        value=value,
        observed_value=value,
        ocr_confidence=confidence,
        extraction_score=score,
        validation="heuristic",
        inferred=False,
        corrections=0,
        source_texts=(value,),
        source_indices=(0,),
        boxes=(),
    )


def test_fusion_prefers_rotated_complete_value_over_unrelated_original() -> None:
    fused = fuse_seal_candidates(
        {
            "original": [candidate("1000508518", 0.1534, 0.2253)],
            "cw90": [candidate("MLCN4062676", 0.9901, 0.9932)],
            "ccw90": [candidate("110402120220", 0.2120, 0.2652)],
        }
    )

    assert fused[0].value == "MLCN4062676"
    assert fused[0].orientation == "cw90"


def test_fusion_prefers_supported_complete_split_number() -> None:
    fused = fuse_seal_candidates(
        {
            "original": [candidate("WWH200000", 0.55, 0.61)],
            "cw90": [
                candidate("0418426", 0.9989, 0.8002),
                candidate("230418426", 0.9993, 0.7155),
            ],
            "ccw90": [candidate("WWH32", 0.6361, 0.6185)],
        }
    )

    assert fused[0].value == "230418426"
    assert "supported_complete_extension" in fused[0].selection_reasons


def test_fusion_uses_longer_cross_orientation_reading_over_partial_suffix() -> None:
    fused = fuse_seal_candidates(
        {
            "original": [candidate("0488217", 0.9994, 0.8006)],
            "cw90": [candidate("22WWH", 0.5850, 0.5838)],
            "ccw90": [candidate("230488217", 0.9730, 0.7827)],
        }
    )

    assert fused[0].value == "230488217"
    assert fused[0].orientation == "ccw90"


def test_fusion_does_not_reward_implausible_eleven_digit_extension() -> None:
    fused = fuse_seal_candidates(
        {
            "original": [candidate("230418423", 0.96, 0.82)],
            "cw90": [candidate("23041842323", 0.97, 0.84)],
            "ccw90": [candidate("230418423", 0.95, 0.81)],
        }
    )

    assert fused[0].value == "230418423"
    assert "cross_orientation_consensus" in fused[0].selection_reasons
