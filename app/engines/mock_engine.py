from __future__ import annotations

from app.domain import EngineInput, OCRDocument, OCRItem
from app.engines.base import EngineError, OcrEngine


_PROFILES: dict[str, tuple[OCRItem, ...]] = {
    "default": (
        OCRItem("HASU", 0.9904, box=(100, 100, 200, 145), source_index=0),
        OCRItem("506039", 0.9976, box=(215, 100, 360, 145), source_index=1),
        OCRItem("V542613", 0.9991, box=(500, 300, 590, 550), source_index=2),
        OCRItem("2023.10.27 14:05", 0.9761, box=(700, 800, 950, 835), source_index=3),
    ),
    "container": (
        OCRItem("MSCU", 0.995, box=(100, 100, 200, 145), source_index=0),
        OCRItem("663987", 0.991, box=(215, 100, 360, 145), source_index=1),
        OCRItem("0", 0.989, box=(370, 100, 395, 145), source_index=2),
    ),
    "seal": (
        OCRItem("V542613", 0.9991, box=(100, 100, 190, 360), source_index=0),
    ),
    "empty": (),
}


class MockEngine(OcrEngine):
    name = "mock"
    description = "Returns deterministic OCR items for API and UI integration tests."

    def recognize(self, input_data: EngineInput) -> OCRDocument:
        profile = input_data.mock_profile.strip().lower()
        if profile not in _PROFILES:
            supported = ", ".join(sorted(_PROFILES))
            raise EngineError(f"unknown mock_profile '{profile}'; supported: {supported}")
        return OCRDocument(
            items=_PROFILES[profile],
            source=f"mock:{profile}",
            metadata={"profile": profile},
        )

