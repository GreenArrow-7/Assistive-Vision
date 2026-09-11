"""Replaceable inference contracts; adapters keep the existing dictionary schema."""
from typing import Protocol, Any

class TextDetector(Protocol):
    def detect(self, frame: Any) -> list[dict]: ...

class OCRProcessor(Protocol):
    def recognize(self, crop: Any) -> list[dict]: ...

class ObjectDetector(Protocol):
    def detect(self, frame: Any) -> tuple[list[dict], list[dict]]: ...

class DistanceEstimator(Protocol):
    def estimate(self, label: str, box: tuple, width: int, height: int) -> int | None: ...

class SpatialAnalyzer(Protocol):
    def annotate(self, items: list[dict], width: int, height: int) -> list[dict]: ...

class NavigationProvider(Protocol):
    def route(self, destination: str, latitude: float | None, longitude: float | None) -> dict: ...

class LocalTextDetector:
    def detect(self, frame):
        from .text_pipeline import detect_text
        return detect_text(frame)

class LocalObjectDetector:
    def detect(self, frame):
        from .detector import detect_objects
        return detect_objects(frame)
