from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionResult:
	predictions: dict[str, float]
	top_label: str
	confidence: float
	artifacts: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"predictions": self.predictions,
			"top_label": self.top_label,
			"confidence": self.confidence,
			"artifacts": self.artifacts,
		}


class BaseModelService(ABC):
	"""Common interface every modality-specific model service must implement."""

	@abstractmethod
	def predict(self, image_path: str) -> PredictionResult:
		raise NotImplementedError
