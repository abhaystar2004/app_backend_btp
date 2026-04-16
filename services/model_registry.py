from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.model_services import (
	BaseModelService,
	FundusClassifierService,
	OCTClassifierService,
)


class UnsupportedModalityError(ValueError):
	pass


@dataclass(frozen=True)
class ModelSpec:
	modality: str
	service: str
	framework: str
	task: str
	model_path: str
	class_names: list[str] | None
	input_size: tuple[int, int]
	multi_label: bool
	positive_threshold: float
	normalize_mean: tuple[float, float, float] | None
	normalize_std: tuple[float, float, float] | None

	@staticmethod
	def _normalize_input_size(raw_input_size: Any) -> tuple[int, int]:
		if not isinstance(raw_input_size, (list, tuple)) or len(raw_input_size) != 2:
			raise ValueError("MODEL_SPECS[*].input_size must be a 2-item list/tuple like [width, height].")
		try:
			width = int(raw_input_size[0])
			height = int(raw_input_size[1])
		except (TypeError, ValueError) as exc:
			raise ValueError("MODEL_SPECS[*].input_size values must be integers.") from exc
		if width <= 0 or height <= 0:
			raise ValueError("MODEL_SPECS[*].input_size values must be positive.")
		return width, height

	@staticmethod
	def _normalize_channel_triplet(raw_values: Any, field_name: str) -> tuple[float, float, float] | None:
		if raw_values is None:
			return None
		if not isinstance(raw_values, (list, tuple)) or len(raw_values) != 3:
			raise ValueError(f"MODEL_SPECS[*].{field_name} must be a 3-item list/tuple.")
		try:
			values = tuple(float(v) for v in raw_values)
		except (TypeError, ValueError) as exc:
			raise ValueError(f"MODEL_SPECS[*].{field_name} values must be numeric.") from exc
		return values

	@classmethod
	def from_dict(cls, modality: str, data: dict[str, Any]) -> "ModelSpec":
		return cls(
			modality=modality,
			service=data["service"],
			framework=data.get("framework", ""),
			task=data.get("task", "classification"),
			model_path=data["model_path"],
			class_names=data.get("class_names"),
			input_size=cls._normalize_input_size(data.get("input_size", [224, 224])),
			multi_label=bool(data.get("multi_label", False)),
			positive_threshold=float(data.get("positive_threshold", 0.5)),
			normalize_mean=cls._normalize_channel_triplet(
				data.get("normalize_mean", [0.485, 0.456, 0.406]), "normalize_mean"
			),
			normalize_std=cls._normalize_channel_triplet(
				data.get("normalize_std", [0.229, 0.224, 0.225]), "normalize_std"
			),
		)


class ModelRegistry:
	def __init__(self, model_specs: dict[str, dict[str, Any]]) -> None:
		self.specs = {k.lower(): ModelSpec.from_dict(k.lower(), v) for k, v in model_specs.items()}
		self._instances: dict[str, BaseModelService] = {}

	def resolve(self, modality: str) -> BaseModelService:
		modality_key = modality.lower()
		if modality_key in self._instances:
			return self._instances[modality_key]

		spec = self.specs.get(modality_key)
		if spec is None:
			raise UnsupportedModalityError(f"Unsupported modality: {modality}")

		service = self._build_service(spec)
		self._instances[modality_key] = service
		return service

	def supported_modalities(self) -> list[str]:
		return sorted(self.specs.keys())

	def _build_service(self, spec: ModelSpec) -> BaseModelService:
		if spec.service == "oct_classifier":
			if not spec.class_names:
				raise ValueError("OCT service requires class_names in MODEL_SPECS.")
			return OCTClassifierService(
				model_path=spec.model_path,
				class_names=spec.class_names,
				input_size=spec.input_size,
			)
		if spec.service == "fundus_classifier":
			return FundusClassifierService(
				model_path=spec.model_path,
				class_names=spec.class_names,
				input_size=spec.input_size,
				multi_label=spec.multi_label,
				positive_threshold=spec.positive_threshold,
				normalize_mean=spec.normalize_mean,
				normalize_std=spec.normalize_std,
			)
		raise ValueError(f"Unknown model service type: {spec.service}")
