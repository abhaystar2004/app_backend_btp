from __future__ import annotations

import numpy as np
from PIL import Image

from services.model_services.base import BaseModelService, PredictionResult


class FundusClassifierService(BaseModelService):
	def __init__(
		self,
		model_path: str,
		class_names: list[str] | None = None,
		input_size: tuple[int, int] = (224, 224),
		multi_label: bool = True,
		positive_threshold: float = 0.5,
		normalize_mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406),
		normalize_std: tuple[float, float, float] | None = (0.229, 0.224, 0.225),
	) -> None:
		self.model_path = model_path
		self.class_names = class_names or []
		self.input_size = (int(input_size[0]), int(input_size[1]))
		if self.input_size[0] <= 0 or self.input_size[1] <= 0:
			raise ValueError("Fundus input_size must contain positive integers.")
		self.multi_label = bool(multi_label)
		self.positive_threshold = float(positive_threshold)
		self.normalize_mean = normalize_mean
		self.normalize_std = normalize_std
		self._model = None
		self._torch = None

	def _lazy_import_torch(self):
		if self._torch is None:
			try:
				import torch
			except ImportError as exc:
				raise RuntimeError(
					"PyTorch is required for fundus inference. Install torch to use modality='fundus'."
				) from exc
			self._torch = torch
		return self._torch

	def _get_model(self):
		torch = self._lazy_import_torch()
		if self._model is None:
			try:
				model = torch.load(self.model_path, map_location="cpu", weights_only=False)
			except ModuleNotFoundError as exc:
				raise RuntimeError(
					f"Fundus model dependency missing while loading checkpoint: {exc}. "
					"Install required packages (at least torchvision) in the backend environment."
				) from exc
			model.eval()
			self._model = model
		return self._model

	def _build_class_names(self, output_dim: int) -> list[str]:
		if self.class_names and len(self.class_names) == output_dim:
			return self.class_names
		return [f"class_{idx}" for idx in range(output_dim)]

	def _preprocess(self, image_path: str):
		torch = self._lazy_import_torch()
		image = Image.open(image_path).convert("RGB").resize(self.input_size)
		arr = np.asarray(image, dtype=np.float32) / 255.0
		if tuple(arr.shape[:2]) != (self.input_size[1], self.input_size[0]):
			raise RuntimeError(
				f"Fundus preprocessing resolution mismatch: expected {(self.input_size[1], self.input_size[0])}, got {tuple(arr.shape[:2])}."
			)
		if self.normalize_mean is not None and self.normalize_std is not None:
			mean = np.asarray(self.normalize_mean, dtype=np.float32)
			std = np.asarray(self.normalize_std, dtype=np.float32)
			arr = (arr - mean) / std
		arr = np.transpose(arr, (2, 0, 1))
		tensor = torch.tensor(arr).unsqueeze(0)
		return tensor

	def predict(self, image_path: str) -> PredictionResult:
		torch = self._lazy_import_torch()
		model = self._get_model()
		input_tensor = self._preprocess(image_path)

		with torch.no_grad():
			output = model(input_tensor)

		if hasattr(output, "logits"):
			logits = output.logits
		else:
			logits = output

		if self.multi_label:
			probs_tensor = torch.sigmoid(logits)
			activation_used = "sigmoid"
		else:
			probs_tensor = torch.softmax(logits, dim=1)
			activation_used = "softmax"
		probs = probs_tensor[0].detach().cpu().numpy()
		class_names = self._build_class_names(output_dim=len(probs))
		top_idx = int(np.argmax(probs))

		predictions = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
		positive_labels = [
			class_names[i] for i in range(len(class_names)) if float(probs[i]) >= self.positive_threshold
		]
		return PredictionResult(
			predictions=predictions,
			top_label=class_names[top_idx],
			confidence=float(probs[top_idx]),
			artifacts={
				"preprocessed_resolution": [self.input_size[0], self.input_size[1]],
				"activation_used": activation_used,
				"positive_threshold": self.positive_threshold,
				"positive_labels": positive_labels,
				"normalization": {
					"mean": list(self.normalize_mean) if self.normalize_mean else None,
					"std": list(self.normalize_std) if self.normalize_std else None,
				},
			},
		)
