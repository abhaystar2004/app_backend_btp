from __future__ import annotations

import numpy as np

from services.model_services.base import BaseModelService, PredictionResult


class OCTClassifierService(BaseModelService):
	def __init__(
		self,
		model_path: str,
		class_names: list[str],
		input_size: tuple[int, int] = (299, 299),
	) -> None:
		self.model_path = model_path
		self.class_names = class_names
		self.input_size = (int(input_size[0]), int(input_size[1]))
		if self.input_size[0] <= 0 or self.input_size[1] <= 0:
			raise ValueError("OCT input_size must contain positive integers.")
		self._model = None
		self._keras_modules = None

	def _lazy_import_keras(self):
		if self._keras_modules is None:
			try:
				from tensorflow.keras.applications.inception_v3 import preprocess_input
				from tensorflow.keras.models import load_model
				from tensorflow.keras.preprocessing.image import img_to_array, load_img
			except ImportError as exc:
				raise RuntimeError(
					"TensorFlow is required for OCT inference. Install tensorflow to use modality='oct'."
				) from exc
			self._keras_modules = (load_model, load_img, img_to_array, preprocess_input)
		return self._keras_modules

	def _get_model(self):
		load_model, _, _, _ = self._lazy_import_keras()
		if self._model is None:
			self._model = load_model(self.model_path)
		return self._model

	def predict(self, image_path: str) -> PredictionResult:
		_, load_img, img_to_array, preprocess_input = self._lazy_import_keras()
		model = self._get_model()
		img = load_img(image_path, target_size=self.input_size)
		x = img_to_array(img)
		if tuple(x.shape[:2]) != self.input_size:
			raise RuntimeError(
				f"OCT preprocessing resolution mismatch: expected {self.input_size}, got {tuple(x.shape[:2])}."
			)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		probs = model.predict(x)[0]
		top_idx = int(np.argmax(probs))
		predictions = {
			self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
		}
		return PredictionResult(
			predictions=predictions,
			top_label=self.class_names[top_idx],
			confidence=float(probs[top_idx]),
			artifacts={"preprocessed_resolution": [self.input_size[0], self.input_size[1]]},
		)
