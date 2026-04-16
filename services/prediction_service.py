from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.model_registry import ModelRegistry
from utils.storage import find_image_path_by_id, get_image_metadata_by_id


class ImageNotFoundError(FileNotFoundError):
	pass


class MissingImageMetadataError(ValueError):
	pass


@dataclass
class PredictionContext:
	image_id: str
	image_path: str
	modality: str
	result: dict[str, Any]


class PredictionService:
	def __init__(self, registry: ModelRegistry, upload_folder: str, metadata_file: str) -> None:
		self.registry = registry
		self.upload_folder = upload_folder
		self.metadata_file = metadata_file

	def predict_by_image_id(self, image_id: str) -> PredictionContext:
		image_path = find_image_path_by_id(image_id, self.upload_folder)
		if image_path is None:
			raise ImageNotFoundError(f"Image not found for id: {image_id}")

		metadata = get_image_metadata_by_id(image_id, self.metadata_file)
		if not metadata:
			raise MissingImageMetadataError(
				f"Missing metadata for image id '{image_id}'. Please re-upload with modality."
			)
		modality = metadata.get("modality")
		if not modality:
			raise MissingImageMetadataError(
				f"Missing modality in metadata for image id '{image_id}'."
			)

		service = self.registry.resolve(modality)
		result = service.predict(image_path).to_dict()
		return PredictionContext(
			image_id=image_id,
			image_path=image_path,
			modality=modality,
			result=result,
		)
