import os
import json
import uuid
from werkzeug.utils import secure_filename


def ensure_directories(*directories: str) -> None:
	for directory in directories:
		os.makedirs(directory, exist_ok=True)


def is_allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
	if "." not in filename:
		return False
	extension = filename.rsplit(".", 1)[1].lower()
	return extension in allowed_extensions


def generate_image_id() -> str:
	return f"img_{uuid.uuid4().hex[:8]}"


def build_saved_filename(image_id: str, original_filename: str) -> str:
	return f"{image_id}_" + secure_filename(original_filename)


def find_image_path_by_id(image_id: str, upload_folder: str) -> str | None:
	"""
	Find the first file in upload_folder that starts with '<image_id>_'.
	Returns absolute path or None.
	"""
	if not os.path.isdir(upload_folder):
		return None
	prefix = f"{image_id}_"
	for name in os.listdir(upload_folder):
		if name.startswith(prefix):
			return os.path.join(upload_folder, name)
	return None


def _load_metadata_store(metadata_file: str) -> dict[str, dict]:
	if not os.path.exists(metadata_file):
		return {}
	try:
		with open(metadata_file, "r", encoding="utf-8") as fp:
			data = json.load(fp)
			return data if isinstance(data, dict) else {}
	except (json.JSONDecodeError, OSError):
		return {}


def _save_metadata_store(metadata_file: str, data: dict[str, dict]) -> None:
	parent = os.path.dirname(metadata_file)
	if parent:
		os.makedirs(parent, exist_ok=True)
	with open(metadata_file, "w", encoding="utf-8") as fp:
		json.dump(data, fp, indent=2)


def save_image_metadata(
	image_id: str,
	metadata_file: str,
	modality: str,
	original_filename: str | None = None,
) -> None:
	store = _load_metadata_store(metadata_file)
	store[image_id] = {
		"modality": modality.lower(),
		"original_filename": original_filename,
	}
	_save_metadata_store(metadata_file, store)


def get_image_metadata_by_id(image_id: str, metadata_file: str) -> dict | None:
	store = _load_metadata_store(metadata_file)
	metadata = store.get(image_id)
	if not isinstance(metadata, dict):
		return None
	return metadata

