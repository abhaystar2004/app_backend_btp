import os
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
	return f"oct_{uuid.uuid4().hex[:8]}"


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

