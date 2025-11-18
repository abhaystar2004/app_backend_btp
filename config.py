import os


class BaseConfig:

	# Paths
	BASE_DIR = os.path.abspath(os.path.dirname(__file__))
	UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
	RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
	REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

	# Upload restrictions
	ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}
	MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

	# Public base URL override (set via env when deploying). Leave blank to auto-detect per request.
	SERVER_BASE_URL = os.environ.get("SERVER_BASE_URL", "")

	# CORS
	CORS_ALLOW_ALL = True

	# Model
	MODEL_PATH = os.path.join(BASE_DIR, "best_oct_classifier_ft.h5")  # update filename if different
	CLASS_NAMES = ['AMD','CNV','CSR','DME','DR','DRUSEN','MH','NORMAL']


Config = BaseConfig


