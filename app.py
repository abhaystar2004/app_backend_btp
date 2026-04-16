import os
from flask import Flask, jsonify, request, abort, send_file, Response
from flask_cors import CORS
from datetime import datetime
from werkzeug.exceptions import HTTPException
from config import Config
from services.model_registry import ModelRegistry, UnsupportedModalityError
from services.prediction_service import (
	ImageNotFoundError,
	MissingImageMetadataError,
	PredictionService,
)
from services.report_service import build_diagnostic_pdf
from utils.storage import (
	ensure_directories,
	is_allowed_file,
	build_saved_filename,
	generate_image_id,
	save_image_metadata,
)


def create_app(config_overrides: dict | None = None) -> Flask:
	app = Flask(__name__)
	app.config.from_object(Config)
	if config_overrides:
		app.config.update(config_overrides)

	if getattr(Config, "CORS_ALLOW_ALL", False):
		CORS(app)

	# Ensure storage directories exist on startup
	ensure_directories(
		app.config["UPLOAD_FOLDER"],
		app.config["REPORTS_FOLDER"],
	)
	registry = ModelRegistry(app.config["MODEL_SPECS"])
	prediction_service = PredictionService(
		registry=registry,
		upload_folder=app.config["UPLOAD_FOLDER"],
		metadata_file=app.config["IMAGE_METADATA_FILE"],
	)
	app.extensions["model_registry"] = registry
	app.extensions["prediction_service"] = prediction_service

	def success_response(data: dict, message: str | None = None, status_code: int = 200):
		payload = {
			"status": "success",
			"timestamp": datetime.now().isoformat(),
			"data": data,
		}
		if message:
			payload["message"] = message
		return jsonify(payload), status_code

	@app.errorhandler(HTTPException)
	def handle_http_exception(exc: HTTPException):
		return jsonify(
			status="error",
			timestamp=datetime.now().isoformat(),
			error={
				"code": exc.code,
				"type": exc.name,
				"message": exc.description,
			},
		), exc.code

	@app.errorhandler(Exception)
	def handle_unexpected_exception(exc: Exception):
		if app.config.get("DEBUG") or app.config.get("TESTING"):
			message = str(exc)
		else:
			message = "An unexpected error occurred."
		return jsonify(
			status="error",
			timestamp=datetime.now().isoformat(),
			error={
				"code": 500,
				"type": "InternalServerError",
				"message": message,
			},
		), 500

	@app.route("/api/v1/health", methods=["GET"])
	def index():
		return success_response(
			data={
				"service": "retinal-ai-backend",
				"version": "1.1.0",
				"supported_modalities": registry.supported_modalities(),
			},
			message="Retinal AI API is running.",
		)

	@app.route("/api/v1/modalities", methods=["GET"])
	def list_modalities():
		full_names_by_modality = app.config.get("CLASS_FULL_NAMES_BY_MODALITY", {})
		modalities = []
		for modality in registry.supported_modalities():
			spec = app.config["MODEL_SPECS"].get(modality, {})
			class_names = spec.get("class_names") or []
			class_full_names = full_names_by_modality.get(modality, {})
			modalities.append(
				{
					"id": modality,
					"task": spec.get("task", "classification"),
					"framework": spec.get("framework", ""),
					"input_size": spec.get("input_size"),
					"class_count": len(class_names),
					"classes": [
						{
							"code": class_code,
							"name": class_full_names.get(class_code, class_code),
						}
						for class_code in class_names
					],
				}
			)
		return success_response(
			data={
				"default_modality": app.config["DEFAULT_MODALITY"],
				"modalities": modalities,
			}
		)

	@app.route("/api/v1/images", methods=["POST"])
	def upload_image():
		file = request.files.get("image")
		if file is None or file.filename == "":
			abort(400, description="No file provided.")
		modality = request.form.get("modality", app.config["DEFAULT_MODALITY"]).lower().strip()
		if modality not in registry.supported_modalities():
			abort(
				400,
				description=f"Unsupported modality '{modality}'. Supported: {', '.join(registry.supported_modalities())}",
			)

		if not is_allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
			abort(400, description="Unsupported file type.")

		image_id = generate_image_id()
		safe_name = build_saved_filename(image_id, file.filename)
		save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
		file.save(save_path)
		save_image_metadata(
			image_id=image_id,
			metadata_file=app.config["IMAGE_METADATA_FILE"],
			modality=modality,
			original_filename=file.filename,
		)

		return success_response(
			data={
				"image_id": image_id,
				"modality": modality,
				"next": {
					"predict": f"/api/v1/images/{image_id}/predict",
					"report": f"/api/v1/images/{image_id}/report",
				},
			},
			message="Image uploaded successfully.",
			status_code=201,
		)

	@app.route("/api/v1/images/<string:image_id>/predict", methods=["GET"])
	def predict(image_id: str):
		try:
			prediction_context = prediction_service.predict_by_image_id(image_id)
		except ImageNotFoundError:
			abort(404, description="Image not found.")
		except MissingImageMetadataError as exc:
			abort(400, description=str(exc))
		except UnsupportedModalityError as exc:
			abort(400, description=str(exc))

		result = prediction_context.result

		# Build human-friendly summary and recommendation
		top_label = result["top_label"]
		conf_pct = result["confidence"] * 100.0
		if top_label.upper() == "NORMAL":
			summary_text = f"AI suggests no significant abnormalities detected ({conf_pct:.1f}% confidence)."
			recommendation_text = "Routine monitoring advised; follow standard screening intervals."
		else:
			summary_text = f"AI suggests {top_label} with {conf_pct:.1f}% confidence."
			recommendation_text = "Recommend ophthalmology evaluation and correlation with clinical findings."

		return success_response(
			data={
				"image_id": image_id,
				"modality": prediction_context.modality,
				"predictions": result["predictions"],
				"top_label": result["top_label"],
				"confidence": result["confidence"],
				"artifacts": result.get("artifacts", {}),
				"summary": summary_text,
				"recommendation": recommendation_text,
			}
		)

	@app.route("/api/v1/images/<string:image_id>/report", methods=["GET"])
	def generate_pdf_report(image_id: str):
		try:
			prediction_context = prediction_service.predict_by_image_id(image_id)
		except ImageNotFoundError:
			abort(404, description="Image not found.")
		except MissingImageMetadataError as exc:
			abort(400, description=str(exc))
		except UnsupportedModalityError as exc:
			abort(400, description=str(exc))

		image_path = prediction_context.image_path
		result = prediction_context.result
		pdf_path = build_diagnostic_pdf(
			reports_folder=app.config["REPORTS_FOLDER"],
			image_id=image_id,
			image_path=image_path,
			modality=prediction_context.modality,
			predictions=result["predictions"],
			top_label=result["top_label"],
			confidence=result["confidence"],
			class_full_names_by_modality=app.config.get("CLASS_FULL_NAMES_BY_MODALITY"),
		)
		with open(pdf_path, "rb") as fp:
			pdf_bytes = fp.read()

		resp = Response(pdf_bytes, status=200, mimetype="application/pdf")
		resp.headers["Content-Disposition"] = f"inline; filename=report_{image_id}.pdf"
		resp.headers["Content-Length"] = str(len(pdf_bytes))
		# Disable range requests so mobile clients never get a partial (206) response.
		resp.headers["Accept-Ranges"] = "none"
		resp.headers["Cache-Control"] = "no-store"
		# Remove ETag and Last-Modified to prevent conditional/range requests.
		resp.headers.remove("ETag")
		resp.headers.remove("Last-Modified")
		return resp

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)


