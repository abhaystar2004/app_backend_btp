import os
import base64
from flask import Flask, jsonify, request, abort, render_template, send_file, send_from_directory
from flask_cors import CORS
from datetime import datetime
from config import Config
from utils.storage import (
	ensure_directories,
	is_allowed_file,
	build_saved_filename,
	generate_image_id,
	find_image_path_by_id,
)
from utils.inference import predict_oct
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet


def create_app() -> Flask:
	app = Flask(__name__)
	app.config.from_object(Config)

	if getattr(Config, "CORS_ALLOW_ALL", False):
		CORS(app)

	# Ensure storage directories exist on startup
	ensure_directories(
		app.config["UPLOAD_FOLDER"],
		app.config["RESULTS_FOLDER"],
		app.config["REPORTS_FOLDER"],
	)

	@app.route("/results/<path:filename>", methods=["GET"])
	def get_result_file(filename: str):
		return send_from_directory(app.config["RESULTS_FOLDER"], filename)

	@app.route("/", methods=["GET"])
	def index():
		return jsonify(
			status="success",
			message="OCT-AI API is running!",
			version="1.0.0",
			timestamp=datetime.now().isoformat(),
		)

	@app.route("/upload", methods=["POST"])
	def upload_image():
		file = request.files.get("image")
		if file is None or file.filename == "":
			abort(400, description="No file provided.")

		if not is_allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
			abort(400, description="Unsupported file type.")

		image_id = generate_image_id()
		safe_name = build_saved_filename(image_id, file.filename)
		save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
		file.save(save_path)

		return jsonify(
			status="success",
			image_id=image_id,
			message="Image uploaded successfully.",
		)

	@app.route("/predict/<string:image_id>", methods=["GET"])
	def predict(image_id: str):
		# Find uploaded image path
		image_path = find_image_path_by_id(image_id, app.config["UPLOAD_FOLDER"])
		if image_path is None:
			abort(404, description="Image not found.")

		# Run model inference
		model_path = app.config["MODEL_PATH"]
		result = predict_oct(image_path=image_path, model_path=model_path, class_names=app.config.get("CLASS_NAMES"))

		# Build human-friendly summary and recommendation
		top_label = result["top_disease"]
		conf_pct = result["confidence"] * 100.0
		if top_label.upper() == "NORMAL":
			summary_text = f"AI suggests no significant abnormalities detected ({conf_pct:.1f}% confidence)."
			recommendation_text = "Routine monitoring advised; follow standard screening intervals."
		else:
			summary_text = f"AI suggests {top_label} with {conf_pct:.1f}% confidence."
			recommendation_text = "Recommend ophthalmology evaluation and correlation with clinical findings."

		return jsonify(
			image_id=image_id,
			predictions=result["predictions"],
			top_disease=result["top_disease"],
			confidence=result["confidence"],
			summary=summary_text,
			recommendation=recommendation_text,
		)

	@app.route("/mask/<string:image_id>", methods=["GET"])
	def mask(image_id: str):

		## TODO: Implement mask generation
		if app.config["SERVER_BASE_URL"]:
			base = app.config["SERVER_BASE_URL"].rstrip("/")
		else:
			base = request.host_url.rstrip("/")
		# Provide a fetchable URL for demo result image
		overlay_mask_url = f"{base}/results/result.jpg"
		# retinal_layer_thickness = {
		# 	"rnfl": 76.4,
		# 	"gcl": 31.9,
		# 	"ipl": 21.7,
		# 	"opl": 19.4,
		# }
		retinal_layer_thickness = {
			"rnfl": 76.4,
			"gcl": 31.9,
			"ipl": 21.7,
			"opl": 19.4,
			"inl": 18.1,
			"onl": 85.2
		}

		return jsonify(
			image_id=image_id,
			overlay_mask_url=overlay_mask_url,
			retinal_layer_thickness=retinal_layer_thickness,
			notes="RNFL thinning detected, consistent with glaucoma risk.",
		)

	@app.route("/report/<string:image_id>", methods=["GET"])
	def generate_pdf_report(image_id: str):
		# Locate uploaded OCT image
		image_path = find_image_path_by_id(image_id, app.config["UPLOAD_FOLDER"])
		if image_path is None:
			abort(404, description="Image not found.")

		# Run model inference
		model_path = app.config["MODEL_PATH"]
		result = predict_oct(image_path=image_path, model_path=model_path, class_names=app.config.get("CLASS_NAMES"))

		# Prepare report data (keep thickness as placeholder until segmentation is integrated)
		# TODO: hardcoded data
		data = {
			"image_id": image_id,
			"predictions": result["predictions"],
			"top_disease": result["top_disease"],
			"confidence": result["confidence"],
			"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			# TODO: Implement retinal layer thickness calculation
			"retinal_layer_thickness": {
				"RNFL": 76.4,
				"GCL": 31.9,
				"IPL": 21.7,
				"OPL": 19.4,
				"INL": 18.1,
				"ONL": 85.2,
			},
		}

		pdf_path = os.path.join(app.config["REPORTS_FOLDER"], f"{image_id}_report.pdf")
		doc = SimpleDocTemplate(
			pdf_path,
			pagesize=A4,
			title="OCT Diagnostic Report",
			leftMargin=36,
			rightMargin=36,
			topMargin=48,
			bottomMargin=48,
		)
		styles = getSampleStyleSheet()
		elements = []

		# Title and subtitle
		elements.append(Paragraph("OCT Diagnostic Report", styles["Title"]))
		elements.append(Paragraph("Generated using AI-assisted retinal imaging analysis", styles["Italic"]))
		elements.append(Spacer(1, 12))
		elements.append(Paragraph(f"Scan ID: <b>{data['image_id']}</b>", styles["Normal"]))
		elements.append(Paragraph(f"Generated: {data['generated_at']}", styles["Normal"]))
		elements.append(Spacer(1, 18))

		# Summary and recommendation
		conf_pct = data["confidence"] * 100.0
		if data["top_disease"].upper() == "NORMAL":
			summary_text = f"AI suggests no significant abnormalities detected ({conf_pct:.1f}% confidence)."
			recommendation_text = "Routine monitoring advised; follow standard screening intervals."
		else:
			summary_text = f"AI suggests {data['top_disease']} with {conf_pct:.1f}% confidence."
			recommendation_text = "Recommend ophthalmology evaluation and correlation with clinical findings."

		elements.append(Paragraph("Summary", styles["Heading3"]))
		elements.append(Paragraph(summary_text, styles["BodyText"]))
		elements.append(Spacer(1, 6))
		elements.append(Paragraph("Recommendation", styles["Heading3"]))
		elements.append(Paragraph(recommendation_text, styles["BodyText"]))
		elements.append(Spacer(1, 18))

		# Embedded input and mask images (side-by-side if available)
		def _safe_image(path, width, height):
			try:
				return RLImage(path, width=width, height=height, kind='proportional')
			except Exception:
				return None

		# Locate mask image: prefer per-image overlay; fallback to demo result.jpg
		mask_candidate_1 = os.path.join(app.config["RESULTS_FOLDER"], f"{image_id}_overlay.jpg")
		mask_candidate_2 = os.path.join(app.config["RESULTS_FOLDER"], "result.jpg")
		mask_path = mask_candidate_1 if os.path.exists(mask_candidate_1) else (mask_candidate_2 if os.path.exists(mask_candidate_2) else None)

		img_section_width = doc.width
		col_width = (img_section_width - 12) / 2  # small gap
		img_height = col_width * 0.75

		input_img_flow = _safe_image(image_path, col_width, img_height)
		mask_img_flow = _safe_image(mask_path, col_width, img_height) if mask_path else None

		if input_img_flow or mask_img_flow:
			elements.append(Paragraph("Images", styles["Heading3"]))
			img_cells = [
				[input_img_flow if input_img_flow else Paragraph("Input image unavailable", styles["BodyText"]),
				 mask_img_flow if mask_img_flow else Paragraph("Mask image unavailable", styles["BodyText"])]
			]
			img_table = Table(img_cells, colWidths=[col_width, col_width])
			img_table.setStyle(TableStyle([
				("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
				("ALIGN", (0, 0), (-1, -1), "CENTER"),
				("LEFTPADDING", (0, 0), (-1, -1), 6),
				("RIGHTPADDING", (0, 0), (-1, -1), 6),
				("TOPPADDING", (0, 0), (-1, -1), 6),
				("BOTTOMPADDING", (0, 0), (-1, -1), 6),
			]))
			elements.append(img_table)
			elements.append(Spacer(1, 18))

		# Predictions table
		elements.append(Paragraph("Disease Probability Analysis", styles["Heading3"]))
		pred_rows = [["Disease", "Probability"]]
		disease_full_name = {
			"AMD": "AMD (Age-Related Macular Degeneration)",
			"CNV": "CNV (Choroidal Neovascularization)",
			"CSR": "CSR (Central Serous Retinopathy)",
			"DME": "DME (Diabetic Macular Edema)",
			"DR": "DR (Diabetic Retinopathy)",
			"DRUSEN": "DRUSEN (Diabetic Retinopathy with Uveitis)",
			"MH": "MH (Macular Hole)",
			"NORMAL": "NORMAL (Normal)",
		}
		for disease, prob in data["predictions"].items():
			pred_rows.append([disease_full_name[disease], f"{prob * 100:.2f}%"])
		pred_table = Table(pred_rows, hAlign="LEFT", colWidths=[doc.width * 0.6, doc.width * 0.4])
		pred_table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5AAC")),
			("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
			("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
			("FONTSIZE", (0, 0), (-1, -1), 10),
			("ALIGN", (1, 1), (1, -1), "RIGHT"),
			("LINEBEFORE", (1, 0), (1, -1), 0.25, colors.HexColor("#e1e5ee")),
			("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e1e5ee")),
			("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7cddb")),
			("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fb"), colors.HexColor("#eef2f9")]),
			("LEFTPADDING", (0, 0), (-1, -1), 8),
			("RIGHTPADDING", (0, 0), (-1, -1), 8),
			("TOPPADDING", (0, 0), (-1, -1), 6),
			("BOTTOMPADDING", (0, 0), (-1, -1), 6),
		]))
		elements.append(pred_table)
		elements.append(Spacer(1, 18))

		# Thickness table
		elements.append(Paragraph("Retinal Layer Thickness (µm)", styles["Heading3"]))
		th_rows = [["Layer", "Thickness (µm)"]]
		for layer, value in data["retinal_layer_thickness"].items():
			th_rows.append([layer, f"{value}"])
		th_table = Table(th_rows, hAlign="LEFT", colWidths=[doc.width * 0.6, doc.width * 0.4])
		th_table.setStyle(TableStyle([
			("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E5AAC")),
			("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
			("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
			("FONTSIZE", (0, 0), (-1, -1), 10),
			("ALIGN", (1, 1), (1, -1), "RIGHT"),
			("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e1e5ee")),
			("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7cddb")),
			("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fb"), colors.HexColor("#eef2f9")]),
			("LEFTPADDING", (0, 0), (-1, -1), 8),
			("RIGHTPADDING", (0, 0), (-1, -1), 8),
			("TOPPADDING", (0, 0), (-1, -1), 6),
			("BOTTOMPADDING", (0, 0), (-1, -1), 6),
		]))
		elements.append(th_table)

		# Footer note
		elements.append(Spacer(1, 24))
		elements.append(Paragraph(
			"This report is automatically generated and intended for clinical review assistance.\n"
			"It should not be used as the sole basis for diagnosis.",
			styles["BodyText"],
		))

		# Footer: right-aligned 'AI generated' mark with timestamp
		def draw_footer(canvas, doc_):
			canvas.saveState()
			footer_text = f"AI generated • {data['generated_at']}"
			x = doc_.pagesize[0] - doc_.rightMargin
			y = doc_.bottomMargin - 10
			canvas.setFont("Helvetica", 8)
			canvas.drawRightString(x, y, footer_text)
			canvas.restoreState()

		doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)
		return send_file(
			pdf_path,
			as_attachment=True,
			mimetype="application/pdf",
			download_name=f"report_{image_id}.pdf",
			conditional=True,
		)

	return app


app = create_app()


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)


