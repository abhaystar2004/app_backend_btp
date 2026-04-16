from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
	Image as RLImage,
	Paragraph,
	SimpleDocTemplate,
	Spacer,
	Table,
	TableStyle,
)


def _safe_image(path: str | None, width: float, height: float):
	if not path:
		return None
	try:
		return RLImage(path, width=width, height=height, kind="proportional")
	except Exception:
		return None


def build_diagnostic_pdf(
	reports_folder: str,
	image_id: str,
	image_path: str,
	modality: str,
	predictions: dict[str, float],
	top_label: str,
	confidence: float,
	class_full_names_by_modality: dict[str, dict[str, str]] | None = None,
) -> str:
	generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	pdf_path = os.path.join(reports_folder, f"{image_id}_report.pdf")
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
	elements.append(Paragraph(f"Scan ID: <b>{image_id}</b>", styles["Normal"]))
	elements.append(Paragraph(f"Modality: <b>{modality}</b>", styles["Normal"]))
	elements.append(Paragraph(f"Generated: {generated_at}", styles["Normal"]))
	elements.append(Spacer(1, 18))

	# Summary and recommendation
	conf_pct = confidence * 100.0
	if top_label.upper() == "NORMAL":
		summary_text = f"AI suggests no significant abnormalities detected ({conf_pct:.1f}% confidence)."
		recommendation_text = "Routine monitoring advised; follow standard screening intervals."
	else:
		summary_text = f"AI suggests {top_label} with {conf_pct:.1f}% confidence."
		recommendation_text = "Recommend ophthalmology evaluation and correlation with clinical findings."

	elements.append(Paragraph("Summary", styles["Heading3"]))
	elements.append(Paragraph(summary_text, styles["BodyText"]))
	elements.append(Spacer(1, 6))
	elements.append(Paragraph("Recommendation", styles["Heading3"]))
	elements.append(Paragraph(recommendation_text, styles["BodyText"]))
	elements.append(Spacer(1, 18))

	# Input image
	img_section_width = doc.width
	img_width = img_section_width
	img_height = img_width * 0.55
	input_img_flow = _safe_image(image_path, img_width, img_height)
	if input_img_flow:
		elements.append(Paragraph("Images", styles["Heading3"]))
		img_cells = [[input_img_flow]]
		img_table = Table(img_cells, colWidths=[img_width])
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
	full_names = (class_full_names_by_modality or {}).get(modality, {})
	for disease, prob in predictions.items():
		display_name = full_names.get(disease, disease)
		pred_rows.append([f"{disease} ({display_name})", f"{prob * 100:.2f}%"])
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

	# Footer note
	elements.append(Spacer(1, 24))
	elements.append(Paragraph(
		"This report is automatically generated and intended for clinical review assistance.\n"
		"It should not be used as the sole basis for diagnosis.",
		styles["BodyText"],
	))

	def draw_footer(canvas, doc_: Any):
		canvas.saveState()
		footer_text = f"AI generated • {generated_at}"
		x = doc_.pagesize[0] - doc_.rightMargin
		y = doc_.bottomMargin - 10
		canvas.setFont("Helvetica", 8)
		canvas.drawRightString(x, y, footer_text)
		canvas.restoreState()

	doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)
	return pdf_path
