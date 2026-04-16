import io
import os
import tempfile
import unittest

from app import create_app
from services.model_services.base import PredictionResult
from utils.storage import save_image_metadata


class _FakeService:
	def __init__(self, top_label: str):
		self.top_label = top_label

	def predict(self, image_path: str) -> PredictionResult:
		_ = image_path
		return PredictionResult(
			predictions={self.top_label: 0.93},
			top_label=self.top_label,
			confidence=0.93,
		)


class MultiModelRoutingTestCase(unittest.TestCase):
	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()
		self.upload_dir = os.path.join(self.tmp.name, "uploads")
		self.reports_dir = os.path.join(self.tmp.name, "reports")
		metadata_file = os.path.join(self.upload_dir, "image_metadata.json")

		model_specs = {
			"oct": {
				"service": "oct_classifier",
				"framework": "tensorflow",
				"task": "classification",
				"model_path": "fake_oct.h5",
				"class_names": ["NORMAL"],
				"input_size": [299, 299],
			},
			"fundus": {
				"service": "fundus_classifier",
				"framework": "pytorch",
				"task": "classification",
				"model_path": "fake_fundus.pt",
				"class_names": ["DR"],
				"input_size": [224, 224],
			},
		}

		self.app = create_app(
			{
				"TESTING": True,
				"UPLOAD_FOLDER": self.upload_dir,
				"REPORTS_FOLDER": self.reports_dir,
				"IMAGE_METADATA_FILE": metadata_file,
				"MODEL_SPECS": model_specs,
				"DEFAULT_MODALITY": "oct",
			}
		)
		registry = self.app.extensions["model_registry"]
		registry._instances["oct"] = _FakeService("NORMAL")
		registry._instances["fundus"] = _FakeService("DR")
		self.client = self.app.test_client()

	def tearDown(self):
		self.tmp.cleanup()

	def test_upload_rejects_unsupported_modality(self):
		response = self.client.post(
			"/api/v1/images",
			data={
				"modality": "ultrasound",
				"image": (io.BytesIO(b"fake"), "scan.jpg"),
			},
			content_type="multipart/form-data",
		)
		self.assertEqual(response.status_code, 400)
		self.assertIn("Unsupported modality", response.get_data(as_text=True))

	def test_upload_then_predict_oct(self):
		upload_response = self.client.post(
			"/api/v1/images",
			data={
				"modality": "oct",
				"image": (io.BytesIO(b"fake"), "scan.jpg"),
			},
			content_type="multipart/form-data",
		)
		self.assertEqual(upload_response.status_code, 201)
		image_id = upload_response.get_json()["data"]["image_id"]

		predict_response = self.client.get(f"/api/v1/images/{image_id}/predict")
		self.assertEqual(predict_response.status_code, 200)
		payload = predict_response.get_json()["data"]
		self.assertEqual(payload["modality"], "oct")
		self.assertEqual(payload["top_label"], "NORMAL")

	def test_upload_then_predict_fundus(self):
		upload_response = self.client.post(
			"/api/v1/images",
			data={
				"modality": "fundus",
				"image": (io.BytesIO(b"fake"), "scan.jpg"),
			},
			content_type="multipart/form-data",
		)
		self.assertEqual(upload_response.status_code, 201)
		image_id = upload_response.get_json()["data"]["image_id"]

		predict_response = self.client.get(f"/api/v1/images/{image_id}/predict")
		self.assertEqual(predict_response.status_code, 200)
		payload = predict_response.get_json()["data"]
		self.assertEqual(payload["modality"], "fundus")
		self.assertEqual(payload["top_label"], "DR")

	def test_modalities_endpoint(self):
		response = self.client.get("/api/v1/modalities")
		self.assertEqual(response.status_code, 200)
		payload = response.get_json()
		self.assertIn("data", payload)
		self.assertIn("modalities", payload["data"])
		self.assertGreaterEqual(len(payload["data"]["modalities"]), 2)

	def test_health_endpoint(self):
		response = self.client.get("/api/v1/health")
		self.assertEqual(response.status_code, 200)
		payload = response.get_json()
		self.assertEqual(payload["status"], "success")
		self.assertIn("data", payload)

	def test_predict_rejects_missing_metadata(self):
		image_id = "img_nometa01"
		file_path = os.path.join(self.upload_dir, f"{image_id}_scan.jpg")
		os.makedirs(self.upload_dir, exist_ok=True)
		with open(file_path, "wb") as fp:
			fp.write(b"fake")

		response = self.client.get(f"/api/v1/images/{image_id}/predict")
		self.assertEqual(response.status_code, 400)
		self.assertIn("Missing metadata", response.get_data(as_text=True))

	def test_predict_rejects_unsupported_modality_in_metadata(self):
		image_id = "img_badmod01"
		file_path = os.path.join(self.upload_dir, f"{image_id}_scan.jpg")
		os.makedirs(self.upload_dir, exist_ok=True)
		with open(file_path, "wb") as fp:
			fp.write(b"fake")
		save_image_metadata(
			image_id=image_id,
			metadata_file=self.app.config["IMAGE_METADATA_FILE"],
			modality="ultrasound",
			original_filename="scan.jpg",
		)

		response = self.client.get(f"/api/v1/images/{image_id}/predict")
		self.assertEqual(response.status_code, 400)
		self.assertIn("Unsupported modality", response.get_data(as_text=True))


if __name__ == "__main__":
	unittest.main()
