import unittest

from services.model_registry import ModelSpec


class ModelSpecValidationTestCase(unittest.TestCase):
	def test_valid_input_size_normalized(self):
		spec = ModelSpec.from_dict(
			"oct",
			{
				"service": "oct_classifier",
				"model_path": "dummy.h5",
				"input_size": [299, 299],
			},
		)
		self.assertEqual(spec.input_size, (299, 299))
		self.assertFalse(spec.multi_label)
		self.assertEqual(spec.positive_threshold, 0.5)

	def test_multilabel_spec_parsed(self):
		spec = ModelSpec.from_dict(
			"fundus",
			{
				"service": "fundus_classifier",
				"model_path": "dummy.pt",
				"input_size": [224, 224],
				"multi_label": True,
				"positive_threshold": 0.35,
				"normalize_mean": [0.485, 0.456, 0.406],
				"normalize_std": [0.229, 0.224, 0.225],
			},
		)
		self.assertTrue(spec.multi_label)
		self.assertEqual(spec.positive_threshold, 0.35)
		self.assertEqual(spec.normalize_mean, (0.485, 0.456, 0.406))
		self.assertEqual(spec.normalize_std, (0.229, 0.224, 0.225))

	def test_invalid_normalization_triplet_raises(self):
		with self.assertRaises(ValueError):
			ModelSpec.from_dict(
				"fundus",
				{
					"service": "fundus_classifier",
					"model_path": "dummy.pt",
					"normalize_mean": [0.5, 0.5],
				},
			)

	def test_invalid_input_size_length_raises(self):
		with self.assertRaises(ValueError):
			ModelSpec.from_dict(
				"oct",
				{
					"service": "oct_classifier",
					"model_path": "dummy.h5",
					"input_size": [299],
				},
			)

	def test_invalid_input_size_non_positive_raises(self):
		with self.assertRaises(ValueError):
			ModelSpec.from_dict(
				"fundus",
				{
					"service": "fundus_classifier",
					"model_path": "dummy.pt",
					"input_size": [0, 224],
				},
			)


if __name__ == "__main__":
	unittest.main()
