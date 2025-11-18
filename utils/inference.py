import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Singleton-like model cache
_model = None

# Default classes; can be overridden by caller if needed
DEFAULT_CLASS_NAMES = ['AMD','CNV','CSR','DME','DR','DRUSEN','MH','NORMAL']


def get_model(model_path: str):
	global _model
	if _model is None:
		_model = load_model(model_path)
	return _model


def predict_oct(image_path: str, model_path: str, class_names: list[str] | None = None):
	if class_names is None:
		class_names = DEFAULT_CLASS_NAMES

	model = get_model(model_path)
	img = load_img(image_path, target_size=(299, 299))
	x = img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	probs = model.predict(x)[0]
	top_idx = int(np.argmax(probs))

	return {
		"predictions": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
		"top_disease": class_names[top_idx],
		"confidence": float(probs[top_idx]),
	}


