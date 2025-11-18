import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

model = load_model("best_oct_classifier_ft.h5")
class_names = ['AMD','CNV','CSR','DME','DR','DRUSEN','MH','NORMAL']

def predict_oct(image_path):
    img = load_img(image_path, target_size=(299,299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    probs = model.predict(x)[0]
    top_idx = np.argmax(probs)

    return {
        "predictions": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "top_disease": class_names[top_idx],
        "confidence": float(probs[top_idx])
    }
