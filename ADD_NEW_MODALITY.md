# How to Add a New Modality (New Model) to the App

This repo supports multiple AI models via a single concept: **modality**.

When a user uploads an image, the frontend sends the selected `modality` to:
- `POST /api/v1/images` (upload + save metadata)
- `GET /api/v1/images/:id/predict` (backend routes to the correct model service)
- `GET /api/v1/images/:id/report` (PDF generation)

To add a new modality end-to-end, you will update the backend model registry + services and then (optionally) update the frontend label display mappings.

---

## 0) Decide the modality id and “normal” label

Choose:
1. **Modality id**: the string key you will use in `config.py` (example: `angiography`).
2. **Model service type**: a string in `MODEL_SPECS[*].service` that must be supported by `ModelRegistry`.
3. **Normal label code**: your backend + UI currently treat `top_label.upper() === "NORMAL"` as the “normal/no abnormality” case.

If your new model uses a different label code for normal, you need to update that logic (backend and frontend).

---

## 1) Backend: add modality config

Edit: `app_backend/config.py`

### 1.1) Add a `MODEL_SPECS[<modality>]` entry

The modality keys you add here automatically appear in `GET /api/v1/modalities` because your backend uses:
- `registry = ModelRegistry(app.config["MODEL_SPECS"])`
- `registry.supported_modalities()`

Add something like:

```ts
MODEL_SPECS = {
  ...,
  "angiography": {
    "service": "angiography_classifier",  // must be implemented in model_registry/model_services
    "framework": "tensorflow",             // informational for /modalities; registry uses service string
    "task": "classification",            // informational for /modalities
    "model_path": os.path.join(MODELS_FOLDER, "angiography_classifier.h5"),
    "class_names": ["NORMAL", "SOME_LABEL", "ANOTHER_LABEL"],
    "input_size": [224, 224],           // match your preprocessing expectations
  },
}
```

Notes:
- `input_size` must be a 2-item list/tuple of positive integers.
- For OCT vs Fundus, your existing registry expects specific preprocessing/services.
- If your modality is multi-label, you can reuse keys like `multi_label` and `positive_threshold` (only if your new service supports them).

### 1.2) Add class display names for the PDF

Edit the same file: `CLASS_FULL_NAMES_BY_MODALITY`

The PDF generator uses:
- `class_full_names_by_modality[modality][disease_code]`

Add:

```python
CLASS_FULL_NAMES_BY_MODALITY = {
  ...,
  "angiography": {
    "NORMAL": "Normal (no abnormality)",
    "SOME_LABEL": "Human readable label for SOME_LABEL",
  },
}
```

---

## 2) Backend: implement a model service

Your services live in:
- `app_backend/services/model_services/`

Each modality-specific service must implement the interface:
- `BaseModelService.predict(image_path: str) -> PredictionResult`

The existing pattern:
- `oct_classifier_service.py` implements `OCTClassifierService`
- `fundus_classifier_service.py` implements `FundusClassifierService`

Create a new file, for example:
- `app_backend/services/model_services/angiography_classifier_service.py`

It should:
1. Load the model lazily (recommended pattern in the repo)
2. Preprocess input images to `input_size`
3. Run inference
4. Return a `PredictionResult` with:
   - `predictions`: `{label_code: probability_float}`
   - `top_label`: label_code with max probability (or your preferred top definition)
   - `confidence`: probability for `top_label`
   - optional `artifacts` (used by the frontend results screen for extra fields)

---

## 3) Backend: wire the service into the registry

Edit: `app_backend/services/model_registry.py`

In `_build_service(self, spec: ModelSpec)`, add a branch for your `spec.service` string.

Example (conceptual):

```python
if spec.service == "angiography_classifier":
    return AngiographyClassifierService(
        model_path=spec.model_path,
        class_names=spec.class_names,
        input_size=spec.input_size,
        # plus any additional config fields you added to MODEL_SPECS
    )
```

Also update exports:
- `app_backend/services/model_services/__init__.py`
so `ModelRegistry` (and your codebase conventions) can import the new class cleanly.

---

## 4) Backend: update “normal” logic (if needed)

Two places currently hardcode “normal means label code `NORMAL`”:
- `app_backend/app.py` (summary/recommendation text)
- `app_backend/services/report_service.py` (summary/recommendation text)

If your new modality’s normal label is not literally `"NORMAL"`, you should change this logic.

Recommended approach:
- Add a configurable `NORMAL_LABEL` per modality in `config.py`
- Use it in both places when building summary/recommendation

If you set your new model’s normal label code to `"NORMAL"`, you can skip this section.

---

## 5) Frontend: add label display mapping (recommended)

Even though predictions will work automatically, the UI uses hardcoded mappings for nicer disease names.

Edit: `app_frontend/app/results.tsx`

This file contains:
- `CLASS_FULL_NAMES_BY_MODALITY`
- only `fundus` and `oct` are mapped today

Add your modality:

```ts
const CLASS_FULL_NAMES_BY_MODALITY: Record<string, Record<string, string>> = {
  fundus: FUNDUS_LABEL_FULL_NAMES,
  oct: OCT_LABEL_FULL_NAMES,
  angiography: ANGIOGRAPHY_LABEL_FULL_NAMES,
};
```

Then define `ANGIOGRAPHY_LABEL_FULL_NAMES` with your label codes.

If you skip this, the app will fall back to showing raw label codes.

---

## 6) Frontend: update risk coloring + copy (optional but consistent)

### 6.1) History risk coloring

Edit: `app_frontend/app/history.tsx`

It currently sets “low risk” when:
- `item.prediction.top_label.toUpperCase() === "NORMAL"`

If your new modality normal label differs, update this logic similarly to the backend.

### 6.2) UX strings that mention only OCT/Fundus

Edit:
- `app_frontend/app/index.tsx`
- `app_frontend/app/upload.tsx`

They contain text like “Choose OCT or Fundus”. You may want to generalize wording to “Select modality”.

---

## 7) Testing (optional)

There is a routing-focused test:
- `app_backend/tests/test_multi_model_routing.py`

It uses a fake service and ensures modality routing works.

You can extend it by:
- adding a new modality entry to `MODEL_SPECS` in the test
- registering a `_FakeService` in `registry._instances[...]`
- asserting the new modality comes back in `/predict`

---

## Checklist

- [ ] Add `MODEL_SPECS["<modality>"]` in `app_backend/config.py`
- [ ] Implement a new service in `app_backend/services/model_services/`
- [ ] Add a `ModelRegistry` branch for your new `spec.service`
- [ ] (Optional) Add PDF label mapping in `CLASS_FULL_NAMES_BY_MODALITY`
- [ ] (If needed) Update backend + frontend “NORMAL” assumption
- [ ] (Recommended) Add frontend label display mapping in `app_frontend/app/results.tsx`
- [ ] Update any UI copy that mentions only OCT/Fundus

