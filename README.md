# Retinal AI Flask Backend

Flask backend exposing multi-modality retinal AI APIs (OCT, fundus, etc).

## API Endpoints

Preferred versioned endpoints for frontend:
- `GET /api/v1/health`
- `GET /api/v1/modalities`
- `POST /api/v1/images` (`multipart/form-data` with `image` and `modality`)
- `GET /api/v1/images/<image_id>/predict`
- `GET /api/v1/images/<image_id>/report`

Response envelope:
- Success: `{ "status": "success", "timestamp": "...", "data": { ... } }`
- Error: `{ "status": "error", "timestamp": "...", "error": { "code": 400, "type": "...", "message": "..." } }`

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
flask run --host=0.0.0.0 --port=5000
```

## Notes
- Uploads saved in `uploads/`.
- Upload metadata (including modality) is stored in `uploads/image_metadata.json`.
- Supported modalities are configured in `config.py` under `MODEL_SPECS`.
- Modality class display names are configured in `config.py` under `CLASS_FULL_NAMES_BY_MODALITY`.
- CORS enabled for all origins by default.

