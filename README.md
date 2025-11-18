# OCT Flask Backend

Flask backend exposing endpoints for OCT analysis.

## Endpoints
- POST `/upload`
- GET `/predict/<image_id>`
- GET `/mask/<image_id>`
- GET `/report/<image_id>`

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
- Links use `http://localhost:5000/` as base; adjust in `config.py`.
- CORS enabled for all origins by default.

