# Pneumonia classification capstone

Academic prototype; not for diagnosis.

Install: `python -m pip install -r requirements.txt`. Start both services: `python launch_app.py`.

Frontend: http://localhost:8501. Internal Flask API: http://127.0.0.1:5000/health and POST /predict (file field).

Docker: `docker build -t pneumonia-capstone .` then `docker run --rm -p 8501:8501 pneumonia-capstone`.

Codespaces: launch both services and open forwarded port 8501. Record the actual assigned URL and inference screenshot.

Requires model/weights.pt and model/metadata.json. Never commit raw images or dataset CSVs.
Compressed DICOM may require additional decoder plugins; use the same versions tested by the notebook.
