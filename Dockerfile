FROM python:3.13-slim

# Tesseract = OCR engine for scanned drawings
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl \
    DRAWCAM_JOBS=/tmp/drawcam-jobs PORT=10000
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY . .
# Fail the build early if the OCR engine or the trained model can't load
RUN tesseract --version | head -1 && python -c "import joblib; joblib.load('drawcam/ml/machine_classifier.joblib'); import drawcam.web.app"

EXPOSE 10000
CMD gunicorn drawcam.web.app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180
