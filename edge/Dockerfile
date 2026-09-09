FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY envoi_cloud.py .
COPY cerveau_lstm.tflite .
COPY scaler_cloud.pkl .
COPY data/ ./data/

CMD ["python", "envoi_cloud.py"]