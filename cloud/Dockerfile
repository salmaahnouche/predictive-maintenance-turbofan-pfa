FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY API_server.py .
COPY config.py .
COPY rf_fault_classifier.pkl .
COPY scaler_fault_classifier.pkl .
COPY imputer_fault_classifier.pkl .

EXPOSE 8000

CMD ["uvicorn", "API_server:app", "--host", "0.0.0.0", "--port", "8000"]