import os
import json
import time
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf

# ── Configuration via variables d'environnement (injectées par docker-compose) ──
MOTEUR_ID   = os.environ.get("MOTEUR_ID", "1")
CSV_PATH    = os.environ.get("CSV_PATH", "/app/data/test_FD003_moteur_1.csv")
MODEL_PATH  = os.environ.get("MODEL_PATH", "/app/cerveau_lstm.tflite")
CLOUD_URL   = os.environ.get("CLOUD_URL", "http://cloud:8000/api/maintenance/upload")
SEND_DELAY  = float(os.environ.get("SEND_DELAY", "1.0"))
SCALER_PATH = os.environ.get("SCALER_PATH", "/app/scaler_cloud.pkl")

# ── Features et seuils (identiques à config.py du pipeline d'entraînement) ──────
ESP32_FEATURES = [
    "setting_1", "setting_2",
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",
    "sensor_8",  "sensor_9",  "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]
RUL_SEUIL_CRITIQUE = 30
RUL_SEUIL_DEGRADE  = 80


def statut_depuis_rul(rul: float) -> str:
    if rul <= RUL_SEUIL_CRITIQUE:
        return "CRITIQUE"
    if rul <= RUL_SEUIL_DEGRADE:
        return "DEGRADE"
    return "SAIN"


if __name__ == "__main__":
    print(f"[Moteur {MOTEUR_ID}] Démarrage — CSV={CSV_PATH} | Modèle={MODEL_PATH}")

    # ── Chargement modèle + scaler ──
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Déduit TIME_STEPS directement de la forme du modèle chargé
    # (évite toute désynchronisation avec config.py)
    input_shape = input_details[0]["shape"]  # (1, TIME_STEPS, NB_FEATURES)
    TIME_STEPS  = int(input_shape[1])
    NB_FEATURES = int(input_shape[2])
    print(f"[Moteur {MOTEUR_ID}] Modèle attend TIME_STEPS={TIME_STEPS}, NB_FEATURES={NB_FEATURES}")

    scaler = joblib.load(SCALER_PATH)

    # ── Chargement CSV ──
    df = pd.read_csv(CSV_PATH)
    buffer = deque(maxlen=TIME_STEPS)

    resultats_complets = {
        "session": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "moteurs": {str(MOTEUR_ID): []}
    }

    for index_ligne in range(len(df)):
        ligne = df.iloc[index_ligne]
        capteurs_dict = ligne.to_dict()

        # Normalisation via le scaler entraîné (mêmes colonnes/ordre qu'à l'entraînement)
        valeurs_features = ligne[ESP32_FEATURES].values.reshape(1, -1)
        valeurs_normalisees = scaler.transform(valeurs_features)[0]
        buffer.append(valeurs_normalisees)

        if len(buffer) < TIME_STEPS:
            continue

        fenetre = np.array(buffer, dtype=np.float32).reshape(input_details[0]["shape"])
        interpreter.set_tensor(input_details[0]["index"], fenetre)
        interpreter.invoke()
        sortie = interpreter.get_tensor(output_details[0]["index"])
        rul = float(np.clip(sortie.flatten()[0], 0.0, 999.0))
        statut = statut_depuis_rul(rul)

        entree = {
            "ligne": index_ligne,
            "capteurs": capteurs_dict,
            "rul": rul,
            "statut": statut
        }
        resultats_complets["moteurs"][str(MOTEUR_ID)].append(entree)

        payload_unique = {
            "session": resultats_complets["session"],
            "moteurs": {str(MOTEUR_ID): [entree]}
        }
        try:
            r = requests.post(CLOUD_URL, json=payload_unique, timeout=5)
            print(f"[Moteur {MOTEUR_ID}] cycle {index_ligne} RUL={rul:.1f} ({statut}) → HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ [Moteur {MOTEUR_ID}] cycle {index_ligne} — cloud injoignable : {e}")

        time.sleep(SEND_DELAY)

    with open(f"resultats_rul_moteur{MOTEUR_ID}.json", "w", encoding="utf-8") as f:
        json.dump(resultats_complets, f, indent=2, ensure_ascii=False)
    print(f"[Moteur {MOTEUR_ID}] 💾 JSON sauvegardé. ✅ Simulation terminée.")