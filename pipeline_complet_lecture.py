# etape4_pipeline_complet.py
from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf

TIME_STEPS = 50
NB_FEATURES = 16

RUL_SEUIL_CRITIQUE = 30.0
RUL_SEUIL_DEGRADE = 80.0

DROP_COLS = [
    'unit_number', 'time_cycles', 'setting_3',
    'sensor_1', 'sensor_5', 'sensor_6',
    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'
]

SCALER_MIN = [
    -0.0086, -0.0006, 640.84, 1564.3,
    1377.06, 549.61, 2386.9, 9017.98,
    46.69, 517.77, 2386.93, 8099.68,
    8.1563, 388.0, 38.17, 22.8726
]
SCALER_RANGE = [
    0.0172, 0.0013, 4.2700, 51.0900,
    64.1000, 20.8800, 1.7000, 216.3700,
    1.7500, 19.6300, 1.6800, 190.8700,
    0.4142, 11.0000, 1.6800, 1.0779
]


def normaliser(valeurs_brutes):
    out = []
    for i in range(NB_FEATURES):
        if SCALER_RANGE[i] > 0.0:
            v = (valeurs_brutes[i] - SCALER_MIN[i]) / SCALER_RANGE[i]
        else:
            v = 0.0
        out.append(min(1.0, max(0.0, v)))
    return out


def statut_depuis_rul(rul):
    if rul <= RUL_SEUIL_CRITIQUE:
        return "CRITIQUE"
    elif rul <= RUL_SEUIL_DEGRADE:
        return "DEGRADE"
    else:
        return "SAIN"


if __name__ == "__main__":
    CSV_PATH = r"C:\Users\HP\Desktop\pfa_proj\test_FD003_moteur_1.csv"
    MODEL_PATH = r"C:\Users\HP\Desktop\pfa_proj\cerveau_lstm.tflite"
    # --- Charger le modèle ---
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Forme d'entrée attendue par le modèle :", input_details[0]['shape'])

    # --- Charger le CSV ---
    df = pd.read_csv(CSV_PATH)
    features_cols = [c for c in df.columns if c not in DROP_COLS]

    buffer = deque(maxlen=TIME_STEPS)

    for index_ligne in range(len(df)):
        valeurs_brutes = df.iloc[index_ligne][features_cols].tolist()
        valeurs_normalisees = normaliser(valeurs_brutes)
        buffer.append(valeurs_normalisees)

        if len(buffer) < TIME_STEPS:
            continue  # buffer pas encore plein, comme le firmware

        fenetre = np.array(buffer, dtype=np.float32).reshape(input_details[0]['shape'])

        interpreter.set_tensor(input_details[0]['index'], fenetre)
        interpreter.invoke()
        sortie = interpreter.get_tensor(output_details[0]['index'])

        rul = float(np.clip(sortie.flatten()[0], 0.0, 999.0))
        statut = statut_depuis_rul(rul)

        print(f"Ligne {index_ligne} | RUL: {rul:.1f} -> {statut}")