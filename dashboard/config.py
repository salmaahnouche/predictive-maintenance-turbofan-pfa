from pathlib import Path
BASE_DIR           = Path(__file__).parent
DB_PATH            = BASE_DIR / "historique.db"
MODEL_CLF_PATH     = BASE_DIR / "rf_fault_classifier.pkl"
MODEL_SCALER_PATH  = BASE_DIR / "scaler_fault_classifier.pkl"
MODEL_IMPUTER_PATH = BASE_DIR / "imputer_fault_classifier.pkl"
ESP32_FEATURES = [
    "setting_1", "setting_2",
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",
    "sensor_8",  "sensor_9",  "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]
NB_FEATURES = len(ESP32_FEATURES)  # 16
SENSOR_COLS = [
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",
    "sensor_8",  "sensor_9",  "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]  # 14 colonnes
DROP_COLS_ESP32 = [          # retire time_cycles → 16 features pour l'ESP32
    "unit_number", "time_cycles", "setting_3",
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]
DROP_COLS_JSON = [           # garde time_cycles → 17 clés dans capteurs_dict
    "unit_number", "setting_3",
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]
DROP_COLS_TRAIN = [          # identique à DROP_COLS_ESP32
    "unit_number", "time_cycles", "setting_3",
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]
HPC_SENSORS = ["sensor_3", "sensor_11", "sensor_9"]
FAN_SENSORS = ["sensor_8", "sensor_17", "sensor_2", "sensor_13"]
ALL_SENSORS = sorted(set(HPC_SENSORS + FAN_SENSORS))
FAULT_NAMES  = {0: "HPC Degradation", 1: "Fan Degradation"}
FAULT_COLORS = {0: "#e74c3c", 1: "#3498db"}
RUL_CRITIQUE = 30
RUL_DEGRADE  = 80
ALERT_LEVELS = {
    "CRITIQUE":      {"label": "Critique",     "color": "#e74c3c", "icon": "🔴"},
    "AVERTISSEMENT": {"label": "Avertissement", "color": "#f39c12", "icon": "🟡"},
    "NORMAL":        {"label": "Normal",        "color": "#27ae60", "icon": "🟢"},
}
STATUT_MAP = {
    "SAIN":          "NORMAL",
    "NORMAL":        "NORMAL",
    "DEGRADE":       "AVERTISSEMENT",
    "WARNING":       "AVERTISSEMENT",
    "AVERTISSEMENT": "AVERTISSEMENT",
    "CRITIQUE":      "CRITIQUE",
    "CRITICAL":      "CRITIQUE",
}
TIME_STEPS = 30  
CNN_BATCH  = 64
ASYM_ALPHA = 1.0
CNN_EPOCHS = 50
RUL_SEUILS    = {"critique": RUL_CRITIQUE, "degrade": RUL_DEGRADE}
CLASSE_NOMS   = ["Critique", "Dégradé", "Sain"]
CLASSE_COLORS = ["#e74c3c", "#f39c12", "#27ae60"]
API_HOST      = "0.0.0.0"
API_PORT      = 8000
import os
CLOUD_API_URL = os.environ.get("CLOUD_API_URL", f"http://localhost:{API_PORT}")