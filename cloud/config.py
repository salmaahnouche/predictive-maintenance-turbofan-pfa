"""
config.py — Constantes partagées (Cloud)
=========================================
Source unique de vérité pour tous les paramètres du pipeline.
Importé par train_pipeline.py, api_server.py et dashboard.py.

Cohérence avec pc_edge.py :
  DROP_COLS_ESP32 retire : unit_number, time_cycles, setting_3,
                           sensor_1/5/6/10/16/18/19
  → ce qui reste et est envoyé à l'ESP32 (16 features) :
      setting_1, setting_2,
      sensor_2/3/4/7/8/9/11/12/13/14/15/17/20/21

  SENSOR_COLS (14 colonnes) = les 16 features SANS setting_1/setting_2
  → ce sont les seules colonnes utiles pour la détection de panne
  → et les seules stockées dans sensor_history

  RUL : le CNN apprend directement les valeurs brutes (cycles),
        sortie linéaire — aucune normalisation appliquée.
"""

from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent
DB_PATH            = BASE_DIR / "historique.db"
MODEL_CLF_PATH     = BASE_DIR / "rf_fault_classifier.pkl"
MODEL_SCALER_PATH  = BASE_DIR / "scaler_fault_classifier.pkl"
MODEL_IMPUTER_PATH = BASE_DIR / "imputer_fault_classifier.pkl"

# ── 16 features envoyées à l'ESP32 (normalisation MinMax dans train_pipeline) ─
# Utilisées UNIQUEMENT par train_pipeline.py pour le scaler RUL
ESP32_FEATURES = [
    "setting_1", "setting_2",
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",
    "sensor_8",  "sensor_9",  "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]
NB_FEATURES = len(ESP32_FEATURES)  # 16

# ── Colonnes stockées dans sensor_history ─────────────────────────────────────
# = ESP32_FEATURES SANS setting_1/setting_2
SENSOR_COLS = [
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_7",
    "sensor_8",  "sensor_9",  "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
]  # 14 colonnes

# ── Colonnes DROP utilisées par pc_edge.py ─────────────────────────────────────
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

# ── Capteurs utilisés pour la détection de panne HPC / Fan ────────────────────
HPC_SENSORS = ["sensor_3", "sensor_11", "sensor_9"]
FAN_SENSORS = ["sensor_8", "sensor_17", "sensor_2", "sensor_13"]
ALL_SENSORS = sorted(set(HPC_SENSORS + FAN_SENSORS))

# ── Pannes ─────────────────────────────────────────────────────────────────────
FAULT_NAMES  = {0: "HPC Degradation", 1: "Fan Degradation"}
FAULT_COLORS = {0: "#e74c3c", 1: "#3498db"}

# ── Seuils RUL ─────────────────────────────────────────────────────────────────
# Doit correspondre aux #define dans le sketch Arduino :
#   #define RUL_SEUIL_CRITIQUE  30
#   #define RUL_SEUIL_DEGRADE   80
RUL_CRITIQUE = 30
RUL_DEGRADE  = 80

# ── Niveaux d'alerte ───────────────────────────────────────────────────────────
ALERT_LEVELS = {
    "CRITIQUE":      {"label": "Critique",     "color": "#e74c3c", "icon": "🔴"},
    "AVERTISSEMENT": {"label": "Avertissement", "color": "#f39c12", "icon": "🟡"},
    "NORMAL":        {"label": "Normal",        "color": "#27ae60", "icon": "🟢"},
}

# Mapping statuts ESP32 → niveaux cloud
STATUT_MAP = {
    "SAIN":          "NORMAL",
    "NORMAL":        "NORMAL",
    "DEGRADE":       "AVERTISSEMENT",
    "WARNING":       "AVERTISSEMENT",
    "AVERTISSEMENT": "AVERTISSEMENT",
    "CRITIQUE":      "CRITIQUE",
    "CRITICAL":      "CRITIQUE",
}

# ── Modèle CNN ─────────────────────────────────────────────────────────────────
TIME_STEPS = 30    # doit correspondre à #define TIME_STEPS dans l'Arduino
CNN_BATCH  = 64

# RUL brut — le CNN prédit directement en cycles, sans normalisation.
# ASYM_ALPHA = 1.0 → MSE classique symétrique
# Augmenter alpha pour pénaliser davantage la surestimation (dangereux en industrie)
ASYM_ALPHA = 1.0
CNN_EPOCHS = 50

# ── Classification 3 classes ───────────────────────────────────────────────────
RUL_SEUILS    = {"critique": RUL_CRITIQUE, "degrade": RUL_DEGRADE}
CLASSE_NOMS   = ["Critique", "Dégradé", "Sain"]
CLASSE_COLORS = ["#e74c3c", "#f39c12", "#27ae60"]

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST      = "0.0.0.0"
API_PORT      = 8000
CLOUD_API_URL = f"http://localhost:{API_PORT}"