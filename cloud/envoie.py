# ============================================================
#  server.py  –  Serveur FastAPI : Réception JSON + Prédiction
#  Lancer avec : uvicorn server:app --host 0.0.0.0 --port 8080
# ============================================================

import numpy as np
import pandas as pd
import joblib
from scipy.stats import linregress
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
HPC_SENSORS = ['sensor_3', 'sensor_11', 'sensor_9']
FAN_SENSORS  = ['sensor_8', 'sensor_17', 'sensor_2', 'sensor_13']
FAULT_NAMES  = {0: 'HPC Degradation', 1: 'Fan Degradation'}
FAULT_COLORS = {0: '#e74c3c', 1: '#3498db'}

# ─────────────────────────────────────────
# CHARGEMENT DES MODÈLES (au démarrage)
# ─────────────────────────────────────────
print("Chargement des modèles...")
try:
    clf         = joblib.load('rf_fault_classifier.pkl')
    scaler_clf  = joblib.load('scaler_fault_classifier.pkl')
    imputer_clf = joblib.load('imputer_fault_classifier.pkl')
    print("  ✅ Modèles chargés avec succès.")
except FileNotFoundError as e:
    print(f"  ❌ Fichier modèle manquant : {e}")
    print("     → Lancez d'abord pipeline2.py pour entraîner et sauvegarder les modèles.")
    clf = scaler_clf = imputer_clf = None

# ─────────────────────────────────────────
# SCHÉMA JSON ATTENDU
# ─────────────────────────────────────────
class CycleData(BaseModel):
    """Un cycle moteur = une ligne de mesures capteurs."""
    time_cycles: int
    setting_1:   float
    setting_2:   float
    setting_3:   float
    sensor_1:    float
    sensor_2:    float
    sensor_3:    float
    sensor_4:    float
    sensor_5:    float
    sensor_6:    float
    sensor_7:    float
    sensor_8:    float
    sensor_9:    float
    sensor_10:   float
    sensor_11:   float
    sensor_12:   float
    sensor_13:   float
    sensor_14:   float
    sensor_15:   float
    sensor_16:   float
    sensor_17:   float
    sensor_18:   float
    sensor_19:   float
    sensor_20:   float
    sensor_21:   float

class MoteurHistorique(BaseModel):
    """
    Historique complet d'un moteur.
    Exemple JSON :
    {
        "unit_id": 42,
        "cycles": [
            {"time_cycles": 1, "sensor_1": 518.67, ...},
            {"time_cycles": 2, "sensor_1": 518.67, ...},
            ...
        ]
    }
    """
    unit_id: int
    cycles:  List[CycleData]

# ─────────────────────────────────────────
# FEATURE ENGINEERING (identique à pipeline2.py)
# ─────────────────────────────────────────
def extraire_features_moteur(df_u: pd.DataFrame) -> pd.Series:
    """Extrait les 20 features à partir de l'historique d'un moteur."""
    all_sensors = list(set(HPC_SENSORS + FAN_SENSORS))
    n   = len(df_u)
    t   = np.arange(n)
    n20 = max(int(n * 0.2), 5)

    def clean(v):
        s = pd.Series(v).ffill().bfill()
        return s.values.astype(float)

    def norm_slope(v):
        v = clean(v)
        r = float(v.max()) - float(v.min())
        if r < 1e-6:
            return 0.0
        slope, _, _, _, _ = linregress(t, v)
        return float(slope / r) if np.isfinite(slope) else 0.0

    def end_delta(v):
        v = clean(v)
        if len(v) == 0 or np.all(np.isnan(v)):
            return 0.0
        end_val   = v[-n20:][~np.isnan(v[-n20:])]
        start_val = v[:n20][~np.isnan(v[:n20])]
        if len(end_val) == 0 or len(start_val) == 0:
            return 0.0
        return float(end_val.mean()) - float(start_val.mean())

    row = {}
    for s in all_sensors:
        v = df_u[s].values
        row[f'{s}_ns']    = norm_slope(v)
        row[f'{s}_delta'] = end_delta(v)

    row['T30_vs_Nf']  = norm_slope(df_u['sensor_3'].values)  - norm_slope(df_u['sensor_8'].values)
    row['NRc_vs_NRf'] = norm_slope(df_u['sensor_11'].values) - norm_slope(df_u['sensor_17'].values)
    row['T30_vs_T24'] = norm_slope(df_u['sensor_3'].values)  - norm_slope(df_u['sensor_2'].values)
    row['Nc_vs_Nf']   = norm_slope(df_u['sensor_9'].values)  - norm_slope(df_u['sensor_8'].values)
    row['hpc_score']  = (row['T30_vs_Nf'] + row['NRc_vs_NRf']
                         + row['T30_vs_T24'] + row['Nc_vs_Nf'])
    row['lifetime']   = n

    return pd.Series(row)

# ─────────────────────────────────────────
# APPLICATION FASTAPI
# ─────────────────────────────────────────
app = FastAPI(
    title="Détection de Pannes Moteur",
    description="Reçoit l'historique d'un moteur en JSON et prédit le type de panne (HPC vs Fan).",
    version="1.0"
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Serveur de détection de pannes opérationnel."}

@app.get("/health")
def health():
    """Vérifie que les modèles sont bien chargés."""
    if clf is None:
        return JSONResponse(status_code=503, content={"status": "error",
                            "detail": "Modèles non chargés. Lancez pipeline2.py d'abord."})
    return {"status": "ok", "modele": "rf_fault_classifier", "features": 20}

@app.post("/predict")
def predict(payload: MoteurHistorique):
    """
    Reçoit l'historique d'un moteur et retourne la prédiction de panne.

    Retourne :
    - fault_type     : 'HPC Degradation' ou 'Fan Degradation'
    - fault_id       : 0 (HPC) ou 1 (Fan)
    - confidence     : probabilité de la classe prédite (0.0 → 1.0)
    - prob_hpc       : probabilité HPC Degradation
    - prob_fan       : probabilité Fan Degradation
    - n_cycles       : nombre de cycles reçus
    - features       : valeurs des features extraites
    """
    if clf is None:
        raise HTTPException(status_code=503,
                            detail="Modèles non chargés. Lancez pipeline2.py d'abord.")

    # Vérification du nombre minimum de cycles
    n_cycles = len(payload.cycles)
    if n_cycles < 10:
        raise HTTPException(status_code=422,
                            detail=f"Historique trop court : {n_cycles} cycles reçus (minimum 10).")

    # Conversion en DataFrame
    df_u = pd.DataFrame([c.dict() for c in payload.cycles])
    df_u = df_u.sort_values('time_cycles').reset_index(drop=True)

    # Extraction des features
    features = extraire_features_moteur(df_u)
    feat_names = features.index.tolist()
    X = features.values.reshape(1, -1)

    # Imputation + normalisation
    X_imp   = imputer_clf.transform(X)
    X_scale = scaler_clf.transform(X_imp)

    # Prédiction
    fault_id   = int(clf.predict(X_scale)[0])
    proba      = clf.predict_proba(X_scale)[0]
    prob_hpc   = float(proba[0])
    prob_fan   = float(proba[1])
    confidence = float(proba[fault_id])
    fault_name = FAULT_NAMES[fault_id]

    # Log console
    print(f"[PREDICT] unit_id={payload.unit_id} | cycles={n_cycles} "
          f"| → {fault_name} ({confidence:.1%})")

    return {
        "unit_id":    payload.unit_id,
        "fault_type": fault_name,
        "fault_id":   fault_id,
        "confidence": round(confidence, 4),
        "prob_hpc":   round(prob_hpc, 4),
        "prob_fan":   round(prob_fan, 4),
        "n_cycles":   n_cycles,
        "features":   {k: round(float(v), 6) for k, v in zip(feat_names, features.values)}
    }

# ─────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)