"""
api_server.py  —  FastAPI PC Cloud
====================================
Reçoit le JSON du PC Edge (avec capteurs dans chaque entrée moteur),
fusionne avec SQLite, prédit le type de panne HPC/Fan.

Format JSON reçu depuis pc_edge.py :
{
  "session": "2025-05-20 10:30:00",
  "moteurs": {
    "1": [
      {
        "ligne": 0,
        "capteurs": {"sensor_2": 0.45, "sensor_3": 0.62, "time_cycles": 1, ...},
        "rul": 87.3,
        "statut": "NORMAL"
      },
      ...
    ]
  }
}

Lancer : uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.stats import linregress

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DB_PATH           = "historique.db"
MODEL_CLF_PATH    = "rf_fault_classifier.pkl"
MODEL_SCALER_PATH = "scaler_fault_classifier.pkl"

HPC_SENSORS = ['sensor_3', 'sensor_11', 'sensor_9']
FAN_SENSORS  = ['sensor_8', 'sensor_17', 'sensor_2', 'sensor_13']
ALL_SENSORS  = list(set(HPC_SENSORS + FAN_SENSORS))
FAULT_NAMES  = {0: 'HPC Degradation', 1: 'Fan Degradation'}

ALERT_CRITICAL = 30
ALERT_WARNING  = 60

# Colonnes capteurs stockées en BDD (hors time_cycles qui est une colonne dédiée)
SENSOR_COLS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]

STATUT_MAP = {
    "NORMAL":        "NORMAL",
    "WARNING":       "AVERTISSEMENT",
    "CRITICAL":      "CRITIQUE",
    "AVERTISSEMENT": "AVERTISSEMENT",
    "CRITIQUE":      "CRITIQUE",
}

# ─────────────────────────────────────────
# PYDANTIC — Format exact envoyé par pc_edge.py
# ─────────────────────────────────────────
class EntreeMoteur(BaseModel):
    ligne:    int
    capteurs: Dict[str, float]   # contient sensor_X + éventuellement time_cycles, setting_X...
    rul:      float
    statut:   str

class PayloadEdge(BaseModel):
    session: str
    moteurs: Dict[str, List[EntreeMoteur]]

# ─────────────────────────────────────────
# MODÈLES GLOBAUX
# ─────────────────────────────────────────
clf_model, scaler_clf = None, None

# ─────────────────────────────────────────
# LIFESPAN (remplace @app.on_event déprécié)
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf_model, scaler_clf
    # — Startup —
    if Path(MODEL_CLF_PATH).exists() and Path(MODEL_SCALER_PATH).exists():
        clf_model  = joblib.load(MODEL_CLF_PATH)
        scaler_clf = joblib.load(MODEL_SCALER_PATH)
        print("✅ Modèles RF chargés.")
    else:
        print("⚠️  Modèles introuvables — classification désactivée.")
    init_db()
    print("✅ SQLite prêt.")
    yield
    # — Shutdown — (rien à libérer ici)

# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance — PC Cloud",
    description="Réception JSON Edge, fusion SQLite, détection HPC/Fan",
    version="4.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────
# BASE DE DONNÉES
# ─────────────────────────────────────────
@contextmanager
def get_conn_ctx():
    """Gestionnaire de contexte — garantit la fermeture même en cas d'exception."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_conn_ctx() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sensor_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                received_at TEXT    NOT NULL,
                unit_number INTEGER NOT NULL,
                ligne       INTEGER NOT NULL,
                time_cycles INTEGER NOT NULL,
                rul         REAL,
                statut_edge TEXT,
                alert_level TEXT,
                sensor_2    REAL, sensor_3  REAL, sensor_4  REAL,
                sensor_7    REAL, sensor_8  REAL, sensor_9  REAL,
                sensor_11   REAL, sensor_12 REAL, sensor_13 REAL,
                sensor_14   REAL, sensor_15 REAL, sensor_17 REAL,
                sensor_20   REAL, sensor_21 REAL,
                UNIQUE(unit_number, time_cycles)
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at     TEXT    NOT NULL,
                session_id      TEXT    NOT NULL,
                unit_number     INTEGER NOT NULL,
                derniere_ligne  INTEGER,
                rul_derniere    REAL,
                rul_moyen       REAL,
                rul_min         REAL,
                rul_max         REAL,
                fault_type      INTEGER,
                fault_name      TEXT,
                confidence      REAL,
                proba_hpc       REAL,
                proba_fan       REAL,
                hpc_score       REAL,
                alert_level     TEXT,
                n_cycles_recus  INTEGER,
                n_cycles_total  INTEGER,
                source          TEXT DEFAULT 'edge_python'
            );
        """)
        conn.commit()

# ─────────────────────────────────────────
# HELPERS — extraction depuis le dict capteurs Edge
# ─────────────────────────────────────────
def extraire_time_cycles(capteurs: Dict[str, float], ligne: int) -> int:
    """
    pc_edge.py met time_cycles dans le dict capteurs (via df.iloc[i].to_dict()).
    On l'extrait ici ; si absent, on utilise le numéro de ligne comme fallback.
    """
    if 'time_cycles' in capteurs:
        return int(capteurs['time_cycles'])
    # Fallback : ligne 0-indexée → cycle 1-indexé
    print(f"⚠️  time_cycles absent des capteurs pour ligne {ligne} — utilisation du fallback.")
    return ligne + 1

def extraire_valeurs_sensors(capteurs: Dict[str, float]) -> List[Optional[float]]:
    """Retourne les valeurs des SENSOR_COLS dans l'ordre, None si absent."""
    return [capteurs.get(s, None) for s in SENSOR_COLS]

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def _norm_slope(v: np.ndarray, t: np.ndarray) -> float:
    if len(v) < 2:
        return 0.0
    r = float(v.max()) - float(v.min())
    if r < 1e-6:
        return 0.0
    slope, *_ = linregress(t, v)
    return float(slope / r)

def extraire_features_capteurs(df: pd.DataFrame) -> dict:
    df  = df.sort_values('time_cycles').reset_index(drop=True)
    t   = np.arange(len(df))
    n   = len(df)
    n20 = max(int(n * 0.2), 5)

    def safe(name):
        if name in df.columns:
            return df[name].fillna(0.0).values.astype(float)
        return np.zeros(n)

    def end_delta(v):
        return float(v[-n20:].mean()) - float(v[:n20].mean())

    row = {}
    for s in ALL_SENSORS:
        v = safe(s)
        row[f'{s}_ns']    = _norm_slope(v, t)
        row[f'{s}_delta'] = end_delta(v)

    s2  = safe('sensor_2');  s3  = safe('sensor_3')
    s8  = safe('sensor_8');  s9  = safe('sensor_9')
    s11 = safe('sensor_11'); s17 = safe('sensor_17')

    row['T30_vs_Nf']  = _norm_slope(s3,  t) - _norm_slope(s8,  t)
    row['NRc_vs_NRf'] = _norm_slope(s11, t) - _norm_slope(s17, t)
    row['T30_vs_T24'] = _norm_slope(s3,  t) - _norm_slope(s2,  t)
    row['Nc_vs_Nf']   = _norm_slope(s9,  t) - _norm_slope(s8,  t)
    row['hpc_score']  = (row['T30_vs_Nf'] + row['NRc_vs_NRf']
                         + row['T30_vs_T24'] + row['Nc_vs_Nf'])
    row['lifetime']   = n
    return row

def predire_fault(df_historique: pd.DataFrame):
    if clf_model is None or scaler_clf is None:
        return None
    if len(df_historique) < 5:
        return None

    feat_cols = (clf_model.feature_names_in_.tolist()
                 if hasattr(clf_model, 'feature_names_in_') else [])
    if not feat_cols:
        return None

    feats = extraire_features_capteurs(df_historique)
    X     = np.array([[feats.get(c, 0.0) for c in feat_cols]])
    try:
        Xs   = scaler_clf.transform(X)
        fid  = int(clf_model.predict(Xs)[0])
        prob = clf_model.predict_proba(Xs)[0]
    except Exception as e:
        print(f"Erreur classification : {e}")
        return None

    return {
        'fault_type': fid,
        'fault_name': FAULT_NAMES[fid],
        'confidence': float(prob.max()),
        'proba_hpc':  float(prob[0]),
        'proba_fan':  float(prob[1]),
        'hpc_score':  float(feats['hpc_score']),
    }

def normaliser_alerte(rul: float, statut: str = "") -> str:
    if rul <= ALERT_CRITICAL:
        return "CRITIQUE"
    if rul <= ALERT_WARNING:
        return "AVERTISSEMENT"
    return STATUT_MAP.get(statut.upper(), "NORMAL")

# ─────────────────────────────────────────
# ROUTE PRINCIPALE
# ─────────────────────────────────────────
@app.post("/api/maintenance/upload", tags=["Edge"])
async def recevoir_depuis_edge(payload: PayloadEdge):
    """
    Reçoit le JSON complet du PC Edge (pc_edge.py).
    Les valeurs capteurs + time_cycles sont dans le dict 'capteurs' de chaque entrée.
    """
    session_id  = payload.session
    received_at = datetime.now().isoformat()
    resultats   = []

    try:
        with get_conn_ctx() as conn:
            for unit_str, entrees in payload.moteurs.items():
                unit = int(unit_str)

                if not entrees:
                    continue

                # ── 1. Insérer les cycles dans sensor_history ──
                nouveaux = 0
                for e in entrees:
                    # Extraire time_cycles depuis le dict capteurs
                    time_cycles = extraire_time_cycles(e.capteurs, e.ligne)
                    alert       = normaliser_alerte(e.rul, e.statut)
                    vals        = extraire_valeurs_sensors(e.capteurs)

                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO sensor_history
                               (session_id, received_at, unit_number, ligne,
                                time_cycles, rul, statut_edge, alert_level,
                                sensor_2, sensor_3, sensor_4, sensor_7, sensor_8,
                                sensor_9, sensor_11, sensor_12, sensor_13, sensor_14,
                                sensor_15, sensor_17, sensor_20, sensor_21)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            [session_id, received_at, unit, e.ligne,
                             time_cycles, e.rul, e.statut, alert] + vals
                        )
                        nouveaux += conn.execute("SELECT changes()").fetchone()[0]
                    except Exception as ex:
                        print(f"  Insertion ignorée moteur {unit} ligne {e.ligne}: {ex}")

                conn.commit()

                # ── 2. Fusion : tout l'historique capteurs du moteur ──
                df_historique = pd.read_sql_query(
                    """SELECT time_cycles, rul,
                              sensor_2, sensor_3, sensor_4, sensor_7, sensor_8,
                              sensor_9, sensor_11, sensor_12, sensor_13, sensor_14,
                              sensor_15, sensor_17, sensor_20, sensor_21
                       FROM sensor_history
                       WHERE unit_number = ?
                       ORDER BY time_cycles""",
                    conn, params=(unit,)
                )

                # ── 3. Classification HPC/Fan sur les vrais capteurs ──
                fault_result = predire_fault(df_historique)

                # Stats RUL session courante
                ruls         = [e.rul for e in entrees]
                rul_derniere = ruls[-1]
                alert_finale = normaliser_alerte(rul_derniere, entrees[-1].statut)

                # ── LIGNES DE CONTRÔLE POUR LA SOUTENANCE (Affichage Console) ──
                print("\n" + "=" * 50)
                print(f"📢 [CLOUD DETECT] Réception des données du Moteur n°{unit}")
                print(f"📦 Cycles reçus dans ce lot : {len(entrees)}")
                print(f"🔄 Fusionnement réussi ! Cycles totaux stockés en BDD : {len(df_historique)}")
                if fault_result:
                    print(f"🧠 Verdict de l'IA (Random Forest) : {fault_result['fault_name']}")
                    print(f"🎯 Score de confiance : {fault_result['confidence'] * 100:.2f}%")
                else:
                    print("⚠️  Diagnostic impossible : Pas assez de cycles en historique (minimum 5 requis).")
                print("=" * 50 + "\n")

                # ── 4. Sauvegarder la prédiction ──
                conn.execute(
                    """INSERT INTO predictions
                       (received_at, session_id, unit_number, derniere_ligne,
                        rul_derniere, rul_moyen, rul_min, rul_max,
                        fault_type, fault_name, confidence, proba_hpc, proba_fan,
                        hpc_score, alert_level, n_cycles_recus, n_cycles_total, source)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        received_at, session_id, unit,
                        entrees[-1].ligne,
                        rul_derniere,
                        round(float(np.mean(ruls)), 2),
                        round(float(np.min(ruls)),  2),
                        round(float(np.max(ruls)),  2),
                        fault_result['fault_type'] if fault_result else None,
                        fault_result['fault_name'] if fault_result else None,
                        fault_result['confidence'] if fault_result else None,
                        fault_result['proba_hpc']  if fault_result else None,
                        fault_result['proba_fan']  if fault_result else None,
                        fault_result['hpc_score']  if fault_result else None,
                        alert_finale,
                        len(entrees),
                        len(df_historique),
                        'edge_python'
                    )
                )
                conn.commit()

                resultats.append({
                    "unit_number":         unit,
                    "n_cycles_recus":      len(entrees),
                    "n_cycles_historique": len(df_historique),
                    "cycles_nouveaux":     nouveaux,
                    "rul_derniere":        round(rul_derniere, 2),
                    "rul_moyen":           round(float(np.mean(ruls)), 2),
                    "alert_level":         alert_finale,
                    "fault":               fault_result,
                })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traitement : {e}")

    n_crit  = sum(1 for r in resultats if r["alert_level"] == "CRITIQUE")
    n_avert = sum(1 for r in resultats if r["alert_level"] == "AVERTISSEMENT")

    return JSONResponse(content={
        "status":          "ok",
        "session":         session_id,
        "timestamp":       received_at,
        "moteurs_traites": len(resultats),
        "alertes": {
            "critiques":      n_crit,
            "avertissements": n_avert,
            "normaux":        len(resultats) - n_crit - n_avert,
        },
        "resultats": resultats,
    })

# ─────────────────────────────────────────
# ROUTES LECTURE — Dashboard Streamlit
# ─────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "api": "Predictive Maintenance Cloud v4.2"}

@app.get("/health", tags=["Health"])
def health():
    return {
        "status":     "ok",
        "db":         Path(DB_PATH).exists(),
        "clf_loaded": clf_model is not None,
        "timestamp":  datetime.now().isoformat()
    }

@app.get("/predictions/latest", tags=["Data"])
def get_latest():
    with get_conn_ctx() as conn:
        rows = conn.execute("""
            SELECT p.* FROM predictions p
            INNER JOIN (
                SELECT unit_number, MAX(received_at) AS max_at
                FROM predictions GROUP BY unit_number
            ) l ON p.unit_number = l.unit_number AND p.received_at = l.max_at
            ORDER BY p.unit_number
        """).fetchall()
    return [dict(r) for r in rows]

@app.get("/predictions", tags=["Data"])
def get_predictions(
    unit_number: Optional[int] = Query(None),
    limit: int = Query(200),
    alert_only: bool = Query(False)
):
    with get_conn_ctx() as conn:
        where, params = [], []
        if unit_number:
            where.append("unit_number = ?"); params.append(unit_number)
        if alert_only:
            where.append("alert_level IN ('CRITIQUE','AVERTISSEMENT')")
        q = "SELECT * FROM predictions"
        if where:
            q += " WHERE " + " AND ".join(where)
        q += f" ORDER BY received_at DESC LIMIT {limit}"
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]

@app.get("/predictions/history/{unit_number}", tags=["Data"])
def get_unit_history(unit_number: int):
    with get_conn_ctx() as conn:
        rows = conn.execute(
            """SELECT time_cycles, rul, statut_edge, alert_level, session_id
               FROM sensor_history WHERE unit_number = ? ORDER BY time_cycles""",
            (unit_number,)
        ).fetchall()
    if not rows:
        raise HTTPException(404, f"Moteur {unit_number} introuvable")
    return [dict(r) for r in rows]

@app.get("/sensors/history/{unit_number}", tags=["Data"])
def get_sensor_history(unit_number: int):
    with get_conn_ctx() as conn:
        rows = conn.execute(
            "SELECT * FROM sensor_history WHERE unit_number = ? ORDER BY time_cycles",
            (unit_number,)
        ).fetchall()
    return [dict(r) for r in rows]

@app.get("/stats/summary", tags=["Data"])
def get_summary():
    with get_conn_ctx() as conn:
        sub   = "SELECT MAX(received_at) FROM predictions GROUP BY unit_number"
        total = conn.execute("SELECT COUNT(DISTINCT unit_number) FROM predictions").fetchone()[0]
        crit  = conn.execute(
            f"SELECT COUNT(*) FROM predictions WHERE alert_level='CRITIQUE' AND received_at IN ({sub})"
        ).fetchone()[0]
        avert = conn.execute(
            f"SELECT COUNT(*) FROM predictions WHERE alert_level='AVERTISSEMENT' AND received_at IN ({sub})"
        ).fetchone()[0]
        fdist = conn.execute("""
            SELECT fault_name, COUNT(*) n FROM (
                SELECT unit_number, fault_name FROM predictions
                WHERE fault_name IS NOT NULL GROUP BY unit_number
            ) GROUP BY fault_name
        """).fetchall()
        avg = conn.execute(
            f"SELECT AVG(rul_derniere) FROM predictions WHERE received_at IN ({sub})"
        ).fetchone()[0]
    return {
        "total_units":        total,
        "critiques":          crit,
        "avertissements":     avert,
        "normaux":            max(0, total - crit - avert),
        "avg_rul":            round(avg, 1) if avg else None,
        "fault_distribution": [dict(r) for r in fdist]
    }

@app.get("/sessions", tags=["Data"])
def get_sessions():
    with get_conn_ctx() as conn:
        rows = conn.execute("""
            SELECT session_id,
                   MIN(received_at)            AS premiere,
                   COUNT(DISTINCT unit_number) AS nb_moteurs,
                   AVG(rul_derniere)           AS rul_moyen,
                   SUM(n_cycles_recus)         AS total_cycles
            FROM predictions
            GROUP BY session_id
            ORDER BY premiere DESC
        """).fetchall()
    return [dict(r) for r in rows]

@app.delete("/reset", tags=["Admin"])
def reset_db(token: str = Query(..., description="Jeton d'administration requis")):
    """⚠️  Vide toutes les tables. Protégé par token."""
    import os
    expected = os.environ.get("RESET_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="Token invalide ou absent.")
    with get_conn_ctx() as conn:
        conn.executescript("DELETE FROM predictions; DELETE FROM sensor_history;")
        conn.commit()
    return {"status": "ok", "message": "Base vidée."}