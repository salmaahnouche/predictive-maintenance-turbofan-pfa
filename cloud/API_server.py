"""
api_server.py — FastAPI Cloud
================================
Reçoit le JSON de pc_edge.py, fusionne avec SQLite,
classifie le type de panne HPC/Fan, expose les routes
pour le dashboard Streamlit.

Lancer : uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from scipy.stats import linregress

from config import (
    ALL_SENSORS, DB_PATH,
    FAULT_NAMES, MODEL_CLF_PATH, MODEL_IMPUTER_PATH, MODEL_SCALER_PATH,
    RUL_CRITIQUE, RUL_DEGRADE, SENSOR_COLS, STATUT_MAP,
)

# ── Modèles globaux ───────────────────────────────────────────────────────────
clf_model:   Any = None
scaler_clf:  Any = None
imputer_clf: Any = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf_model, scaler_clf, imputer_clf
    ok = all(Path(p).exists() for p in [MODEL_CLF_PATH, MODEL_SCALER_PATH, MODEL_IMPUTER_PATH])
    if ok:
        try:
            clf_model   = joblib.load(MODEL_CLF_PATH)
            scaler_clf  = joblib.load(MODEL_SCALER_PATH)
            imputer_clf = joblib.load(MODEL_IMPUTER_PATH)
            print("✅ Modèles RF + Scaler + Imputer chargés.")
        except Exception as e:
            print(f"⚠️  Erreur chargement modèles : {e}")
    else:
        print("⚠️  Modèles .pkl introuvables — classification désactivée.")
    init_db()
    print("✅ SQLite prêt.")
    yield


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance — Cloud",
    description="Réception JSON Edge · Fusion SQLite · Détection HPC/Fan",
    version="5.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class EntreeMoteur(BaseModel):
    ligne:    int
    capteurs: Dict[str, float]   # time_cycles + capteurs (17 clés max)
    rul:      float
    statut:   str                # SAIN / DEGRADE / CRITIQUE


class PayloadEdge(BaseModel):
    session: str
    moteurs: Dict[str, List[EntreeMoteur]]


# ── Base de données ───────────────────────────────────────────────────────────
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _colonnes_existantes(conn: sqlite3.Connection, table: str) -> set:
    """Retourne l'ensemble des colonnes actuelles d'une table SQLite."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def init_db():
    """
    Crée les tables si elles n'existent pas.
    Si sensor_history existe mais avec un schéma obsolète (sans les bonnes
    colonnes), la table est supprimée et recréée automatiquement.
    Aucune donnée historique n'est perdue si le schéma est déjà correct.
    """
    cols_ddl = "\n".join(
        f"                {c:<14} REAL," for c in SENSOR_COLS
    )
    schema_sensor_history = f"""
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
            {cols_ddl}
            UNIQUE(unit_number, time_cycles)
        );"""

    schema_predictions = """
        CREATE TABLE IF NOT EXISTS predictions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at    TEXT    NOT NULL,
            session_id     TEXT    NOT NULL,
            unit_number    INTEGER NOT NULL,
            derniere_ligne INTEGER,
            rul_derniere   REAL,
            rul_moyen      REAL,
            rul_min        REAL,
            rul_max        REAL,
            fault_type     INTEGER,
            fault_name     TEXT,
            confidence     REAL,
            proba_hpc      REAL,
            proba_fan      REAL,
            hpc_score      REAL,
            alert_level    TEXT,
            n_cycles_recus INTEGER,
            n_cycles_total INTEGER,
            source         TEXT DEFAULT 'edge_python'
        );"""

    with get_conn() as conn:
        # ── Vérification du schéma sensor_history ────────────────────────────
        existing = _colonnes_existantes(conn, "sensor_history")
        expected = set(SENSOR_COLS) | {
            "id", "session_id", "received_at", "unit_number",
            "ligne", "time_cycles", "rul", "statut_edge", "alert_level",
        }
        if existing and not expected.issubset(existing):
            # Schéma obsolète détecté — drop + recréation
            missing = expected - existing
            extra   = existing - expected
            print(f"⚠️  Schéma sensor_history obsolète détecté.")
            print(f"   Colonnes manquantes : {missing}")
            print(f"   Colonnes inattendues: {extra}")
            print(f"   → Suppression et recréation de sensor_history...")
            conn.execute("DROP TABLE IF EXISTS sensor_history")
            conn.commit()
            print(f"   ✅ Table sensor_history supprimée.")

        conn.executescript(schema_sensor_history + schema_predictions)
        conn.commit()
        print(f"✅ Tables vérifiées — SENSOR_COLS={len(SENSOR_COLS)} colonnes capteurs.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _time_cycles(capteurs: Dict[str, float], ligne: int,
                 unit: int, conn: sqlite3.Connection) -> int:
    """
    Extrait time_cycles depuis le dict capteurs.
    Fallback robuste en 2 niveaux si absent.
    """
    if "time_cycles" in capteurs:
        return int(capteurs["time_cycles"])

    # Fallback 1 : dernier cycle connu en BDD + 1
    try:
        row = conn.execute(
            "SELECT MAX(time_cycles) FROM sensor_history WHERE unit_number = ?",
            (unit,)
        ).fetchone()
        if row and row[0] is not None:
            tc = int(row[0]) + 1
            print(f"  ℹ️  time_cycles absent moteur {unit} ligne {ligne} → BDD+1={tc}")
            return tc
    except Exception as exc:
        print(f"  ⚠️  Erreur BDD reconstruction time_cycles : {exc}")

    # Fallback 2 : numéro de ligne
    print(f"  ⚠️  time_cycles absent moteur {unit} ligne {ligne} → fallback={ligne+1}")
    return ligne + 1


def _sensor_values(capteurs: Dict[str, float]) -> List[Optional[float]]:
    """
    Retourne les valeurs des SENSOR_COLS dans l'ordre exact de config.py.
    None si une colonne est absente (stocké NULL en SQLite).
    setting_1 et setting_2 sont dans capteurs_dict mais pas dans SENSOR_COLS
    → ils sont simplement ignorés ici.
    """
    return [capteurs.get(col) for col in SENSOR_COLS]


def _alert_level(rul: float, statut: str = "") -> str:
    if rul <= RUL_CRITIQUE:
        return "CRITIQUE"
    if rul <= RUL_DEGRADE:
        return "AVERTISSEMENT"
    return STATUT_MAP.get(statut.upper(), "NORMAL")


# ── Feature engineering ───────────────────────────────────────────────────────
def _norm_slope(v: np.ndarray, t: np.ndarray) -> float:
    if len(v) < 2:
        return 0.0
    r = float(v.max()) - float(v.min())
    if r < 1e-6:
        return 0.0
    slope, *_ = linregress(t, v)
    return float(slope / r)


def _extraire_features(df: pd.DataFrame) -> dict:
    """
    Calcule les features de dégradation sur l'historique complet du moteur.
    N'utilise que les colonnes sensor_X — setting_1/setting_2 ne sont pas
    dans sensor_history et ne sont pas nécessaires ici.
    """
    df  = df.sort_values("time_cycles").reset_index(drop=True)
    t   = np.arange(len(df))
    n   = len(df)
    n20 = max(int(n * 0.2), 5)

    def safe(col: str) -> np.ndarray:
        return df[col].fillna(0.0).values.astype(float) if col in df.columns else np.zeros(n)

    def delta(v: np.ndarray) -> float:
        return float(v[-n20:].mean()) - float(v[:n20].mean())

    row: dict = {}
    for s in ALL_SENSORS:
        v = safe(s)
        row[f"{s}_ns"]    = _norm_slope(v, t)
        row[f"{s}_delta"] = delta(v)

    s2  = safe("sensor_2");  s3  = safe("sensor_3")
    s8  = safe("sensor_8");  s9  = safe("sensor_9")
    s11 = safe("sensor_11"); s17 = safe("sensor_17")

    row["T30_vs_Nf"]  = _norm_slope(s3,  t) - _norm_slope(s8,  t)
    row["NRc_vs_NRf"] = _norm_slope(s11, t) - _norm_slope(s17, t)
    row["T30_vs_T24"] = _norm_slope(s3,  t) - _norm_slope(s2,  t)
    row["Nc_vs_Nf"]   = _norm_slope(s9,  t) - _norm_slope(s8,  t)
    row["hpc_score"]  = sum(row[k] for k in ["T30_vs_Nf", "NRc_vs_NRf", "T30_vs_T24", "Nc_vs_Nf"])
    row["lifetime"]   = n
    return row


def _predict_fault(df_hist: pd.DataFrame) -> Optional[dict]:
    """
    Classifie le type de panne sur l'historique complet.
    Pipeline : features → imputer → scaler → Random Forest.
    """
    if any(m is None for m in [clf_model, scaler_clf, imputer_clf]):
        return None
    if len(df_hist) < 5:
        return None

    # Ordre exact des features attendu par l'imputer/scaler/clf
    try:
        feat_cols = imputer_clf.feature_names_in_.tolist()
    except AttributeError:
        if hasattr(clf_model, "feature_names_in_"):
            feat_cols = clf_model.feature_names_in_.tolist()
        else:
            return None

    feats = _extraire_features(df_hist)
    # NaN là où la feature est absente → l'imputer remplace par la médiane d'entraînement
    X_raw = np.array([[feats.get(c, np.nan) for c in feat_cols]])

    try:
        X_imp = imputer_clf.transform(X_raw)
        Xs    = scaler_clf.transform(X_imp)
        fid   = int(clf_model.predict(Xs)[0])
        prob  = clf_model.predict_proba(Xs)[0]
    except Exception as exc:
        print(f"⚠️  Erreur classification : {exc}")
        return None

    return {
        "fault_type": fid,
        "fault_name": FAULT_NAMES[fid],
        "confidence": float(prob.max()),
        "proba_hpc":  float(prob[0]),
        "proba_fan":  float(prob[1]),
        "hpc_score":  float(feats["hpc_score"]),
    }


# ── Route principale ──────────────────────────────────────────────────────────
@app.post("/api/maintenance/upload", tags=["Edge"])
async def recevoir_depuis_edge(payload: PayloadEdge):
    """
    Point d'entrée principal — reçoit le JSON de pc_edge.py.

    Pour chaque moteur :
      1. Insère les nouveaux cycles dans sensor_history.
      2. Recharge tout l'historique depuis la BDD (fusion complète).
      3. Classifie HPC ou Fan sur l'historique complet.
      4. Enregistre la prédiction dans predictions.
    """
    session_id  = payload.session
    received_at = datetime.now().isoformat()
    resultats   = []

    try:
        with get_conn() as conn:
            for unit_str, entrees in payload.moteurs.items():
                unit = int(unit_str)
                if not entrees:
                    continue

                # 1. Insertion des nouveaux cycles
                nouveaux     = 0
                placeholders = ", ".join(["?"] * (8 + len(SENSOR_COLS)))
                cols_str     = ", ".join(SENSOR_COLS)

                for e in entrees:
                    tc    = _time_cycles(e.capteurs, e.ligne, unit, conn)
                    alert = _alert_level(e.rul, e.statut)
                    vals  = _sensor_values(e.capteurs)

                    try:
                        conn.execute(
                            f"""INSERT OR IGNORE INTO sensor_history
                                (session_id, received_at, unit_number, ligne,
                                 time_cycles, rul, statut_edge, alert_level,
                                 {cols_str})
                                VALUES ({placeholders})""",
                            [session_id, received_at, unit, e.ligne,
                             tc, e.rul, e.statut, alert] + vals,
                        )
                        nouveaux += conn.execute("SELECT changes()").fetchone()[0]
                    except Exception as exc:
                        print(f"⚠️  Insertion ignorée moteur {unit} ligne {e.ligne}: {exc}")

                conn.commit()

                # 2. Fusion : historique complet du moteur
                df_hist = pd.read_sql_query(
                    f"""SELECT time_cycles, rul, {cols_str}
                        FROM sensor_history
                        WHERE unit_number = ?
                        ORDER BY time_cycles""",
                    conn, params=(unit,),
                )

                # 3. Classification HPC / Fan
                fault = _predict_fault(df_hist)

                ruls         = [e.rul for e in entrees]
                rul_derniere = ruls[-1]
                alert_finale = _alert_level(rul_derniere, entrees[-1].statut)

                # Affichage console
                print(f"\n{'='*50}")
                print(f"📢 Moteur #{unit} | lot: {len(entrees)} cycles | BDD total: {len(df_hist)}")
                if fault:
                    print(f"🧠 {fault['fault_name']} — confiance: {fault['confidence']:.1%}")
                else:
                    print("⚠️  Diagnostic en attente (< 5 cycles en historique)")
                print(f"📊 RUL: {rul_derniere:.1f} → {alert_finale}")
                print("="*50)

                # 4. Sauvegarder la prédiction
                conn.execute(
                    """INSERT INTO predictions
                       (received_at, session_id, unit_number, derniere_ligne,
                        rul_derniere, rul_moyen, rul_min, rul_max,
                        fault_type, fault_name, confidence, proba_hpc, proba_fan,
                        hpc_score, alert_level, n_cycles_recus, n_cycles_total, source)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        received_at, session_id, unit, entrees[-1].ligne,
                        rul_derniere,
                        round(float(np.mean(ruls)), 2),
                        round(float(np.min(ruls)),  2),
                        round(float(np.max(ruls)),  2),
                        fault["fault_type"] if fault else None,
                        fault["fault_name"] if fault else None,
                        fault["confidence"] if fault else None,
                        fault["proba_hpc"]  if fault else None,
                        fault["proba_fan"]  if fault else None,
                        fault["hpc_score"]  if fault else None,
                        alert_finale,
                        len(entrees),
                        len(df_hist),
                        "edge_python",
                    ),
                )
                conn.commit()

                resultats.append({
                    "unit_number":         unit,
                    "n_cycles_recus":      len(entrees),
                    "n_cycles_historique": len(df_hist),
                    "cycles_nouveaux":     nouveaux,
                    "rul_derniere":        round(rul_derniere, 2),
                    "rul_moyen":           round(float(np.mean(ruls)), 2),
                    "alert_level":         alert_finale,
                    "fault":               fault,
                })

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur traitement : {exc}")

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


# ── Routes de lecture ─────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "api": "Predictive Maintenance Cloud v5.3"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":     "ok",
        "db":         Path(DB_PATH).exists(),
        "clf_loaded": clf_model is not None,
        "timestamp":  datetime.now().isoformat(),
    }


@app.get("/predictions/latest", tags=["Data"])
def get_latest():
    """Dernière prédiction pour chaque moteur."""
    with get_conn() as conn:
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
    limit: int = Query(200, le=1000),
    alert_only: bool = Query(False),
):
    with get_conn() as conn:
        where, params = [], []
        if unit_number is not None:
            where.append("unit_number = ?")
            params.append(unit_number)
        if alert_only:
            where.append("alert_level IN ('CRITIQUE','AVERTISSEMENT')")
        q = "SELECT * FROM predictions"
        if where:
            q += " WHERE " + " AND ".join(where)
        q += f" ORDER BY received_at DESC LIMIT {limit}"
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/predictions/history/{unit_number}", tags=["Data"])
def get_unit_rul_history(unit_number: int):
    """Historique RUL cycle par cycle pour un moteur."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT time_cycles, rul, statut_edge, alert_level, session_id
               FROM sensor_history WHERE unit_number = ? ORDER BY time_cycles""",
            (unit_number,),
        ).fetchall()
    if not rows:
        raise HTTPException(404, f"Moteur {unit_number} introuvable")
    return [dict(r) for r in rows]


@app.get("/sensors/history/{unit_number}", tags=["Data"])
def get_sensor_history(unit_number: int):
    """Historique complet des capteurs pour un moteur."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sensor_history WHERE unit_number = ? ORDER BY time_cycles",
            (unit_number,),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/stats/summary", tags=["Data"])
def get_summary():
    """KPIs globaux : totaux, alertes, pannes, RUL moyen."""
    with get_conn() as conn:
        latest = "SELECT MAX(received_at) FROM predictions GROUP BY unit_number"
        total  = conn.execute(
            "SELECT COUNT(DISTINCT unit_number) FROM predictions"
        ).fetchone()[0]
        crit   = conn.execute(
            f"SELECT COUNT(*) FROM predictions WHERE alert_level='CRITIQUE' "
            f"AND received_at IN ({latest})"
        ).fetchone()[0]
        avert  = conn.execute(
            f"SELECT COUNT(*) FROM predictions WHERE alert_level='AVERTISSEMENT' "
            f"AND received_at IN ({latest})"
        ).fetchone()[0]
        fdist  = conn.execute("""
            SELECT fault_name, COUNT(*) n FROM (
                SELECT unit_number, fault_name FROM predictions
                WHERE fault_name IS NOT NULL GROUP BY unit_number
            ) GROUP BY fault_name
        """).fetchall()
        avg = conn.execute(
            f"SELECT AVG(rul_derniere) FROM predictions WHERE received_at IN ({latest})"
        ).fetchone()[0]
    return {
        "total_units":        total,
        "critiques":          crit,
        "avertissements":     avert,
        "normaux":            max(0, total - crit - avert),
        "avg_rul":            round(avg, 1) if avg else None,
        "fault_distribution": [dict(r) for r in fdist],
    }


@app.get("/sessions", tags=["Data"])
def get_sessions():
    with get_conn() as conn:
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


@app.get("/export/csv", tags=["Export"])
def export_csv():
    """Télécharge toutes les prédictions au format CSV."""
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY received_at DESC", conn
        )
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.delete("/reset", tags=["Admin"])
def reset_db(token: str = Query(...)):
    """Vide toutes les tables. Protégé par RESET_TOKEN (variable d'environnement)."""
    expected = os.environ.get("RESET_TOKEN", "")
    if not expected or token != expected:
        raise HTTPException(403, "Token invalide ou absent.")
    with get_conn() as conn:
        conn.executescript("DELETE FROM predictions; DELETE FROM sensor_history;")
        conn.commit()
    return {"status": "ok", "message": "Base de données vidée."}