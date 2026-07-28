"""
train_pipeline.py — Pipeline d'entraînement Cloud
===================================================
Pipeline complet sur les 97 moteurs historiques FD003 :
  1. Chargement robuste des CSV
  2. Calcul + écrêtage du RUL (plafond souple 1.5× médiane)
  3. Normalisation MinMax des features (partagée train/test)
  4. Comparaison LR / RF / CNN / LSTM
  5. Export du CNN en TFLite INT8 + cerveau_cnn.h pour l'Arduino
  6. Détection de pannes HPC/Fan (GMM non supervisé + Random Forest)
  7. Sauvegarde des artefacts .pkl

CORRECTIONS v3 — Suppression normalisation RUL :
  [R1] Clip RUL : plafond souple à 1.5× la durée médiane.
       Le clip_value est sauvegardé pour cohérence Arduino / api_server.

  [R2] RUL brut : le CNN apprend directement les valeurs en cycles (0–327).
       Sortie Dense(1, activation='linear') — aucune normalisation/dénorm.
       Avantage : cohérence totale entre entraînement et inférence Arduino,
       pas de constante RUL_MAX à maintenir dans le sketch.

  [R3] Architecture CNN : deux blocs Conv1D + BatchNorm + Dropout,
       GlobalAveragePooling1D, Dense(64)+Dense(32), sortie linéaire.
       Clamp à 0 post-inférence (empêche les RUL négatifs).

  [R4] Évaluation cohérente : preparer_test_3d prend les TIME_STEPS
       derniers cycles du fichier test (condition réelle Arduino).

  [R5] Loss asymétrique : avec ASYM_ALPHA=1.0 → MSE pure.
       Fonctionne directement sur les cycles bruts.

Usage :
    python train_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import (
    BatchNormalization, Conv1D, Dense, Dropout,
    Flatten, GlobalAveragePooling1D, LSTM, MaxPooling1D,
)

from config import (
    ALL_SENSORS, ASYM_ALPHA, CLASSE_COLORS, CLASSE_NOMS,
    CNN_BATCH, CNN_EPOCHS, DROP_COLS_TRAIN, ESP32_FEATURES,
    FAULT_COLORS, FAULT_NAMES, HPC_SENSORS, FAN_SENSORS,
    RUL_SEUILS, TIME_STEPS,
)

# ── Chemins des données ────────────────────────────────────────────────────────
CHEMIN_TRAIN = r"C:\Users\HP\Desktop\pfa_proj\train_FD003_historique.csv"
CHEMIN_TEST  = r"C:\Users\HP\Desktop\pfa_proj\test_FD003_historique.csv"
CHEMIN_RUL   = r"C:\Users\HP\Desktop\pfa_proj\RUL_FD003_historique.csv"

# ── Colonnes du dataset CMAPSS (26 colonnes) ──────────────────────────────────
CMAPSS_COLS = (
    ["unit_number", "time_cycles", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ROBUSTE
# ═════════════════════════════════════════════════════════════════════════════
def charger_csv(path: str, cols: list = CMAPSS_COLS) -> pd.DataFrame:
    """Détecte automatiquement le séparateur et la présence d'un header."""
    with open(path, "r") as f:
        first_line = f.readline().strip()

    sep = "," if first_line.count(",") > first_line.count(" ") else r"\s+"
    try:
        float(first_line.split("," if sep == "," else None)[0])
        has_header = False
    except ValueError:
        has_header = True

    df = pd.read_csv(
        path, sep=sep,
        header=0 if has_header else None,
        names=None if has_header else cols,
        engine="python",
    )
    if has_header and df.shape[1] == len(cols):
        df.columns = cols

    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    nan_total = df.isna().sum().sum()
    if nan_total > 0:
        print(f"    ⚠️  {nan_total} NaN détectés dans {path}")
    return df


print("=" * 60)
print("  Chargement des données...")
print("=" * 60)
df_train = charger_csv(CHEMIN_TRAIN)
df_test  = charger_csv(CHEMIN_TEST)
df_rul   = pd.read_csv(CHEMIN_RUL, header=None, names=["true_rul"])
print(f"    Train : {df_train.shape[0]} lignes, {df_train['unit_number'].nunique()} moteurs")
print(f"    Test  : {df_test.shape[0]}  lignes, {df_test['unit_number'].nunique()} moteurs")


# ═════════════════════════════════════════════════════════════════════════════
# 2. CALCUL + ÉCRÊTAGE DU RUL
# ═════════════════════════════════════════════════════════════════════════════
def calculer_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    return df.drop(columns=["max_cycle"])

df_train = calculer_rul(df_train)

# [R1] Clip souple : 1.5× la durée médiane des moteurs d'entraînement.
durees_train = df_train.groupby("unit_number")["time_cycles"].max()
clip_value   = int(np.ceil(durees_train.median() * 1.5))
print(f"\n    Durée médiane train  : {durees_train.median():.0f} cycles")
print(f"    Durée max train      : {durees_train.max()} cycles")
print(f"    Clip RUL (×1.5 med)  : {clip_value} cycles")

# [R2] RUL brut directement — pas de normalisation
df_train["RUL"] = df_train["RUL"].clip(upper=clip_value)

# Sauvegarde du clip_value pour cohérence Arduino / api_server
with open("rul_clip_value.txt", "w") as f:
    f.write(str(clip_value))
print(f"    rul_clip_value.txt sauvegardé ({clip_value} cycles)")


# ═════════════════════════════════════════════════════════════════════════════
# 3. NORMALISATION DES FEATURES (partagée train / test)
# ═════════════════════════════════════════════════════════════════════════════
def pretraitement(df_train, df_test, chemin_scaler="scaler_cloud.pkl"):
    features = [c for c in df_train.columns if c not in DROP_COLS_TRAIN + ["RUL"]]
    assert set(features) == set(ESP32_FEATURES), (
        f"Mismatch features !\n"
        f"  Train  : {sorted(features)}\n"
        f"  Config : {sorted(ESP32_FEATURES)}"
    )
    scaler = MinMaxScaler()
    df_train[features] = scaler.fit_transform(df_train[features])
    df_test[features]  = scaler.transform(df_test[features])
    joblib.dump(scaler, chemin_scaler)
    print(f"    Scaler features sauvegardé → {chemin_scaler}")
    print(f"    Features normalisées : {len(features)} colonnes")
    return df_train, df_test, features

print("\n[1] Normalisation des features...")
df_train, df_test, features = pretraitement(df_train, df_test)


# ═════════════════════════════════════════════════════════════════════════════
# 4. PRÉPARATION DES DONNÉES
# ═════════════════════════════════════════════════════════════════════════════
def preparer_test_2d(df_test, df_rul, features):
    """Dernière ligne de chaque moteur test → prédiction ponctuelle."""
    last = df_test.groupby("unit_number").last().reset_index()
    last = last.sort_values("unit_number").reset_index(drop=True)
    return last[features].values, df_rul["true_rul"].values


def creer_sequences_3d(df, features, time_steps):
    """
    [R2] Séquences CNN avec RUL brut en cible.
    Chaque séquence = TIME_STEPS cycles consécutifs.
    La cible est le RUL brut (cycles) au dernier pas de temps.
    """
    X, y = [], []
    for unit in df["unit_number"].unique():
        df_u = df[df["unit_number"] == unit]
        data = df_u[features].values
        ruls = df_u["RUL"].values
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(ruls[i + time_steps])   # RUL brut en cycles
    return np.array(X), np.array(y)


def preparer_test_3d(df_test, df_rul, features, time_steps):
    """
    [R4] TIME_STEPS derniers cycles disponibles dans le fichier test.
    Padding avec zéros si moins de TIME_STEPS cycles (comme l'Arduino en début de vie).
    La cible est le vrai RUL brut (cycles).
    """
    X, y = [], []
    for i, unit in enumerate(sorted(df_test["unit_number"].unique())):
        df_u = df_test[df_test["unit_number"] == unit]
        data = df_u[features].values
        if len(data) >= time_steps:
            seq = data[-time_steps:]
        else:
            pad = np.zeros((time_steps - len(data), len(features)))
            seq = np.vstack([pad, data])
        X.append(seq)
        y.append(df_rul["true_rul"].iloc[i])
    return np.array(X), np.array(y)


# Données 2D pour LR et RF
X_train_2D = df_train[features].values
y_train_2D = df_train["RUL"].values
X_test_2D, y_test_2D = preparer_test_2d(df_test, df_rul, features)

# [R2] Séquences CNN — cible = RUL brut en cycles
X_train_3D, y_train_3D = creer_sequences_3d(df_train, features, TIME_STEPS)
X_test_3D,  y_test_3D  = preparer_test_3d(df_test, df_rul, features, TIME_STEPS)

print(f"\n    Train 2D : {X_train_2D.shape} | Test 2D : {X_test_2D.shape}")
print(f"    Train 3D : {X_train_3D.shape} | Test 3D : {X_test_3D.shape}")
print(f"    y_train_3D brut : min={y_train_3D.min():.0f}  max={y_train_3D.max():.0f} cycles")
print(f"    y_test_3D  brut : min={y_test_3D.min():.0f}  max={y_test_3D.max():.0f} cycles")


# ═════════════════════════════════════════════════════════════════════════════
# 5. LOSS ASYMÉTRIQUE (opère sur RUL brut en cycles)
# ═════════════════════════════════════════════════════════════════════════════
def asym_loss(y_true, y_pred):
    """
    Pénalise davantage la surestimation du RUL.
    Avec ASYM_ALPHA=1.0 → MSE pure (symétrique).
    Augmenter alpha pour un biais conservateur (sécurité industrielle).
    """
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    d = y_pred - y_true
    perte = tf.where(d > 0,
                     ASYM_ALPHA * tf.square(d),
                     tf.square(d))
    return tf.reduce_mean(perte, axis=-1)


# ═════════════════════════════════════════════════════════════════════════════
# 6. MODÈLES
# ═════════════════════════════════════════════════════════════════════════════
def rul_to_classe(rul_array, seuils=RUL_SEUILS):
    classes = np.full(len(rul_array), 2, dtype=int)
    classes[rul_array <= seuils["degrade"]]  = 1
    classes[rul_array <= seuils["critique"]] = 0
    return classes


def evaluer(modele, X_test, y_test_rul_brut, nom_modele, clip_max=None):
    """
    [R5] Toutes les métriques calculées sur les RUL bruts (cycles).
    Clamp à 0 pour éviter les prédictions négatives (sortie linéaire).
    """
    y_pred = modele.predict(X_test).flatten()
    # Clamp : pas de RUL négatif, pas de RUL absurde
    y_pred = np.clip(y_pred, 0, clip_max if clip_max else y_pred.max() * 2)

    y_true_cls = rul_to_classe(y_test_rul_brut)
    y_pred_cls = rul_to_classe(y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_rul_brut, y_pred))
    mae  = np.mean(np.abs(y_test_rul_brut - y_pred))
    acc  = (y_true_cls == y_pred_cls).mean()

    print(f"\n{'='*55}")
    print(f"  [{nom_modele}]  RMSE={rmse:.2f}  MAE={mae:.2f}  Acc={acc:.1%}")
    print(f"    RUL prédit : min={y_pred.min():.1f}  max={y_pred.max():.1f}  moy={y_pred.mean():.1f}")
    print(classification_report(y_true_cls, y_pred_cls,
          target_names=CLASSE_NOMS, labels=[0,1,2], zero_division=0))
    return {
        "Modèle": nom_modele, "RMSE": round(rmse,2), "MAE": round(mae,2),
        "Acc_cls": round(acc,3), "y_true_cls": y_true_cls,
        "y_pred_cls": y_pred_cls, "y_pred_rul": y_pred,
        "y_true_rul": y_test_rul_brut,
    }


def entrainer_cnn(X_train, y_train, epochs=CNN_EPOCHS, batch_size=CNN_BATCH):
    """
    [R2][R3] CNN entraîné directement sur RUL brut (cycles).
    Sortie Dense(1, activation='linear') — prédit en cycles.
    Clamp post-inférence à [0, clip_value] dans evaluer().

    Architecture :
    - 2 blocs Conv1D (64 puis 128 filtres) + BatchNorm + Dropout(0.2)
    - GlobalAveragePooling1D
    - Dense(64) + Dropout(0.2) + Dense(32) + Dense(1, linear)
    """
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    ts, nb = X_train.shape[1], X_train.shape[2]
    model = Sequential([
        # Bloc 1
        Conv1D(64, kernel_size=3, activation="relu", padding="causal",
               input_shape=(ts, nb)),
        BatchNormalization(),
        Dropout(0.2),
        # Bloc 2
        Conv1D(128, kernel_size=3, activation="relu", padding="causal"),
        BatchNormalization(),
        Dropout(0.2),
        # Pooling global
        GlobalAveragePooling1D(),
        # Têtes denses
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        # [R2] Sortie linéaire : prédiction directe en cycles
        Dense(1, activation="linear"),
    ])
    model.compile(loss=asym_loss, optimizer="adam")
    model.summary()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=0),
    ]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, verbose=1, callbacks=callbacks)
    return model

    import matplotlib.pyplot as plt
hist = model.history.history
plt.figure(figsize=(8,4))
plt.plot(hist["loss"], label="Train loss")
plt.plot(hist["val_loss"], label="Val loss")
plt.axvline(len(hist["loss"])-1, color="red", 
            linestyle="--", label="Early stopping")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Courbe d'entraînement CNN — FD003")
plt.legend()
plt.tight_layout()
plt.savefig("cnn_training_curve.png", dpi=150)


def entrainer_lstm(X_train, y_train, epochs=CNN_EPOCHS, batch_size=CNN_BATCH):
    """LSTM de référence — entraîné sur RUL brut, MSE classique."""
    from keras.callbacks import EarlyStopping
    ts, nb = X_train.shape[1], X_train.shape[2]
    model = Sequential([
        LSTM(64, input_shape=(ts, nb), return_sequences=True, unroll=True),
        LSTM(32, return_sequences=False, unroll=True),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, verbose=0,
              callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])
    return model


def regression_lineaire(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


def random_forest_regressor(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train.ravel())
    return rf


# ── Entraînement ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  PIPELINE 1 : PRÉDICTION DU RUL (valeurs brutes en cycles)")
print("="*60)

results = []

print("\nEntraînement Régression Linéaire...")
lr = regression_lineaire(X_train_2D, y_train_2D)
results.append(evaluer(lr, X_test_2D, y_test_2D, "LR", clip_max=clip_value))

print("Entraînement Random Forest...")
rf = random_forest_regressor(X_train_2D, y_train_2D)
results.append(evaluer(rf, X_test_2D, y_test_2D, "RF", clip_max=clip_value))

# [R2] CNN entraîné et évalué sur RUL brut
print(f"Entraînement CNN (RUL brut en cycles, α={ASYM_ALPHA})...")
cnn = entrainer_cnn(X_train_3D, y_train_3D)
results.append(evaluer(cnn, X_test_3D, y_test_3D, "CNN", clip_max=clip_value))

print("Entraînement LSTM (référence, RUL brut, MSE classique)...")
lstm = entrainer_lstm(X_train_3D, y_train_3D)
results.append(evaluer(lstm, X_test_3D, y_test_3D, "LSTM", clip_max=clip_value))

# Tableau récapitulatif
print("\n" + "="*55)
print("  RÉSULTATS FINAUX (RUL en cycles bruts)")
print("="*55)
df_res = pd.DataFrame([
    {"Modèle": r["Modèle"], "RMSE": r["RMSE"], "MAE": r["MAE"], "Acc_cls": r["Acc_cls"]}
    for r in results
]).sort_values("RMSE")
print(df_res.to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATIONS RUL
# ═════════════════════════════════════════════════════════════════════════════
print("\nGénération des visualisations RUL...")

fig_cm, axes_cm = plt.subplots(1, 4, figsize=(20, 4))
fig_cm.suptitle(
    f"Matrices de Confusion RUL → Classes\n"
    f"Critique ≤ {RUL_SEUILS['critique']}  |  "
    f"Dégradé ≤ {RUL_SEUILS['degrade']}  |  Sain > {RUL_SEUILS['degrade']}",
    fontsize=12, fontweight="bold",
)
for ax, res in zip(axes_cm, results):
    cm = confusion_matrix(res["y_true_cls"], res["y_pred_cls"], labels=[0,1,2])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="YlOrRd",
                xticklabels=CLASSE_NOMS, yticklabels=CLASSE_NOMS, linewidths=0.5)
    ax.set_title(f"{res['Modèle']}  (acc={res['Acc_cls']:.0%})", fontweight="bold")
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
plt.tight_layout()
plt.savefig("rul_classification_matrices.png", dpi=150, bbox_inches="tight")
print("    → rul_classification_matrices.png")

fig_rul, axes_rul = plt.subplots(2, 2, figsize=(14, 10))
fig_rul.suptitle("RUL Prédit vs Réel avec Zones de Classification (cycles bruts)",
                 fontsize=13, fontweight="bold")
for ax, res in zip(axes_rul.flatten(), results):
    y_true  = res["y_true_rul"]
    y_pred  = res["y_pred_rul"]
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax.axhspan(0,                       RUL_SEUILS["critique"], alpha=0.12, color=CLASSE_COLORS[0])
    ax.axhspan(RUL_SEUILS["critique"],  RUL_SEUILS["degrade"],  alpha=0.10, color=CLASSE_COLORS[1])
    ax.axhspan(RUL_SEUILS["degrade"],   max_val,                alpha=0.08, color=CLASSE_COLORS[2])
    ax.axhline(RUL_SEUILS["critique"], color=CLASSE_COLORS[0], linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(RUL_SEUILS["degrade"],  color=CLASSE_COLORS[1], linestyle="--", linewidth=1.2, alpha=0.7)
    ax.scatter(y_true, y_pred, c=[CLASSE_COLORS[c] for c in res["y_true_cls"]],
               alpha=0.65, s=40, edgecolors="white", linewidths=0.4)
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("RUL réel (cycles)"); ax.set_ylabel("RUL prédit (cycles)")
    ax.set_title(f"{res['Modèle']}  RMSE={res['RMSE']}  Acc={res['Acc_cls']:.0%}",
                 fontweight="bold")
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("rul_pred_vs_real_zones.png", dpi=150, bbox_inches="tight")
print("    → rul_pred_vs_real_zones.png")


# ═════════════════════════════════════════════════════════════════════════════
# 8. EXPORT CNN → TFLite INT8 + cerveau_cnn.h
# ═════════════════════════════════════════════════════════════════════════════
print("\nConversion CNN → TFLite Full INT8...")

def representative_dataset():
    for i in range(min(200, len(X_train_3D))):
        yield [X_train_3D[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

with open("cerveau_cnn.tflite", "wb") as f:
    f.write(tflite_model)
print(f"    cerveau_cnn.tflite : {len(tflite_model)/1024:.1f} Ko")


def generer_header_c(tflite_bytes: bytes, nom_fichier: str, nom_variable: str):
    """
    [R2] Le header est simplifié : plus de RUL_MAX ni de dénormalisation.
    La sortie du CNN est directement en cycles — appliquer les seuils
    RUL_SEUIL_CRITIQUE et RUL_SEUIL_DEGRADE sans aucun calcul intermédiaire.
    Clamp à 0 si la sortie est négative (possible avec activation linéaire).
    """
    hex_lines = [
        ", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i+12])
        for i in range(0, len(tflite_bytes), 12)
    ]
    contenu = "\n".join([
        f"// Fichier généré automatiquement — modèle CNN RUL brut (cycles)",
        f"// Seuils RUL : Critique≤{RUL_SEUILS['critique']}  Dégradé≤{RUL_SEUILS['degrade']}",
        f"// NB_FEATURES={len(ESP32_FEATURES)}  TIME_STEPS={TIME_STEPS}",
        f"// Sortie CNN : valeur directe en cycles — aucune dénormalisation nécessaire",
        f"// Utilisation : float rul = output[0]; if (rul < 0) rul = 0;",
        f"#ifndef {nom_variable.upper()}_H",
        f"#define {nom_variable.upper()}_H",
        f"",
        f"#include <pgmspace.h>",
        f"",
        f"#define RUL_SEUIL_CRITIQUE {RUL_SEUILS['critique']}",
        f"#define RUL_SEUIL_DEGRADE  {RUL_SEUILS['degrade']}",
        f"#define NB_FEATURES        {len(ESP32_FEATURES)}",
        f"#define TIME_STEPS         {TIME_STEPS}",
        f"",
        f"alignas(16) const unsigned char {nom_variable}[] PROGMEM = {{",
        "  " + ',\n  '.join(hex_lines),
        f"}};",
        f"const unsigned int {nom_variable}_len = {len(tflite_bytes)};",
        f"",
        f"// Sortie directement en cycles — clamp à 0 si négatif :",
        f"//   float rul = output[0];",
        f"//   if (rul < 0) rul = 0;",
        f"//   int classe = rul_to_classe(rul);",
        f"inline int rul_to_classe(float rul) {{",
        f"  if (rul <= RUL_SEUIL_CRITIQUE) return 0;  // Critique",
        f"  if (rul <= RUL_SEUIL_DEGRADE)  return 1;  // Dégradé",
        f"  return 2;                                  // Sain",
        f"}}",
        f"",
        f"#endif // {nom_variable.upper()}_H",
    ])
    with open(nom_fichier, "w", encoding="utf-8") as f:
        f.write(contenu)

generer_header_c(tflite_model, "cerveau_cnn.h", "cerveau_cnn_tflite")
print(f"    cerveau_cnn.h généré (sortie directe en cycles, pas de RUL_MAX)")
print("    Dans le sketch Arduino :")
print("        float rul = output[0];")
print("        if (rul < 0) rul = 0;   // clamp sécurité")
print("        int classe = rul_to_classe(rul);")


# ═════════════════════════════════════════════════════════════════════════════
# 9. PIPELINE 2 : DÉTECTION DE PANNES (GMM + Random Forest)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  PIPELINE 2 : DÉTECTION DE PANNES")
print("="*60)

def extraire_features(df: pd.DataFrame) -> pd.DataFrame:
    all_s = list(set(HPC_SENSORS + FAN_SENSORS))
    rows  = {}
    for unit in sorted(df["unit_number"].unique()):
        df_u = df[df["unit_number"] == unit].sort_values("time_cycles")
        n    = len(df_u)
        t    = np.arange(n)
        n20  = max(int(n * 0.2), 5)

        def clean(v):
            return pd.Series(v).ffill().bfill().values.astype(float)

        def norm_slope(v):
            v = clean(v)
            r = float(v.max()) - float(v.min())
            if r < 1e-6: return 0.0
            slope, *_ = linregress(t, v)
            return float(slope / r) if np.isfinite(slope) else 0.0

        def end_delta(v):
            v = clean(v)
            e = v[-n20:][~np.isnan(v[-n20:])]
            s = v[:n20][~np.isnan(v[:n20])]
            return float(e.mean()) - float(s.mean()) if len(e) and len(s) else 0.0

        row = {}
        for s in all_s:
            v = df_u[s].values
            row[f"{s}_ns"]    = norm_slope(v)
            row[f"{s}_delta"] = end_delta(v)

        row["T30_vs_Nf"]  = norm_slope(df_u["sensor_3"].values) - norm_slope(df_u["sensor_8"].values)
        row["NRc_vs_NRf"] = norm_slope(df_u["sensor_11"].values) - norm_slope(df_u["sensor_17"].values)
        row["T30_vs_T24"] = norm_slope(df_u["sensor_3"].values) - norm_slope(df_u["sensor_2"].values)
        row["Nc_vs_Nf"]   = norm_slope(df_u["sensor_9"].values) - norm_slope(df_u["sensor_8"].values)
        row["hpc_score"]  = (row["T30_vs_Nf"] + row["NRc_vs_NRf"]
                             + row["T30_vs_T24"] + row["Nc_vs_Nf"])
        row["lifetime"]   = n
        rows[unit] = row
    return pd.DataFrame(rows).T.astype(float)

feat_train = extraire_features(df_train)
feat_test  = extraire_features(df_test)
print(f"    Features extraites : {feat_train.shape[1]} par moteur")

print("\n[A] Clustering GMM...")
DIFF_COLS = ["T30_vs_Nf", "NRc_vs_NRf", "T30_vs_T24", "Nc_vs_Nf", "hpc_score"]

scaler_gmm  = StandardScaler()
imputer_gmm = SimpleImputer(strategy="median")
X_diff = imputer_gmm.fit_transform(scaler_gmm.fit_transform(feat_train[DIFF_COLS]))

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42, n_init=50)
gmm_labels     = gmm.fit_predict(X_diff)
gmm_proba      = gmm.predict_proba(X_diff)
gmm_confidence = gmm_proba.max(axis=1)

hpc_cluster = max([0,1], key=lambda c: feat_train.loc[gmm_labels==c, "hpc_score"].mean())
label_map   = {hpc_cluster: 0, 1-hpc_cluster: 1}
y_train_clf = np.array([label_map[l] for l in gmm_labels])

sil = silhouette_score(X_diff, y_train_clf)
print(f"    Score Silhouette   : {sil:.3f}")
print(f"    Confiance moyenne  : {gmm_confidence.mean():.1%}")
print(f"    HPC : {(y_train_clf==0).sum()} moteurs | Fan : {(y_train_clf==1).sum()} moteurs")

print("\n[B] Classifieur Random Forest...")
ALL_FEAT_COLS = feat_train.columns.tolist()

imputer_clf = SimpleImputer(strategy="median")
scaler_clf  = StandardScaler()
X_train_clf = scaler_clf.fit_transform(imputer_clf.fit_transform(feat_train[ALL_FEAT_COLS]))
X_test_clf  = scaler_clf.transform(imputer_clf.transform(feat_test[ALL_FEAT_COLS]))

clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                              min_samples_leaf=2, random_state=42, n_jobs=-1)
clf.fit(X_train_clf, y_train_clf)

cv_scores = cross_val_score(clf, X_train_clf, y_train_clf,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42),
                             scoring="accuracy")
print(f"    Accuracy CV 5-fold : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
y_pred_train_clf = clf.predict(X_train_clf)
y_prob_train_clf = clf.predict_proba(X_train_clf)
print(classification_report(y_train_clf, y_pred_train_clf,
                             target_names=list(FAULT_NAMES.values()), digits=3))

y_pred_test_clf = clf.predict(X_test_clf)
print(f"    Test : HPC={(y_pred_test_clf==0).sum()} | Fan={(y_pred_test_clf==1).sum()}")

print("\n[C] Visualisations...")
importances = pd.Series(clf.feature_importances_, index=ALL_FEAT_COLS)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Pipeline Détection de Pannes — FD003\nHPC Degradation vs Fan Degradation",
             fontsize=14, fontweight="bold", y=0.98)
colors_train = [FAULT_COLORS[l] for l in y_train_clf]

ax = axes[0, 0]
pca  = PCA(n_components=2)
Xpca = pca.fit_transform(X_diff)
for fid, name in FAULT_NAMES.items():
    mask = y_train_clf == fid
    ax.scatter(Xpca[mask,0], Xpca[mask,1], c=FAULT_COLORS[fid],
               label=name, alpha=0.75, s=60, edgecolors="white", linewidths=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("PCA — Séparation des 2 Fault Modes")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for fid, name in FAULT_NAMES.items():
    mask = y_train_clf == fid
    ax.hist(feat_train["hpc_score"].values[mask], bins=15, alpha=0.7,
            color=FAULT_COLORS[fid], label=name, edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("HPC Score"); ax.set_ylabel("Nombre de moteurs")
ax.set_title("Distribution du Score HPC"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.scatter(feat_train["sensor_3_ns"].values, feat_train["sensor_8_ns"].values,
           c=colors_train, alpha=0.75, s=60, edgecolors="white", linewidths=0.5)
ax.set_xlabel("T30 slope"); ax.set_ylabel("Nf slope")
ax.set_title("T30 vs Nf — Vitesse de dégradation")
patches = [mpatches.Patch(color=FAULT_COLORS[k], label=v) for k,v in FAULT_NAMES.items()]
ax.legend(handles=patches, fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
cm = confusion_matrix(y_train_clf, y_pred_train_clf)
sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="RdBu_r",
            xticklabels=list(FAULT_NAMES.values()),
            yticklabels=list(FAULT_NAMES.values()))
ax.set_title(f"Matrice de Confusion (Train) — CV: {cv_scores.mean():.1%}")
ax.set_ylabel("Vrai label (GMM)"); ax.set_xlabel("Prédit (RF)")

ax = axes[1, 1]
top8 = importances.nlargest(8)
colors_bar = ["#e74c3c" if any(k in n for k in ["hpc","sensor_3","sensor_11","sensor_9"])
              else "#3498db" for n in top8.index]
ax.barh(range(len(top8)), top8.values, color=colors_bar, alpha=0.8, edgecolor="white")
ax.set_yticks(range(len(top8))); ax.set_yticklabels(top8.index, fontsize=9)
ax.set_xlabel("Importance (Gini)"); ax.set_title("Top 8 Features (rouge=HPC, bleu=Fan)")
ax.grid(True, alpha=0.3, axis="x"); ax.invert_yaxis()

ax = axes[1, 2]
conf_hpc = y_prob_train_clf[y_train_clf==0].max(axis=1)
conf_fan  = y_prob_train_clf[y_train_clf==1].max(axis=1)
ax.hist(conf_hpc, bins=12, alpha=0.7, color=FAULT_COLORS[0],
        label=f"HPC (moy={conf_hpc.mean():.0%})", edgecolor="white")
ax.hist(conf_fan, bins=12, alpha=0.7, color=FAULT_COLORS[1],
        label=f"Fan (moy={conf_fan.mean():.0%})", edgecolor="white")
ax.axvline(0.7, color="gray", linestyle="--", linewidth=1, label="Seuil 70%")
ax.set_xlabel("Probabilité"); ax.set_ylabel("Nombre de moteurs")
ax.set_title("Confiance du Classifieur"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fault_detection_results.png", dpi=150, bbox_inches="tight")
print("    → fault_detection_results.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle("Profils de Dégradation par Fault Mode", fontsize=13, fontweight="bold")
for (sensor, title, ax) in [
    ("sensor_3",  "T30 — HPC Outlet Temperature", axes2[0,0]),
    ("sensor_8",  "Nf  — Fan Speed (Physical)",   axes2[0,1]),
    ("sensor_11", "NRc — Corrected Core Speed",   axes2[1,0]),
    ("sensor_17", "NRf — Corrected Fan Speed",    axes2[1,1]),
]:
    for fid, name in FAULT_NAMES.items():
        units_f  = feat_train.index[y_train_clf == fid].astype(int)
        profiles = []
        for unit in units_f:
            df_u = df_train[df_train["unit_number"] == unit].sort_values("time_cycles")
            vals = pd.Series(df_u[sensor].values.astype(float)).interpolate(limit_direction="both").values
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            vn = (vals-vmin)/(vmax-vmin) if vmax-vmin > 1e-6 else np.zeros_like(vals)
            profiles.append(np.interp(np.linspace(0,1,100), np.linspace(0,1,len(vn)), vn))
        profiles = np.array(profiles)
        x = np.linspace(0, 100, 100)
        ax.plot(x, profiles.mean(0), color=FAULT_COLORS[fid], label=name, linewidth=2)
        ax.fill_between(x, profiles.mean(0)-profiles.std(0),
                           profiles.mean(0)+profiles.std(0),
                        color=FAULT_COLORS[fid], alpha=0.15)
    ax.set_xlabel("% de durée de vie"); ax.set_ylabel("Valeur normalisée [0,1]")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fault_degradation_profiles.png", dpi=150, bbox_inches="tight")
print("    → fault_degradation_profiles.png")


# ═════════════════════════════════════════════════════════════════════════════
# 10. SAUVEGARDE DES ARTEFACTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n[D] Sauvegarde des artefacts...")
joblib.dump(gmm,         "gmm_fault_model.pkl")
joblib.dump(clf,         "rf_fault_classifier.pkl")
joblib.dump(scaler_clf,  "scaler_fault_classifier.pkl")
joblib.dump(imputer_clf, "imputer_fault_classifier.pkl")
joblib.dump(scaler_gmm,  "scaler_fault_gmm.pkl")
joblib.dump(imputer_gmm, "imputer_fault_gmm.pkl")
print("    6 fichiers .pkl sauvegardés.")
print(f"    rul_clip_value.txt : {clip_value} cycles (pour info, pas pour dénorm)")

print("\n" + "="*60)
print("  RÉSUMÉ FINAL")
print("="*60)
for name, count in {v: (y_train_clf==k).sum() for k,v in FAULT_NAMES.items()}.items():
    print(f"  TRAIN {name}: {count} moteurs ({count/len(y_train_clf):.0%})")
for name, count in {v: (y_pred_test_clf==k).sum() for k,v in FAULT_NAMES.items()}.items():
    print(f"  TEST  {name}: {count} moteurs ({count/len(y_pred_test_clf):.0%})")
print(f"\n  CV Accuracy     : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"  Silhouette GMM  : {sil:.3f}")
print(f"  Confiance RF    : {gmm_confidence.mean():.1%}")
print(f"  Loss α (CNN)    : {ASYM_ALPHA}")
print(f"  Clip RUL        : {clip_value} cycles")
print(f"  Sortie CNN      : valeur directe en cycles (activation linéaire)")
print("\n  Pipeline terminé avec succès ✅")

# ── Scalers pour l'Arduino ────────────────────────────────────────────────────
scaler_rul = joblib.load("scaler_cloud.pkl")

print("\n=== SCALERS POUR L'ARDUINO ===")
print("const float SCALER_MIN[NB_FEATURES] PROGMEM = {")
vals = ", ".join(f"{v:.6f}f" for v in scaler_rul.data_min_)
print(f"  {vals}")
print("};")
print("const float SCALER_RANGE[NB_FEATURES] PROGMEM = {")
vals = ", ".join(f"{v:.6f}f" for v in scaler_rul.data_range_)
print(f"  {vals}")
print("};")
print(f"\n// Inférence Arduino (pas de RUL_MAX, pas de dénormalisation) :")
print(f"// float rul = output[0];")
print(f"// if (rul < 0) rul = 0;   // clamp sécurité (sortie linéaire)")
print(f"// int classe = rul_to_classe(rul);")