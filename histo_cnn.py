import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (mean_squared_error, classification_report,
                             confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CHEMIN_TRAIN = r"C:\Users\HP\Desktop\pfa_proj\train_FD003_historique.csv"
CHEMIN_TEST  = r"C:\Users\HP\Desktop\pfa_proj\test_FD003_historique.csv"
CHEMIN_RUL   = r"C:\Users\HP\Desktop\pfa_proj\RUL_FD003_historique.csv"
TIME_STEPS   = 50

DROP_COLS = [
    'unit_number', 'time_cycles',
    'setting_3',
    'sensor_1', 'sensor_5', 'sensor_6',
    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'
]

HPC_SENSORS  = ['sensor_3', 'sensor_11', 'sensor_9']
FAN_SENSORS  = ['sensor_8', 'sensor_17', 'sensor_2', 'sensor_13']
FAULT_NAMES  = {0: 'HPC Degradation', 1: 'Fan Degradation'}
FAULT_COLORS = {0: '#e74c3c', 1: '#3498db'}

RUL_SEUILS = {'critique': 30, 'degrade': 80}
CLASSE_NOMS   = ['Critique', 'Dégradé', 'Sain']
CLASSE_COLORS = ['#e74c3c', '#f39c12', '#27ae60']

# ─────────────────────────────────────────
# CHARGEMENT ROBUSTE DES CSV
# ─────────────────────────────────────────
# Noms des 26 colonnes du dataset CMAPSS
COLS = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
COLS += [f'sensor_{i}' for i in range(1, 22)]

def charger_csv(path, cols=COLS):
    """
    Chargement robuste : détecte automatiquement si le fichier est séparé
    par virgule ou par espace, et gère la présence ou non d'un header.
    """
    with open(path, 'r') as f:
        first_line = f.readline().strip()

    sep = ',' if first_line.count(',') > first_line.count(' ') else r'\s+'

    try:
        float(first_line.split(',' if sep == ',' else None)[0])
        has_header = False
    except ValueError:
        has_header = True

    df = pd.read_csv(
        path,
        sep=sep,
        header=0 if has_header else None,
        names=None if has_header else cols,
        engine='python'
    )

    if has_header:
        if df.shape[1] == len(cols):
            df.columns = cols
        else:
            raise ValueError(
                f"Nombre de colonnes inattendu : {df.shape[1]} (attendu {len(cols)})\n"
                f"Colonnes trouvées : {df.columns.tolist()}"
            )

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    nan_total = df.isna().sum().sum()
    if nan_total > 0:
        cols_with_nan = df.columns[df.isna().any()].tolist()
        print(f"    ⚠️  {nan_total} NaN après chargement dans : {cols_with_nan}")

    return df

print("Chargement des données d'entraînement et de test...")
df_train = charger_csv(CHEMIN_TRAIN)
df_test  = charger_csv(CHEMIN_TEST)
df_rul   = pd.read_csv(CHEMIN_RUL, header=None, names=['true_rul'])

print(f"    Train : {df_train.shape[0]} lignes, {df_train['unit_number'].nunique()} moteurs")
print(f"    Test  : {df_test.shape[0]}  lignes, {df_test['unit_number'].nunique()} moteurs")

# ═════════════════════════════════════════
# PIPELINE 1 : PRÉDICTION DU RUL
# ═════════════════════════════════════════

# ─────────────────────────────────────────
# 1. Calcul du RUL
# ─────────────────────────────────────────
def calculer_rul(df):
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop(columns=['max_cycle'])
    return df

df_train = calculer_rul(df_train)
df_train['RUL'] = df_train['RUL'].clip(upper=125)

# ─────────────────────────────────────────
# 2. Preprocessing (normalisation partagée)
# ─────────────────────────────────────────
def pretraitement(df_train, df_test, chemin_scaler='scaler_cloud.pkl'):
    features = [c for c in df_train.columns if c not in DROP_COLS + ['RUL']]
    scaler = MinMaxScaler()
    df_train[features] = scaler.fit_transform(df_train[features])
    df_test[features]  = scaler.transform(df_test[features])
    joblib.dump(scaler, chemin_scaler)
    print(f"Scaler sauvegardé → {chemin_scaler}")
    return df_train, df_test, features

df_train, df_test, features = pretraitement(df_train, df_test)

# ─────────────────────────────────────────
# 3. Préparation des données
# ─────────────────────────────────────────
def preparer_test_2d(df_test, df_rul, features):
    last_cycles = df_test.groupby('unit_number').last().reset_index()
    last_cycles = last_cycles.sort_values('unit_number').reset_index(drop=True)
    X_test = last_cycles[features].values
    y_test = df_rul['true_rul'].values
    return X_test, y_test

def creer_sequences_3d(df, features, time_steps=30):
    X_seq, y_seq = [], []
    for unit in df['unit_number'].unique():
        df_unit = df[df['unit_number'] == unit]
        data = df_unit[features].values
        ruls = df_unit['RUL'].values
        for i in range(len(data) - time_steps):
            X_seq.append(data[i:i + time_steps])
            y_seq.append(ruls[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def preparer_test_3d(df_test, df_rul, features, time_steps=30):
    X_seq, y_seq = [], []
    units = sorted(df_test['unit_number'].unique())
    for i, unit in enumerate(units):
        df_unit = df_test[df_test['unit_number'] == unit]
        data = df_unit[features].values
        if len(data) >= time_steps:
            seq = data[-time_steps:]
        else:
            pad = np.zeros((time_steps - len(data), len(features)))
            seq = np.vstack([pad, data])
        X_seq.append(seq)
        y_seq.append(df_rul['true_rul'].iloc[i])
    return np.array(X_seq), np.array(y_seq)

X_train_2D = df_train[features].values
y_train_2D = df_train['RUL'].values
X_test_2D, y_test_2D = preparer_test_2d(df_test, df_rul, features)

X_train_3D, y_train_3D = creer_sequences_3d(df_train, features, TIME_STEPS)
X_test_3D, y_test_3D   = preparer_test_3d(df_test, df_rul, features, TIME_STEPS)

print(f"\nFormes → Train 2D: {X_train_2D.shape} | Test 2D: {X_test_2D.shape}")
print(f"         Train 3D: {X_train_3D.shape} | Test 3D: {X_test_3D.shape}\n")

# ─────────────────────────────────────────
# 4. Modèles RUL
# ─────────────────────────────────────────
def rul_to_classe(rul_array, seuils=RUL_SEUILS):
    classes = np.full(len(rul_array), 2, dtype=int)
    classes[rul_array <= seuils['degrade']]  = 1
    classes[rul_array <= seuils['critique']] = 0
    return classes

def evaluer_classification(modele, X_test, y_test_rul, nom_modele, seuils=RUL_SEUILS):
    y_pred_rul = modele.predict(X_test).flatten()
    y_true_cls = rul_to_classe(y_test_rul, seuils)
    y_pred_cls = rul_to_classe(y_pred_rul, seuils)

    rmse = np.sqrt(mean_squared_error(y_test_rul, y_pred_rul))
    mae  = np.mean(np.abs(y_test_rul - y_pred_rul))
    acc  = (y_true_cls == y_pred_cls).mean()

    print(f"\n{'='*55}")
    print(f"  [{nom_modele}]")
    print(f"  Régression  → RMSE={rmse:.2f}  MAE={mae:.2f}")
    print(f"  Seuils RUL  → Critique≤{seuils['critique']}  "
          f"Dégradé≤{seuils['degrade']}  Sain>{seuils['degrade']}")
    print(f"\n  Rapport de classification :")
    print(classification_report(
        y_true_cls, y_pred_cls,
        target_names=CLASSE_NOMS,
        labels=[0, 1, 2],
        zero_division=0
    ))

    return {
        "Modèle":     nom_modele,
        "RMSE":       round(rmse, 2),
        "MAE":        round(mae, 2),
        "Acc_cls":    round(acc, 3),
        "y_true_cls": y_true_cls,
        "y_pred_cls": y_pred_cls,
        "y_pred_rul": y_pred_rul,
        "y_true_rul": y_test_rul,
    }

def regression_lineaire(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def random_forest_regressor(X_train, y_train, n_estimators=100):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train.ravel())
    return rf

def entrainer_lstm(X_train, y_train, epochs=30, batch_size=64):
    time_steps, nb_features = X_train.shape[1], X_train.shape[2]
    model = Sequential([
        LSTM(64, input_shape=(time_steps, nb_features), return_sequences=True, unroll=True),
        LSTM(32, return_sequences=False, unroll=True),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, verbose=0)
    return model

def entrainer_cnn(X_train, y_train, epochs=30, batch_size=64):
    time_steps, nb_features = X_train.shape[1], X_train.shape[2]
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu',
               input_shape=(time_steps, nb_features)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, verbose=0)
    return model

# ─────────────────────────────────────────
# Entraînement et Évaluation
# ─────────────────────────────────────────
results = []

print("Entraînement Régression Linéaire...")
lr = regression_lineaire(X_train_2D, y_train_2D)
results.append(evaluer_classification(lr, X_test_2D, y_test_2D, "LR"))

print("Entraînement Random Forest...")
rf = random_forest_regressor(X_train_2D, y_train_2D)
results.append(evaluer_classification(rf, X_test_2D, y_test_2D, "RF"))

print("Entraînement CNN...")
cnn = entrainer_cnn(X_train_3D, y_train_3D, epochs=30)
results.append(evaluer_classification(cnn, X_test_3D, y_test_3D, "CNN"))

print("Entraînement LSTM...")
lstm = entrainer_lstm(X_train_3D, y_train_3D, epochs=30)
results.append(evaluer_classification(lstm, X_test_3D, y_test_3D, "LSTM"))

# ─────────────────────────────────────────
# Tableau récapitulatif
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  RÉSULTATS FINAUX RUL + CLASSIFICATION")
print("=" * 55)
df_results = pd.DataFrame([
    {"Modèle": r["Modèle"], "RMSE": r["RMSE"], "MAE": r["MAE"], "Acc_cls": r["Acc_cls"]}
    for r in results
]).sort_values("RMSE")
print(df_results.to_string(index=False))

# ─────────────────────────────────────────
# Visualisations RUL
# ─────────────────────────────────────────
print("\nGénération des matrices de confusion RUL → classes...")

fig_cm, axes_cm = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
fig_cm.suptitle(
    f"Matrices de Confusion RUL → Classes\n"
    f"Seuils : Critique ≤ {RUL_SEUILS['critique']}  |  "
    f"Dégradé ≤ {RUL_SEUILS['degrade']}  |  Sain > {RUL_SEUILS['degrade']}",
    fontsize=12, fontweight='bold'
)
for ax, res in zip(axes_cm, results):
    cm = confusion_matrix(res['y_true_cls'], res['y_pred_cls'], labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='YlOrRd',
                xticklabels=CLASSE_NOMS, yticklabels=CLASSE_NOMS, linewidths=0.5)
    ax.set_title(f"{res['Modèle']}  (acc={res['Acc_cls']:.0%})", fontweight='bold')
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
plt.tight_layout()
plt.savefig('rul_classification_matrices.png', dpi=150, bbox_inches='tight')
print("    Sauvegardé → rul_classification_matrices.png")

fig_rul, axes_rul = plt.subplots(2, 2, figsize=(14, 10))
fig_rul.suptitle('RUL Prédit vs Réel avec Zones de Classification',
                 fontsize=13, fontweight='bold')
for ax, res in zip(axes_rul.flatten(), results):
    y_true  = res['y_true_rul']
    y_pred  = res['y_pred_rul']
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax.axhspan(0,                      RUL_SEUILS['critique'], alpha=0.12, color=CLASSE_COLORS[0])
    ax.axhspan(RUL_SEUILS['critique'], RUL_SEUILS['degrade'],  alpha=0.10, color=CLASSE_COLORS[1])
    ax.axhspan(RUL_SEUILS['degrade'],  max_val,                alpha=0.08, color=CLASSE_COLORS[2])
    ax.axhline(RUL_SEUILS['critique'], color=CLASSE_COLORS[0], linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(RUL_SEUILS['degrade'],  color=CLASSE_COLORS[1], linestyle='--', linewidth=1.2, alpha=0.7)
    ax.scatter(y_true, y_pred, c=[CLASSE_COLORS[c] for c in res['y_true_cls']],
               alpha=0.65, s=40, edgecolors='white', linewidths=0.4)
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('RUL réel'); ax.set_ylabel('RUL prédit')
    ax.set_title(f"{res['Modèle']}  RMSE={res['RMSE']}  Acc={res['Acc_cls']:.0%}", fontweight='bold')
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('rul_pred_vs_real_zones.png', dpi=150, bbox_inches='tight')
print("    Sauvegardé → rul_pred_vs_real_zones.png")

# ─────────────────────────────────────────
# Export CNN → TFLite → C header for ESP32 (QUANTIFICATION TOTALE INT8)
# ─────────────────────────────────────────
print("\nConversion du modèle CNN en TensorFlow Lite (Full INT8)...")

def representative_dataset():
    # Échantillon nécessaire pour calibrer les poids en entiers
    for i in range(100):
        yield [X_train_3D[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Forcer toutes les opérations internes en INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Garder les entrées/sorties en Float32 pour ne pas casser le code C++ actuel
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open("cerveau_cnn.tflite", "wb") as f:
    f.write(tflite_model)

def convertir_en_c_array(tflite_bytes, nom_fichier_h, nom_variable):
    hex_array = ', '.join([f'0x{b:02x}' for b in tflite_bytes])
    c_code = f'// Fichier généré automatiquement pour l\'ESP32 (Full INT8 CNN)\n'
    c_code += f'// Seuils RUL embarqués : Critique<={RUL_SEUILS["critique"]}  Dégradé<={RUL_SEUILS["degrade"]}\n'
    c_code += f'#ifndef {nom_variable.upper()}_H\n'
    c_code += f'#define {nom_variable.upper()}_H\n\n'
    c_code += f'#include <pgmspace.h>\n\n'
    c_code += f'#define RUL_SEUIL_CRITIQUE {RUL_SEUILS["critique"]}\n'
    c_code += f'#define RUL_SEUIL_DEGRADE  {RUL_SEUILS["degrade"]}\n\n'
    
    # Séparateur propre pour éviter les lignes trop longues
    hex_lines = [
        ", ".join([f"0x{b:02x}" for b in tflite_bytes[i:i + 12]])
        for i in range(0, len(tflite_bytes), 12)
    ]
    separateur = ",\n  "
    
    c_code += f'alignas(16) const unsigned char {nom_variable}[] PROGMEM = {{\n  {separateur.join(hex_lines)}\n}};\n\n'
    c_code += f'const unsigned int {nom_variable}_len = {len(tflite_bytes)};\n\n'
    c_code += f'inline int rul_to_classe(float rul) {{\n'
    c_code += f'  if (rul <= RUL_SEUIL_CRITIQUE) return 0;\n'
    c_code += f'  if (rul <= RUL_SEUIL_DEGRADE)  return 1;\n'
    c_code += f'  return 2;\n}}\n\n'
    c_code += f'#endif // {nom_variable.upper()}_H\n'
    
    with open(nom_fichier_h, 'w', encoding="utf-8") as f:
        f.write(c_code)
        
convertir_en_c_array(tflite_model, "cerveau_cnn.h", "cerveau_cnn_tflite")
print("Export ESP32 : cerveau_cnn.tflite + cerveau_cnn.h générés avec succès !")
# ═════════════════════════════════════════
# PIPELINE 2 : DÉTECTION DE PANNES
# ═════════════════════════════════════════
print("\n" + "=" * 50)
print("  Pipeline Détection de Pannes - FD003")
print("=" * 50)

# ─────────────────────────────────────────
# Feature engineering (robuste aux NaN) 
# ─────────────────────────────────────────
def extraire_features(df):
    all_sensors = list(set(HPC_SENSORS + FAN_SENSORS))
    rows = {}

    for unit in sorted(df['unit_number'].unique()):
        df_u = df[df['unit_number'] == unit].sort_values('time_cycles')
        n    = len(df_u)
        t    = np.arange(n)
        n20  = max(int(n * 0.2), 5)

        def clean(v):
            """Forward-fill puis backward-fill les NaN."""
            return pd.Series(v).ffill().bfill().values.astype(float)

        def norm_slope(v):
            v = clean(v)
            r = float(v.max()) - float(v.min())
            if r < 1e-6:
                return 0.0
            slope, *_ = linregress(t, v)
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
        rows[unit] = row

    return pd.DataFrame(rows).T.astype(float)

feat_train = extraire_features(df_train)
feat_test  = extraire_features(df_test)
print(f"    Features extraites: {feat_train.shape[1]} par moteur")

nan_train = feat_train.isna().sum().sum()
nan_test  = feat_test.isna().sum().sum()
if nan_train > 0 or nan_test > 0:
    print(f"    ⚠️  NaN résiduels — train: {nan_train}, test: {nan_test} → imputation par médiane")

# ─────────────────────────────────────────
# Clustering GMM (labellisation non supervisée)
# ─────────────────────────────────────────
print("\n[3] Clustering GMM pour labellisation...")

DIFF_COLS = ['T30_vs_Nf', 'NRc_vs_NRf', 'T30_vs_T24', 'Nc_vs_Nf', 'hpc_score']

scaler_gmm  = StandardScaler()
X_diff_raw  = scaler_gmm.fit_transform(feat_train[DIFF_COLS])

imputer_gmm = SimpleImputer(strategy='median')
X_diff      = imputer_gmm.fit_transform(X_diff_raw)

if np.isnan(X_diff_raw).sum() > 0:
    print(f"    ℹ️  {np.isnan(X_diff_raw).sum()} NaN imputés avant GMM")

gmm = GaussianMixture(n_components=2, covariance_type='full',
                      random_state=42, n_init=50)
gmm_labels     = gmm.fit_predict(X_diff)
gmm_proba      = gmm.predict_proba(X_diff)
gmm_confidence = gmm_proba.max(axis=1)

cluster_hpc_score = {
    c: feat_train.loc[gmm_labels == c, 'hpc_score'].mean()
    for c in [0, 1]
}
hpc_cluster = max(cluster_hpc_score, key=cluster_hpc_score.get)
fan_cluster  = 1 - hpc_cluster
label_map    = {hpc_cluster: 0, fan_cluster: 1}
y_train      = np.array([label_map[l] for l in gmm_labels])

sil = silhouette_score(X_diff, y_train)
print(f"    Score Silhouette: {sil:.3f}  (>0.3 = clustering acceptable)")
print(f"    Distribution: HPC={(y_train == 0).sum()} moteurs | Fan={(y_train == 1).sum()} moteurs")
print(f"    Confiance moyenne: {gmm_confidence.mean():.1%}")

for fault_id, name in FAULT_NAMES.items():
    mask = y_train == fault_id
    sub  = feat_train[mask]
    print(f"\n    [{name}] n={mask.sum()}")
    print(f"      T30_slope:  {sub['sensor_3_ns'].mean():.5f}")
    print(f"      NRc_slope:  {sub['sensor_11_ns'].mean():.5f}")
    print(f"      Nf_slope:   {sub['sensor_8_ns'].mean():.5f}")
    print(f"      NRf_slope:  {sub['sensor_17_ns'].mean():.5f}")
    print(f"      hpc_score:  {sub['hpc_score'].mean():.5f}")

# ─────────────────────────────────────────
# Classifieur supervisé Random Forest
# ─────────────────────────────────────────
print("\n[4] Entraînement du classifieur Random Forest...")

ALL_FEAT_COLS = feat_train.columns.tolist()

imputer_clf = SimpleImputer(strategy='median')
scaler_clf  = StandardScaler()

X_train_imp = imputer_clf.fit_transform(feat_train[ALL_FEAT_COLS])
X_test_imp  = imputer_clf.transform(feat_test[ALL_FEAT_COLS])

X_train_clf = scaler_clf.fit_transform(X_train_imp)
X_test_clf  = scaler_clf.transform(X_test_imp)

clf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
clf.fit(X_train_clf, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train_clf, y_train, cv=cv, scoring='accuracy')
print(f"    Accuracy CV 5-fold: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

y_pred_train = clf.predict(X_train_clf)
y_prob_train = clf.predict_proba(X_train_clf)
print(f"\n    Rapport de classification (Train):")
print(classification_report(y_train, y_pred_train,
                             target_names=list(FAULT_NAMES.values()), digits=3))

y_pred_test = clf.predict(X_test_clf)
y_prob_test = clf.predict_proba(X_test_clf)
print(f"    Distribution test prédit: HPC={(y_pred_test == 0).sum()} | Fan={(y_pred_test == 1).sum()}")

# ─────────────────────────────────────────
# Importance des features
# ─────────────────────────────────────────
print("\n[5] Features les plus importantes...")
importances = pd.Series(clf.feature_importances_, index=ALL_FEAT_COLS)
top10 = importances.nlargest(10)
for feat_name, imp in top10.items():
    print(f"    {feat_name:20s} {imp:.4f}  {'█' * int(imp * 200)}")

# ─────────────────────────────────────────
# Visualisations (6 plots)
# ─────────────────────────────────────────
print("\n[6] Génération des visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Pipeline Détection de Pannes - FD003\nHPC Degradation vs Fan Degradation',
             fontsize=14, fontweight='bold', y=0.98)
colors_train = [FAULT_COLORS[l] for l in y_train]

# Plot 1 : PCA
ax = axes[0, 0]
pca  = PCA(n_components=2)
Xpca = pca.fit_transform(X_diff)
for fault_id, name in FAULT_NAMES.items():
    mask = y_train == fault_id
    ax.scatter(Xpca[mask, 0], Xpca[mask, 1], c=FAULT_COLORS[fault_id],
               label=name, alpha=0.75, s=60, edgecolors='white', linewidths=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
ax.set_title('PCA - Séparation des 2 Fault Modes')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Plot 2 : Distribution hpc_score
ax = axes[0, 1]
for fault_id, name in FAULT_NAMES.items():
    mask = y_train == fault_id
    ax.hist(feat_train['hpc_score'].values[mask], bins=15, alpha=0.7,
            color=FAULT_COLORS[fault_id], label=name, edgecolor='white')
ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Seuil 0')
ax.set_xlabel('HPC Score'); ax.set_ylabel('Nombre de moteurs')
ax.set_title('Distribution du Score HPC')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Plot 3 : T30 vs Nf slopes
ax = axes[0, 2]
ax.scatter(feat_train['sensor_3_ns'].values, feat_train['sensor_8_ns'].values,
           c=colors_train, alpha=0.75, s=60, edgecolors='white', linewidths=0.5)
lim = max(feat_train['sensor_3_ns'].abs().max(), feat_train['sensor_8_ns'].abs().max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5, label='HPC = Fan')
ax.set_xlabel('T30 slope (HPC outlet temp)'); ax.set_ylabel('Nf slope (Fan speed)')
ax.set_title('T30 vs Nf - Vitesse de dégradation')
patches = [mpatches.Patch(color=FAULT_COLORS[k], label=v) for k, v in FAULT_NAMES.items()]
ax.legend(handles=patches, fontsize=9); ax.grid(True, alpha=0.3)

# Plot 4 : Matrice de confusion
ax = axes[1, 0]
cm_fault = confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_fault, annot=True, fmt='d', ax=ax, cmap='RdBu_r',
            xticklabels=list(FAULT_NAMES.values()),
            yticklabels=list(FAULT_NAMES.values()))
ax.set_title(f'Matrice de Confusion (Train) — CV: {cv_scores.mean():.1%}')
ax.set_ylabel('Vrai label (GMM)'); ax.set_xlabel('Prédit (RF)')

# Plot 5 : Importance features
ax = axes[1, 1]
top8 = importances.nlargest(8)
colors_bar = ['#e74c3c' if any(k in n for k in ['hpc', 'sensor_3', 'sensor_11', 'sensor_9'])
              else '#3498db' for n in top8.index]
ax.barh(range(len(top8)), top8.values, color=colors_bar, alpha=0.8, edgecolor='white')
ax.set_yticks(range(len(top8))); ax.set_yticklabels(top8.index, fontsize=9)
ax.set_xlabel('Importance (Gini)'); ax.set_title('Top 8 Features (rouge=HPC, bleu=Fan)')
ax.grid(True, alpha=0.3, axis='x'); ax.invert_yaxis()

# Plot 6 : Confiance par classe
ax = axes[1, 2]
conf_hpc = y_prob_train[y_train == 0].max(axis=1)
conf_fan  = y_prob_train[y_train == 1].max(axis=1)
ax.hist(conf_hpc, bins=12, alpha=0.7, color=FAULT_COLORS[0],
        label=f'HPC (moy={conf_hpc.mean():.0%})', edgecolor='white')
ax.hist(conf_fan, bins=12, alpha=0.7, color=FAULT_COLORS[1],
        label=f'Fan (moy={conf_fan.mean():.0%})', edgecolor='white')
ax.axvline(0.7, color='gray', linestyle='--', linewidth=1, label='Seuil 70%')
ax.set_xlabel('Probabilité'); ax.set_ylabel('Nombre de moteurs')
ax.set_title('Confiance du Classifieur')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fault_detection_results.png', dpi=150, bbox_inches='tight')
print("    Graphique sauvegardé → fault_detection_results.png")

# ─────────────────────────────────────────
# Profils de dégradation par fault mode
# ─────────────────────────────────────────
print("\n[7] Visualisation des profils de dégradation...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle('Profils de Dégradation par Fault Mode', fontsize=13, fontweight='bold')

sensor_plots = [
    ('sensor_3',  'T30 - HPC Outlet Temperature', axes2[0, 0]),
    ('sensor_8',  'Nf  - Fan Speed (Physical)',    axes2[0, 1]),
    ('sensor_11', 'NRc - Corrected Core Speed',    axes2[1, 0]),
    ('sensor_17', 'NRf - Corrected Fan Speed',     axes2[1, 1]),
]

for sensor, title, ax in sensor_plots:
    for fault_id, name in FAULT_NAMES.items():
        units_fault = feat_train.index[y_train == fault_id].astype(int)
        profiles = []
        for unit in units_fault:
            df_u = df_train[df_train['unit_number'] == unit].sort_values('time_cycles')
            vals = df_u[sensor].values.astype(float)
            vals = pd.Series(vals).interpolate(limit_direction='both').values
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            vals_norm = (vals - vmin) / (vmax - vmin) if vmax - vmin > 1e-6 else np.zeros_like(vals)
            x_norm   = np.linspace(0, 1, len(vals_norm))
            x_interp = np.linspace(0, 1, 100)
            profiles.append(np.interp(x_interp, x_norm, vals_norm))

        profiles     = np.array(profiles)
        mean_profile = np.nanmean(profiles, axis=0)
        std_profile  = np.nanstd(profiles, axis=0)
        x = np.linspace(0, 100, 100)
        ax.plot(x, mean_profile, color=FAULT_COLORS[fault_id], label=name, linewidth=2)
        ax.fill_between(x, mean_profile - std_profile, mean_profile + std_profile,
                        color=FAULT_COLORS[fault_id], alpha=0.15)

    ax.set_xlabel('% de durée de vie'); ax.set_ylabel('Valeur normalisée [0,1]')
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fault_degradation_profiles.png', dpi=150, bbox_inches='tight')
print("    Graphique sauvegardé → fault_degradation_profiles.png")

# ─────────────────────────────────────────
# Sauvegarde des modèles
# ─────────────────────────────────────────
print("\n[8] Sauvegarde des modèles...")
joblib.dump(gmm,         'gmm_fault_model.pkl')
joblib.dump(clf,         'rf_fault_classifier.pkl')
joblib.dump(scaler_clf,  'scaler_fault_classifier.pkl')
joblib.dump(scaler_gmm,  'scaler_fault_gmm.pkl')
joblib.dump(imputer_clf, 'imputer_fault_classifier.pkl') 
joblib.dump(imputer_gmm, 'imputer_fault_gmm.pkl') 
print("    6 fichiers .pkl sauvegardés.")

# ─────────────────────────────────────────
# Résumé final
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  RÉSULTATS FINAUX")
print("=" * 60)

for name, count in {v: (y_train == k).sum() for k, v in FAULT_NAMES.items()}.items():
    pct = count / len(y_train) * 100
    print(f"  TRAIN {name}: {count} moteurs ({pct:.0f}%) {'█' * int(pct / 3)}")

for name, count in {v: (y_pred_test == k).sum() for k, v in FAULT_NAMES.items()}.items():
    pct = count / len(y_pred_test) * 100
    print(f"  TEST  {name}: {count} moteurs ({pct:.0f}%) {'█' * int(pct / 3)}")

print(f"\n  Accuracy CV 5-fold : {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"  Score Silhouette   : {sil:.3f}")
print(f"  Confiance moyenne  : {gmm_confidence.mean():.1%}")

print(f"\n  Seuils RUL → Classification :")
print(f"    Critique  : RUL ≤ {RUL_SEUILS['critique']}  (maintenance immédiate)")
print(f"    Dégradé   : RUL ≤ {RUL_SEUILS['degrade']}  (surveillance renforcée)")
print(f"    Sain      : RUL > {RUL_SEUILS['degrade']}  (fonctionnement normal)")

print("\n" + "=" * 60)
print("  Pipeline terminé avec succès !")
print("=" * 60)