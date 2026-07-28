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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (mean_squared_error, classification_report,
                              confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CHEMIN_TRAIN = r"C:\Users\HP\Desktop\pfa_proj\train_FD003_historique.csv"
CHEMIN_TEST  = r"C:\Users\HP\Desktop\pfa_proj\test_FD003_historique.csv"
CHEMIN_RUL   = r"C:\Users\HP\Desktop\pfa_proj\RUL_FD003_historique.csv"

TIME_STEPS = 30

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

RUL_SEUILS = {
    'critique': 30,
    'degrade':  80,
}
CLASSE_NOMS   = ['Critique', 'Dégradé', 'Sain']
CLASSE_COLORS = ['#e74c3c', '#f39c12', '#27ae60']

# ─────────────────────────────────────────
# 1. CALCUL DU RUL
# ─────────────────────────────────────────
def calculer_rul(df):
    max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop(columns=['max_cycle'])
    return df

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
def pretraitement(df_train, df_test, chemin_scaler='scaler_cloud.pkl'):
    features = [c for c in df_train.columns if c not in DROP_COLS + ['RUL']]
    scaler = MinMaxScaler()
    df_train[features] = scaler.fit_transform(df_train[features])
    df_test[features]  = scaler.transform(df_test[features])
    joblib.dump(scaler, chemin_scaler)
    print(f"Scaler sauvegardé → {chemin_scaler}")
    return df_train, df_test, features

# ─────────────────────────────────────────
# 3. PRÉPARATION DES DONNÉES TEST (2D)
# ─────────────────────────────────────────
def preparer_test_2d(df_test, df_rul, features):
    last_cycles = df_test.groupby('unit_number').last().reset_index()
    last_cycles = last_cycles.sort_values('unit_number').reset_index(drop=True)
    X_test = last_cycles[features].values
    y_test = df_rul['true_rul'].values
    return X_test, y_test

# ─────────────────────────────────────────
# 4. SÉQUENCES 3D (LSTM / CNN)
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# 5. MODÈLES RUL
# ─────────────────────────────────────────
def regression_lineaire(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def random_forest_regressor(X_train, y_train, n_estimators=100):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train.ravel())
    return rf

# ══════════════════════════════════════════════════════════════
# ✅ LSTM CORRIGÉ POUR TFLite INT8 SANS SELECT_TF_OPS
#
#  Clé : unroll=True + batch_shape fixe (batch=1)
#  → graphe statique → pas de tensor list ops dynamiques
#  → compatible TFLITE_BUILTINS_INT8 pur → Flash minimal
# ══════════════════════════════════════════════════════════════
def entrainer_lstm(X_train, y_train, epochs=50, batch_size=64):
    time_steps, nb_features = X_train.shape[1], X_train.shape[2]

    # ── Functional API avec batch_shape=1 fixe ──────────────
    # (batch=1 obligatoire pour unroll=True en TFLite)
    inputs = tf.keras.Input(
        batch_shape=(1, time_steps, nb_features),  # ← batch fixe à 1
        name='input_sequence'
    )
    x = tf.keras.layers.LSTM(
        24,
        return_sequences=False,
        unroll=True,          # ← clé : déplie les pas de temps → graphe statique
        stateful=False,
        name='lstm_24'
    )(inputs)
    x = tf.keras.layers.Dense(16, activation='relu', name='dense_16')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', name='rul_output')(x)

    model = tf.keras.Model(inputs, outputs, name='lstm_lite_esp32')
    model.compile(loss='mean_squared_error', optimizer='adam')

    # ── Entraînement avec batch_size variable (reshape après) ─
    # On entraîne avec un modèle temporaire sans batch fixe
    inputs_train = tf.keras.Input(
        shape=(time_steps, nb_features),
        name='input_train'
    )
    x_t = tf.keras.layers.LSTM(
        24, return_sequences=False, unroll=True, stateful=False
    )(inputs_train)
    x_t = tf.keras.layers.Dense(16, activation='relu')(x_t)
    out_t = tf.keras.layers.Dense(1, activation='linear')(x_t)

    model_train = tf.keras.Model(inputs_train, out_t)
    model_train.compile(loss='mean_squared_error', optimizer='adam')

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_loss')
    ]

    model_train.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )

    # ── Transfert des poids vers le modèle batch=1 (pour export) ──
    model.set_weights(model_train.get_weights())

    print(f"\n  Paramètres LSTM : {model.count_params():,}")
    print(f"  → Arena ESP32 estimée : ~{int(model.count_params() * 4 * 3 / 1024)} Ko")

    return model, model_train   # model=export, model_train=inférence Python

def entrainer_cnn(X_train, y_train, epochs=30, batch_size=64):
    time_steps, nb_features = X_train.shape[1], X_train.shape[2]
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                               input_shape=(time_steps, nb_features)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, verbose=0)
    return model

# ─────────────────────────────────────────
# 6. CONVERSION RUL → CLASSE
# ─────────────────────────────────────────
def rul_to_classe(rul_array, seuils=RUL_SEUILS):
    classes = np.full(len(rul_array), 2, dtype=int)
    classes[rul_array <= seuils['degrade']]  = 1
    classes[rul_array <= seuils['critique']] = 0
    return classes

# ─────────────────────────────────────────
# 7. ÉVALUATION
# ─────────────────────────────────────────
def evaluer_classification(modele, X_test, y_test_rul, nom_modele, seuils=RUL_SEUILS):
    y_pred_rul = modele.predict(X_test).flatten()
    y_true_cls = rul_to_classe(y_test_rul, seuils)
    y_pred_cls = rul_to_classe(y_pred_rul, seuils)

    rmse = np.sqrt(mean_squared_error(y_test_rul, y_pred_rul))
    mae  = np.mean(np.abs(y_test_rul - y_pred_rul))

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

    acc = (y_true_cls == y_pred_cls).mean()
    return {
        "Modèle":      nom_modele,
        "RMSE":        round(rmse, 2),
        "MAE":         round(mae, 2),
        "Acc_cls":     round(acc, 3),
        "y_true_cls":  y_true_cls,
        "y_pred_cls":  y_pred_cls,
        "y_pred_rul":  y_pred_rul,
        "y_true_rul":  y_test_rul,
    }

# ═════════════════════════════════════════
# PIPELINE 1 : PRÉDICTION DU RUL
# ═════════════════════════════════════════
df_train = pd.read_csv(CHEMIN_TRAIN)
df_test  = pd.read_csv(CHEMIN_TEST)
df_rul   = pd.read_csv(CHEMIN_RUL, header=None, names=['true_rul'])

df_train = calculer_rul(df_train)
df_train['RUL'] = df_train['RUL'].clip(upper=125)

df_train, df_test, features = pretraitement(df_train, df_test)

X_train_2D = df_train[features].values
y_train_2D = df_train['RUL'].values
X_test_2D, y_test_2D = preparer_test_2d(df_test, df_rul, features)

X_train_3D, y_train_3D = creer_sequences_3d(df_train, features, TIME_STEPS)
X_test_3D, y_test_3D   = preparer_test_3d(df_test, df_rul, features, TIME_STEPS)

print(f"\nFormes → Train 2D: {X_train_2D.shape} | Test 2D: {X_test_2D.shape}")
print(f"         Train 3D: {X_train_3D.shape} | Test 3D: {X_test_3D.shape}\n")

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

print("Entraînement LSTM (version ESP32 lite)...")
# ── lstm = modèle batch=1 pour export TFLite
# ── lstm_train = modèle flexible pour évaluation Python
lstm, lstm_train = entrainer_lstm(X_train_3D, y_train_3D, epochs=50)
results.append(evaluer_classification(lstm_train, X_test_3D, y_test_3D, "LSTM-Lite"))

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
# Visualisation 1 : Matrices de confusion
# ─────────────────────────────────────────
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
print("Sauvegardé → rul_classification_matrices.png")

# ─────────────────────────────────────────
# Visualisation 2 : RUL prédit vs réel
# ─────────────────────────────────────────
fig_rul, axes_rul = plt.subplots(2, 2, figsize=(14, 10))
fig_rul.suptitle('RUL Prédit vs Réel avec Zones de Classification',
                 fontsize=13, fontweight='bold')
for ax, res in zip(axes_rul.flatten(), results):
    y_true  = res['y_true_rul']
    y_pred  = res['y_pred_rul']
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax.axhspan(0,                      RUL_SEUILS['critique'], alpha=0.12,
               color=CLASSE_COLORS[0], label=f"Critique (≤{RUL_SEUILS['critique']})")
    ax.axhspan(RUL_SEUILS['critique'], RUL_SEUILS['degrade'],  alpha=0.10,
               color=CLASSE_COLORS[1], label=f"Dégradé (≤{RUL_SEUILS['degrade']})")
    ax.axhspan(RUL_SEUILS['degrade'],  max_val,                alpha=0.08,
               color=CLASSE_COLORS[2], label=f"Sain (>{RUL_SEUILS['degrade']})")
    ax.axhline(RUL_SEUILS['critique'], color=CLASSE_COLORS[0],
               linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(RUL_SEUILS['degrade'],  color=CLASSE_COLORS[1],
               linestyle='--', linewidth=1.2, alpha=0.7)
    colors_pts = [CLASSE_COLORS[c] for c in res['y_true_cls']]
    ax.scatter(y_true, y_pred, c=colors_pts, alpha=0.65, s=40,
               edgecolors='white', linewidths=0.4)
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Parfait')
    ax.set_xlabel('RUL réel'); ax.set_ylabel('RUL prédit')
    ax.set_title(f"{res['Modèle']}  RMSE={res['RMSE']}  Acc={res['Acc_cls']:.0%}",
                 fontweight='bold')
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('rul_pred_vs_real_zones.png', dpi=150, bbox_inches='tight')
print("Sauvegardé → rul_pred_vs_real_zones.png")

# ─────────────────────────────────────────
# Visualisation 3 : Distribution des classes
# ─────────────────────────────────────────
fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
fig_dist.suptitle('Distribution des Classes Prédites vs Réelles',
                  fontsize=12, fontweight='bold')
x       = np.arange(len(CLASSE_NOMS))
n_models = len(results)
width   = 0.15
offsets = np.linspace(-(n_models * width) / 2, (n_models * width) / 2, n_models + 1)
true_counts = [(results[0]['y_true_cls'] == c).sum() for c in range(3)]
ax_dist.bar(x + offsets[0], true_counts, width, label='Réel',
            color='#2c3e50', alpha=0.85, edgecolor='white')
for i, res in enumerate(results):
    pred_counts = [(res['y_pred_cls'] == c).sum() for c in range(3)]
    ax_dist.bar(x + offsets[i + 1], pred_counts, width,
                label=res['Modèle'], alpha=0.80, edgecolor='white')
ax_dist.set_xticks(x)
ax_dist.set_xticklabels(CLASSE_NOMS, fontsize=11)
ax_dist.set_ylabel('Nombre de moteurs')
ax_dist.set_xlabel('Classe RUL')
ax_dist.legend(fontsize=9)
ax_dist.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('rul_class_distribution.png', dpi=150, bbox_inches='tight')
print("Sauvegardé → rul_class_distribution.png")

# ═════════════════════════════════════════
# EXPORT LSTM → TFLite → C header ESP32
# ═════════════════════════════════════════
print("\n[Export] Conversion LSTM → TFLite INT8 pur (sans SELECT_TF_OPS)...")

def representative_dataset():
    # batch_shape=1 → on passe 1 sample à la fois
    for i in range(min(200, len(X_train_3D))):
        yield [X_train_3D[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(lstm)  # ← modèle batch=1
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
# ✅ INT8 pur — pas de SELECT_TF_OPS → Flash minimal sur ESP32
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

tflite_size_kb = len(tflite_model) / 1024
print(f"  Taille .tflite : {tflite_size_kb:.1f} Ko")
print(f"  {'✓  Taille acceptable pour ESP32' if tflite_size_kb <= 150 else '⚠️  Modèle encore grand'}")

with open("cerveau_lstm.tflite", "wb") as f:
    f.write(tflite_model)
print("  Sauvegardé → cerveau_lstm.tflite")

# ── Export scaler values ──────────────────────────────────────
scaler_cloud = joblib.load('scaler_cloud.pkl')
scaler_min   = scaler_cloud.data_min_
scaler_range = scaler_cloud.data_range_
print(f"\n  SCALER_MIN   = {list(np.round(scaler_min, 4))}")
print(f"  SCALER_RANGE = {list(np.round(scaler_range, 4))}")

# ── Génération du fichier C header ───────────────────────────
with open("cerveau_lstm.tflite", "rb") as f:
    tflite_content = f.read()

hex_lines = [
    ", ".join([f"0x{b:02x}" for b in tflite_content[i:i + 12]])
    for i in range(0, len(tflite_content), 12)
]
hex_array = ",\n  ".join(hex_lines)

with open("model_esp32.h", "w", encoding="utf-8") as f:
    f.write("// Fichier généré automatiquement pour l'ESP32\n")
    f.write(f"// Modèle LSTM lite : LSTM(24, unroll=True) + Dense(16) + Dense(1)\n")
    f.write(f"// TIME_STEPS={TIME_STEPS}  NB_FEATURES={len(features)}\n")
    f.write(f"// Seuils RUL : Critique≤{RUL_SEUILS['critique']}  "
            f"Dégradé≤{RUL_SEUILS['degrade']}  Sain>{RUL_SEUILS['degrade']}\n")
    f.write("#include <pgmspace.h>\n\n")
    f.write(f"#define RUL_SEUIL_CRITIQUE {RUL_SEUILS['critique']}\n")
    f.write(f"#define RUL_SEUIL_DEGRADE  {RUL_SEUILS['degrade']}\n\n")
    f.write(f"const unsigned char cerveau_lstm_tflite[] PROGMEM = {{\n  {hex_array}\n}};\n")
    f.write(f"const int cerveau_lstm_tflite_len = {len(tflite_content)};\n\n")
    f.write("inline int rul_to_classe(float rul) {\n")
    f.write(f"  if (rul <= RUL_SEUIL_CRITIQUE) return 0;\n")
    f.write(f"  if (rul <= RUL_SEUIL_DEGRADE)  return 1;\n")
    f.write(f"  return 2;\n")
    f.write("}\n")

print(f"\n  model_esp32.h généré ({len(tflite_content)} octets)")
print(f"  ✓ #define TIME_STEPS {TIME_STEPS} dans le .ino Arduino")
print(f"  ✓ kTensorArenaSize = 45 * 1024 dans le .ino Arduino")

# ═════════════════════════════════════════
# PIPELINE 2 : DÉTECTION DU TYPE DE PANNE
# ═════════════════════════════════════════
print("\n[Pipeline 2] Détection du type de panne (GMM + RF)...")

def extraire_features_single(df_u):
    all_sensors = list(set(HPC_SENSORS + FAN_SENSORS))
    df_u = df_u.sort_values('time_cycles').reset_index(drop=True)
    t   = np.arange(len(df_u))
    n   = len(df_u)
    n20 = max(int(n * 0.2), 5)

    def norm_slope(v):
        if len(v) < 2: return 0.0
        r = float(v.max()) - float(v.min())
        if r < 1e-6: return 0.0
        slope, *_ = linregress(t, v)
        return slope / r

    def end_delta(v):
        return float(v[-n20:].mean()) - float(v[:n20].mean())

    row = {}
    for s in all_sensors:
        v = df_u[s].values
        row[f'{s}_ns']    = norm_slope(v)
        row[f'{s}_delta'] = end_delta(v)

    row['T30_vs_Nf']  = norm_slope(df_u['sensor_3'].values)  - norm_slope(df_u['sensor_8'].values)
    row['NRc_vs_NRf'] = norm_slope(df_u['sensor_11'].values) - norm_slope(df_u['sensor_17'].values)
    row['T30_vs_T24'] = norm_slope(df_u['sensor_3'].values)  - norm_slope(df_u['sensor_2'].values)
    row['Nc_vs_Nf']   = norm_slope(df_u['sensor_9'].values)  - norm_slope(df_u['sensor_8'].values)
    row['hpc_score']  = row['T30_vs_Nf'] + row['NRc_vs_NRf'] + row['T30_vs_T24'] + row['Nc_vs_Nf']
    row['lifetime']   = n
    return row

def extraire_features_batch(df, units=None):
    if units is None:
        units = df['unit_number'].unique()
    rows = []
    for unit in units:
        df_u = df[df['unit_number'] == unit]
        row  = extraire_features_single(df_u)
        row['unit_number'] = unit
        rows.append(row)
    return pd.DataFrame(rows).set_index('unit_number')

feat_train    = extraire_features_batch(df_train)
feat_test     = extraire_features_batch(df_test)
ALL_FEAT_COLS = list(feat_train.columns)

# GMM
scaler_gmm = StandardScaler()
X_gmm = scaler_gmm.fit_transform(feat_train[ALL_FEAT_COLS])
gmm = GaussianMixture(n_components=2, covariance_type='full',
                      random_state=42, n_init=5)
gmm.fit(X_gmm)
y_train_raw = gmm.predict(X_gmm)

hpc_idx = feat_train[['sensor_3_ns', 'sensor_11_ns']].mean(axis=1).groupby(
    pd.Series(y_train_raw, index=feat_train.index)).mean().idxmax()
label_map = {hpc_idx: 0, 1 - hpc_idx: 1}
y_train   = np.array([label_map[l] for l in y_train_raw])

sil            = silhouette_score(X_gmm, y_train)
gmm_confidence = gmm.predict_proba(X_gmm).max(axis=1)
print(f"  GMM → Score Silhouette={sil:.3f}  Confiance moy.={gmm_confidence.mean():.1%}")

# RF classifieur
scaler_clf  = StandardScaler()
X_clf_train = scaler_clf.fit_transform(feat_train[ALL_FEAT_COLS])
X_clf_test  = scaler_clf.transform(feat_test[ALL_FEAT_COLS])

clf       = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_clf_train, y_train, cv=skf, scoring='accuracy')
clf.fit(X_clf_train, y_train)
y_pred_test  = clf.predict(X_clf_test)
y_prob_train = clf.predict_proba(X_clf_train)
print(f"  RF  → Accuracy CV={cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

# Sauvegarde
print("\n[Sauvegarde] Modèles...")
joblib.dump(gmm,        'gmm_fault_model.pkl')
joblib.dump(clf,        'rf_fault_classifier.pkl')
joblib.dump(scaler_clf, 'scaler_fault_classifier.pkl')
joblib.dump(scaler_gmm, 'scaler_fault_gmm.pkl')
print("  4 fichiers .pkl sauvegardés.")

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
print(f"\n  Seuils RUL :")
print(f"    Critique : RUL ≤ {RUL_SEUILS['critique']}  (maintenance immédiate)")
print(f"    Dégradé  : RUL ≤ {RUL_SEUILS['degrade']}  (surveillance renforcée)")
print(f"    Sain     : RUL > {RUL_SEUILS['degrade']}  (fonctionnement normal)")
print("\n" + "=" * 60)
print(f"  RÉCAPITULATIF ARDUINO")
print("=" * 60)
print(f"    #define TIME_STEPS        {TIME_STEPS}")
print(f"    #define NB_FEATURES       {len(features)}")
print(f"    constexpr int kTensorArenaSize = 45 * 1024;")
print("=" * 60)
print("  Pipeline terminé avec succès !")
print("=" * 60)

# ─────────────────────────────────────────
# INFÉRENCE EDGE
# ─────────────────────────────────────────
def predire_fault_mode(df_moteur, clf_model, scaler_model, feat_cols):
    row  = extraire_features_single(df_moteur)
    X    = np.array([[row[c] for c in feat_cols]])
    Xs   = scaler_model.transform(X)
    fid  = clf_model.predict(Xs)[0]
    prob = clf_model.predict_proba(Xs)[0]
    return {
        'fault_type': fid,
        'fault_name': FAULT_NAMES[fid],
        'confidence': prob.max(),
        'hpc_score':  row['hpc_score'],
        'proba_HPC':  prob[0],
        'proba_Fan':  prob[1],
    }

def predire_rul_et_classe(df_moteur, lstm_model, scaler_rul, feat_cols,
                           time_steps=TIME_STEPS, seuils=RUL_SEUILS):
    df_moteur = df_moteur.sort_values('time_cycles').reset_index(drop=True)
    data = df_moteur[feat_cols].values
    if len(data) >= time_steps:
        seq = data[-time_steps:]
    else:
        pad = np.zeros((time_steps - len(data), len(feat_cols)))
        seq = np.vstack([pad, data])
    # ── batch_shape=1 → reshape (1, time_steps, nb_features)
    X_seq      = seq.reshape(1, time_steps, len(feat_cols))
    rul_predit = float(lstm_model.predict(X_seq, verbose=0).flatten()[0])
    classe     = int(rul_to_classe(np.array([rul_predit]), seuils)[0])
    return {
        'rul_predit': round(rul_predit, 1),
        'classe_id':  classe,
        'classe_nom': CLASSE_NOMS[classe],
        'alerte':     classe == 0,
    }

print(f"\n[BONUS] Inférence Edge sur 5 moteurs du test :")
print(f"\n  {'Moteur':<10} {'Type de panne':<22} {'Conf':>7}  "
      f"{'RUL prédit':>10}  {'Classe RUL':>12}  {'Alerte':>7}")
print(f"  {'-' * 75}")

for uid in [1, 25, 50, 75, 97]:
    df_u    = df_test[df_test['unit_number'] == uid].copy()
    fault   = predire_fault_mode(df_u, clf, scaler_clf, ALL_FEAT_COLS)
    # ── utilise lstm (batch=1) pour l'inférence edge ──
    rul_res = predire_rul_et_classe(df_u, lstm, scaler_clf, features)
    alerte_str = "⚠ OUI" if rul_res['alerte'] else "—"
    print(f"  #{uid:<9} {fault['fault_name']:<22} {fault['confidence']:>6.1%}  "
          f"{rul_res['rul_predit']:>10.1f}  {rul_res['classe_nom']:>12}  {alerte_str:>7}")