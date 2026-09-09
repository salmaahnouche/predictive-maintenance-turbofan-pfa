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

# Sensor mappings (C-MAPSS FD003)
# sensor_2  = T24  (LPC outlet temp)      → Fan/LPC
# sensor_3  = T30  (HPC outlet temp)      → HPC ★
# sensor_4  = T50  (LPT outlet temp)      → global
# sensor_8  = Nf   (Fan speed physical)   → Fan ★
# sensor_9  = Nc   (Core speed physical)  → HPC
# sensor_11 = NRc  (Corrected core speed) → HPC ★
# sensor_13 = (fan related)               → Fan
# sensor_17 = NRf  (Corrected fan speed)  → Fan ★
HPC_SENSORS  = ['sensor_3', 'sensor_11', 'sensor_9']
FAN_SENSORS  = ['sensor_8', 'sensor_17', 'sensor_2', 'sensor_13']
FAULT_NAMES  = {0: 'HPC Degradation', 1: 'Fan Degradation'}
FAULT_COLORS = {0: '#e74c3c', 1: '#3498db'}

# ─────────────────────────────────────────
# SEUILS RUL → CLASSIFICATION
# ─────────────────────────────────────────
# Trois classes ordinales basées sur le RUL prédit :
#   Critique : RUL ≤ 30  → maintenance immédiate requise
#   Dégradé  : 30 < RUL ≤ 80 → surveiller de près
#   Sain     : RUL > 80  → fonctionnement normal
RUL_SEUILS = {
    'critique': 30,
    'degrade':  80,
}
CLASSE_NOMS   = ['Critique', 'Dégradé', 'Sain']
CLASSE_COLORS = ['#e74c3c', '#f39c12', '#27ae60']

# ─────────────────────────────────────────
# 1. CALCUL DU RUL (train uniquement)
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
# 6. CONVERSION RUL → CLASSE
# ─────────────────────────────────────────
def rul_to_classe(rul_array, seuils=RUL_SEUILS):
    """
    Convertit des valeurs RUL continues en classes ordinales.
    0 = Critique  (RUL ≤ seuils['critique'])
    1 = Dégradé   (seuils['critique'] < RUL ≤ seuils['degrade'])
    2 = Sain      (RUL > seuils['degrade'])
    """
    classes = np.full(len(rul_array), 2, dtype=int)           # défaut : Sain
    classes[rul_array <= seuils['degrade']]  = 1               # Dégradé
    classes[rul_array <= seuils['critique']] = 0               # Critique
    return classes

# ─────────────────────────────────────────
# 7. ÉVALUATION : RÉGRESSION + CLASSIFICATION
# ─────────────────────────────────────────
def evaluer_classification(modele, X_test, y_test_rul, nom_modele, seuils=RUL_SEUILS):
    """
    Prédit le RUL continu, le convertit  en classes via les seuils,
    puis évalue à la fois la régression (RMSE/MAE) et la classification
    (accuracy, precision, recall, F1 par classe).
    """
    y_pred_rul = modele.predict(X_test).flatten()

    # Conversion en classes
    y_true_cls = rul_to_classe(y_test_rul, seuils)
    y_pred_cls = rul_to_classe(y_pred_rul, seuils)

    # Métriques régression
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

# Chargement
df_train = pd.read_csv(CHEMIN_TRAIN)
df_test  = pd.read_csv(CHEMIN_TEST)
df_rul   = pd.read_csv(CHEMIN_RUL, header=None, names=['true_rul'])

# Calcul et écrêtage RUL
df_train = calculer_rul(df_train)
df_train['RUL'] = df_train['RUL'].clip(upper=125)

# Preprocessing (normalisation partagée train/test)
df_train, df_test, features = pretraitement(df_train, df_test)

# Données 2D (LR, RF)
X_train_2D = df_train[features].values
y_train_2D = df_train['RUL'].values
X_test_2D, y_test_2D = preparer_test_2d(df_test, df_rul, features)

# Données 3D (CNN, LSTM)
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
# Visualisation 1 : Matrices de confusion (4 modèles)
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
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='YlOrRd',
                xticklabels=CLASSE_NOMS,
                yticklabels=CLASSE_NOMS,
                linewidths=0.5)
    ax.set_title(f"{res['Modèle']}  (acc={res['Acc_cls']:.0%})", fontweight='bold')
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')

plt.tight_layout()
plt.savefig('rul_classification_matrices.png', dpi=150, bbox_inches='tight')
print("    Sauvegardé → rul_classification_matrices.png")

# ─────────────────────────────────────────
# Visualisation 2 : RUL prédit vs réel avec zones colorées
# ─────────────────────────────────────────
fig_rul, axes_rul = plt.subplots(2, 2, figsize=(14, 10))
fig_rul.suptitle('RUL Prédit vs Réel avec Zones de Classification',
                 fontsize=13, fontweight='bold')
axes_flat = axes_rul.flatten()

for ax, res in zip(axes_flat, results):
    y_true = res['y_true_rul']
    y_pred = res['y_pred_rul']
    max_val = max(y_true.max(), y_pred.max()) * 1.05

    # Zones colorées par classe
    ax.axhspan(0,                         RUL_SEUILS['critique'], alpha=0.12,
               color=CLASSE_COLORS[0], label=f"Critique (≤{RUL_SEUILS['critique']})")
    ax.axhspan(RUL_SEUILS['critique'],    RUL_SEUILS['degrade'],  alpha=0.10,
               color=CLASSE_COLORS[1], label=f"Dégradé (≤{RUL_SEUILS['degrade']})")
    ax.axhspan(RUL_SEUILS['degrade'],     max_val,                alpha=0.08,
               color=CLASSE_COLORS[2], label=f"Sain (>{RUL_SEUILS['degrade']})")

    # Lignes de seuil
    ax.axhline(RUL_SEUILS['critique'], color=CLASSE_COLORS[0], linestyle='--',
               linewidth=1.2, alpha=0.7)
    ax.axhline(RUL_SEUILS['degrade'],  color=CLASSE_COLORS[1], linestyle='--',
               linewidth=1.2, alpha=0.7)

    # Scatter : couleur selon classe réelle
    colors_pts = [CLASSE_COLORS[c] for c in res['y_true_cls']]
    ax.scatter(y_true, y_pred, c=colors_pts, alpha=0.65, s=40,
               edgecolors='white', linewidths=0.4)

    # Droite parfaite
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Parfait')

    ax.set_xlabel('RUL réel')
    ax.set_ylabel('RUL prédit')
    ax.set_title(f"{res['Modèle']}  RMSE={res['RMSE']}  Acc={res['Acc_cls']:.0%}",
                 fontweight='bold')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('rul_pred_vs_real_zones.png', dpi=150, bbox_inches='tight')
print("    Sauvegardé → rul_pred_vs_real_zones.png")

# ─────────────────────────────────────────
# Visualisation 3 : Distribution des classes par modèle
# ─────────────────────────────────────────
fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
fig_dist.suptitle('Distribution des Classes Prédites vs Réelles',
                  fontsize=12, fontweight='bold')

x        = np.arange(len(CLASSE_NOMS))
n_models = len(results)
width    = 0.15
offsets  = np.linspace(-(n_models * width) / 2, (n_models * width) / 2, n_models + 1)

# Barres réelles (utilise le premier résultat, y_true identique pour même dataset)
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
print("    Sauvegardé → rul_class_distribution.png")

# ─────────────────────────────────────────
# Export LSTM → TFLite → C header for ESP32
# ─────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(lstm)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("cerveau_lstm.tflite", "wb") as f:
    f.write(tflite_model)

with open("cerveau_lstm.tflite", "rb") as f:
    tflite_content = f.read()

hex_lines = [
    ", ".join([f"0x{b:02x}" for b in tflite_content[i:i + 12]])
    for i in range(0, len(tflite_content), 12)
]
hex_array = ",\n  ".join(hex_lines)

with open("model_esp32.h", "w") as f:
    f.write("// Fichier généré automatiquement pour l'ESP32\n")
    f.write(f"// Seuils RUL embarqués : Critique≤{RUL_SEUILS['critique']}  "
            f"Dégradé≤{RUL_SEUILS['degrade']}  Sain>{RUL_SEUILS['degrade']}\n")
    f.write("#include <pgmspace.h>\n\n")
    f.write(f"#define RUL_SEUIL_CRITIQUE {RUL_SEUILS['critique']}\n")
    f.write(f"#define RUL_SEUIL_DEGRADE  {RUL_SEUILS['degrade']}\n\n")
    f.write(f"const unsigned char cerveau_lstm_tflite[] PROGMEM = {{\n  {hex_array}\n}};\n")
    f.write(f"const int cerveau_lstm_tflite_len = {len(tflite_content)};\n\n")
    f.write("// Fonction de classification RUL → classe (0=Critique, 1=Dégradé, 2=Sain)\n")
    f.write("inline int rul_to_classe(float rul) {\n")
    f.write(f"  if (rul <= RUL_SEUIL_CRITIQUE) return 0;\n")
    f.write(f"  if (rul <= RUL_SEUIL_DEGRADE)  return 1;\n")
    f.write(f"  return 2;\n")
    f.write("}\n")

print("\nExport ESP32 : cerveau_lstm.tflite + model_esp32.h générés.")

# ═════════════════════════════════════════
# PIPELINE 2 : DÉTECTION DE PANNES
# ═════════════════════════════════════════
print("\n" + "=" * 50)
print("  Pipeline Détection de Pannes - FD003")
print("=" * 50)


# ═════════════════════════════════════════
# INFÉRENCE EDGE (nouveau moteur en temps réel)
# ═════════════════════════════════════════
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
        slope, _, _, _, _ = linregress(t, v)
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
    """
    Prédit le RUL d'un moteur puis retourne la classe de criticité.
    """
    df_moteur = df_moteur.sort_values('time_cycles').reset_index(drop=True)
    data = df_moteur[feat_cols].values
    if len(data) >= time_steps:
        seq = data[-time_steps:]
    else:
        pad = np.zeros((time_steps - len(data), len(feat_cols)))
        seq = np.vstack([pad, data])
    X_seq = seq.reshape(1, time_steps, len(feat_cols))
    rul_predit = float(lstm_model.predict(X_seq, verbose=0).flatten()[0])
    classe     = int(rul_to_classe(np.array([rul_predit]), seuils)[0])
    return {
        'rul_predit': round(rul_predit, 1),
        'classe_id':  classe,
        'classe_nom': CLASSE_NOMS[classe],
        'alerte':     classe == 0,
    }


print(f"\n[BONUS] Inférence Edge complète sur 5 moteurs du test:")
print(f"\n  {'Moteur':<10} {'Type de panne':<22} {'Conf':>7}  "
      f"{'RUL prédit':>10}  {'Classe RUL':>12}  {'Alerte':>7}")
print(f"  {'-' * 75}")

for uid in [1, 25, 50, 75, 97]:
    df_u = df_test[df_test['unit_number'] == uid].copy()

    # Fault mode
    fault = predire_fault_mode(df_u, clf, scaler_clf, ALL_FEAT_COLS)

    # RUL + classe
    rul_res = predire_rul_et_classe(df_u, lstm, scaler_clf, features)

    alerte_str = "⚠ OUI" if rul_res['alerte'] else "—"
    print(f"  #{uid:<9} {fault['fault_name']:<22} {fault['confidence']:>6.1%}  "
          f"{rul_res['rul_predit']:>10.1f}  {rul_res['classe_nom']:>12}  {alerte_str:>7}")