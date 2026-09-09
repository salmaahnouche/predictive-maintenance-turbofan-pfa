import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
CHEMIN_TRAIN = r"C:\Users\HP\Desktop\pfa_proj\train_FD003_historique.csv"
CHEMIN_TEST  = r"C:\Users\HP\Desktop\pfa_proj\test_FD003_historique.csv"
CHEMIN_RUL   = r"C:\Users\HP\Desktop\pfa_proj\RUL_FD003_historique.csv"
TIME_STEPS   = 50

# Colonnes à exclure de la normalisation / features
DROP_COLS = [
    'unit_number', 'time_cycles',
    'setting_3',
    'sensor_1', 'sensor_5', 'sensor_6',
    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'
]

# ─────────────────────────────────────────
# 1. CALCUL DU RUL (train uniquement)
# ─────────────────────────────────────────
def calculer_rul(df):
    """Ajoute la colonne RUL à chaque ligne du dataframe d'entraînement."""
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
    """
    Normalise train ET test avec le même scaler.
    Retourne les features utilisées.
    """
    features = [c for c in df_train.columns if c not in DROP_COLS + ['RUL']]

    scaler = MinMaxScaler()
    df_train[features] = scaler.fit_transform(df_train[features])
    df_test[features]  = scaler.transform(df_test[features])

    joblib.dump(scaler, chemin_scaler)
    print(f"Scaler sauvegardé → {chemin_scaler}")
    return df_train, df_test, features

# ─────────────────────────────────────────
# 3. PRÉPARATION DES DONNÉES TEST (2D)
#    Le RUL test = true_rul au DERNIER cycle de chaque moteur
# ─────────────────────────────────────────
def preparer_test_2d(df_test, df_rul, features):
    """
    Extrait le dernier cycle de chaque moteur dans le test
    et aligne avec les vrais RUL.
    """
    last_cycles = df_test.groupby('unit_number').last().reset_index()
    last_cycles = last_cycles.sort_values('unit_number').reset_index(drop=True)

    X_test = last_cycles[features].values
    y_test = df_rul['true_rul'].values  # déjà trié par unit_number
    return X_test, y_test

# ─────────────────────────────────────────
# 4. SÉQUENCES 3D (LSTM / CNN)
# ─────────────────────────────────────────
def creer_sequences_3d(df, features, time_steps=30):
    """Fenêtre glissante → tenseur (N, time_steps, nb_features)."""
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
    """
    Pour chaque moteur du test, prend les DERNIERS time_steps cycles.
    Si le moteur a moins de time_steps cycles, on pad avec des zéros.
    """
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
# 5. MODÈLES
# ─────────────────────────────────────────
def regression_lineaire(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def random_forest(X_train, y_train, n_estimators=100):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train.ravel())
    return rf

def entrainer_lstm(X_train, y_train, epochs=10, batch_size=64):
    time_steps, nb_features = X_train.shape[1], X_train.shape[2]
    model = Sequential([
        LSTM(64, input_shape=(time_steps, nb_features), return_sequences=True),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.15, verbose=0)
    return model

def entrainer_cnn(X_train, y_train, epochs=10, batch_size=64):
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
# 6. ÉVALUATION
# ─────────────────────────────────────────
def evaluer_modele(modele, X_test, y_test, nom_modele):
    y_pred = modele.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = np.mean(np.abs(y_test - y_pred))
    print(f"[{nom_modele}] RMSE={rmse:.2f}  MAE={mae:.2f}")
    return {"Modèle": nom_modele, "RMSE": round(rmse, 2), "MAE": round(mae, 2)}

# ─────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────
print("=" * 50)
print("  Démarrage du pipeline Cloud - RUL FD003")
print("=" * 50)

# Chargement
df_train = pd.read_csv(CHEMIN_TRAIN)
df_test  = pd.read_csv(CHEMIN_TEST)
df_rul   = pd.read_csv(CHEMIN_RUL)

# Calcul RUL train
df_train = calculer_rul(df_train)

# Preprocessing (normalisation partagée)
df_train, df_test, features = pretraitement(df_train, df_test)

# Données 2D
X_train_2D = df_train[features].values
y_train_2D = df_train['RUL'].values
X_test_2D, y_test_2D = preparer_test_2d(df_test, df_rul, features)

# Données 3D
X_train_3D, y_train_3D = creer_sequences_3d(df_train, features, TIME_STEPS)
X_test_3D, y_test_3D   = preparer_test_3d(df_test, df_rul, features, TIME_STEPS)

print(f"\nFormes → Train 2D: {X_train_2D.shape} | Test 2D: {X_test_2D.shape}")
print(f"         Train 3D: {X_train_3D.shape} | Test 3D: {X_test_3D.shape}\n")

results = []

# A. Régression Linéaire
print("Entraînement Régression Linéaire...")
lr = regression_lineaire(X_train_2D, y_train_2D)
results.append(evaluer_modele(lr, X_test_2D, y_test_2D, "LR"))

# B. Random Forest
print("Entraînement Random Forest...")
rf = random_forest(X_train_2D, y_train_2D)
results.append(evaluer_modele(rf, X_test_2D, y_test_2D, "RF"))

# C. CNN
print("Entraînement CNN...")
cnn = entrainer_cnn(X_train_3D, y_train_3D, epochs=30)
results.append(evaluer_modele(cnn, X_test_3D, y_test_3D, "CNN"))

# D. LSTM
print("Entraînement LSTM...")
lstm = entrainer_lstm(X_train_3D, y_train_3D, epochs=30)
results.append(evaluer_modele(lstm, X_test_3D, y_test_3D, "LSTM"))

# Résumé
print("\n" + "=" * 40)
print("  RÉSULTATS FINAUX")
print("=" * 40)
df_results = pd.DataFrame(results).sort_values("RMSE")
print(df_results.to_string(index=False))