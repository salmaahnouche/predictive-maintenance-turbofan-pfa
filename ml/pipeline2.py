import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
import joblib

# ─────────────────────────────────────────
# CONFIGURATION ET CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
HPC_SENSORS = ['sensor_3', 'sensor_11', 'sensor_9']
FAN_SENSORS  = ['sensor_8', 'sensor_17', 'sensor_2', 'sensor_13']

FAULT_NAMES  = {0: 'HPC Degradation', 1: 'Fan Degradation'}
FAULT_COLORS = {0: '#e74c3c', 1: '#3498db'}
RUL_SEUILS   = {'critique': 30, 'degrade': 80}

# Noms des 26 colonnes du dataset CMAPSS
cols = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
cols += [f'sensor_{i}' for i in range(1, 22)]

print("Chargement des données d'entraînement et de test...")

def charger_csv(path, cols):
    """
    Chargement robuste : détecte automatiquement si le fichier est séparé
    par virgule ou par espace, et gère la présence ou non d'un header.
    """
    # Lire la première ligne pour détecter le séparateur
    with open(path, 'r') as f:
        first_line = f.readline().strip()

    # Détection du séparateur
    sep = ',' if first_line.count(',') > first_line.count(' ') else r'\s+'

    # Détection du header : si la 1ère ligne contient du texte non numérique
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

    # Si header présent, renommer les colonnes avec nos noms standards
    if has_header:
        if df.shape[1] == len(cols):
            df.columns = cols
        else:
            raise ValueError(
                f"Nombre de colonnes inattendu : {df.shape[1]} (attendu {len(cols)})\n"
                f"Colonnes trouvées : {df.columns.tolist()}"
            )

    # Conversion en numérique (force les erreurs à NaN, puis on les signale)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    nan_total = df.isna().sum().sum()
    if nan_total > 0:
        cols_with_nan = df.columns[df.isna().any()].tolist()
        print(f"    ⚠️  {nan_total} NaN après chargement dans : {cols_with_nan}")

    return df

df_train = charger_csv(r"C:\Users\HP\Desktop\pfa_proj\train_FD003_historique.csv", cols)
df_test  = charger_csv(r"C:\Users\HP\Desktop\pfa_proj\test_FD003_historique.csv",  cols)

print(f"    Train : {df_train.shape[0]} lignes, {df_train['unit_number'].nunique()} moteurs")
print(f"    Test  : {df_test.shape[0]}  lignes, {df_test['unit_number'].nunique()} moteurs")

# =========================================
# PIPELINE 2 : DÉTECTION DE PANNES
# =========================================
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
            """Nettoie un array : forward-fill puis backward-fill les NaN."""
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
        rows[unit] = row

    return pd.DataFrame(rows).T.astype(float)


feat_train = extraire_features(df_train)
feat_test  = extraire_features(df_test)
print(f"    Features extraites: {feat_train.shape[1]} par moteur")

# Vérification et rapport des NaN résiduels
nan_train = feat_train.isna().sum().sum()
nan_test  = feat_test.isna().sum().sum()
if nan_train > 0 or nan_test > 0:
    print(f"    ⚠️  NaN résiduels — train: {nan_train}, test: {nan_test} → imputation par médiane")

# ─────────────────────────────────────────
# Clustering GMM (labellisation non supervisée)
# ─────────────────────────────────────────
print("\n[3] Clustering GMM pour labellisation...")

DIFF_COLS = ['T30_vs_Nf', 'NRc_vs_NRf', 'T30_vs_T24', 'Nc_vs_Nf', 'hpc_score']

scaler_gmm = StandardScaler()
X_diff_raw = scaler_gmm.fit_transform(feat_train[DIFF_COLS])

# Imputation de sécurité (médiane) pour tout NaN résiduel
imputer_gmm = SimpleImputer(strategy='median')
X_diff = imputer_gmm.fit_transform(X_diff_raw)

if np.isnan(X_diff_raw).sum() > 0:
    print(f"    ℹ️  {np.isnan(X_diff_raw).sum()} NaN imputés avant GMM")

gmm = GaussianMixture(n_components=2, covariance_type='full',
                      random_state=42, n_init=50)
gmm_labels     = gmm.fit_predict(X_diff)
gmm_proba      = gmm.predict_proba(X_diff)
gmm_confidence = gmm_proba.max(axis=1)

# Identifier quel cluster correspond à HPC (hpc_score le plus élevé)
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

# Imputation pour le classifieur (train + test alignés)
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
cm = confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='RdBu_r',
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
            # Nettoyage NaN par interpolation linéaire
            vals = pd.Series(vals).interpolate(limit_direction='both').values
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax - vmin > 1e-6:
                vals_norm = (vals - vmin) / (vmax - vmin)
            else:
                vals_norm = np.zeros_like(vals)
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