# etape3_lecture_csv.py
import pandas as pd

NB_FEATURES = 16

DROP_COLS = [
    'unit_number', 'time_cycles', 'setting_3',
    'sensor_1', 'sensor_5', 'sensor_6',
    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'
]


def charger_features(csv_path):
    df = pd.read_csv(csv_path)
    features_cols = [c for c in df.columns if c not in DROP_COLS]

    if len(features_cols) != NB_FEATURES:
        raise ValueError(
            f"Attendu {NB_FEATURES} colonnes, trouvé {len(features_cols)} : {features_cols}"
        )

    return df, features_cols


if __name__ == "__main__":
    # Adapte le chemin à ton fichier réel
    CSV_PATH = r"C:\Users\HP\Desktop\pfa_proj\test_FD003_moteur_1.csv"

    df, features_cols = charger_features(CSV_PATH)

    print(f"Nombre de lignes dans le CSV : {len(df)}")
    print(f"Colonnes gardées ({len(features_cols)}) :")
    print(features_cols)

    print("\nPremière ligne (brute, avant normalisation) :")
    premiere_ligne = df.iloc[0][features_cols].tolist()
    print(premiere_ligne)