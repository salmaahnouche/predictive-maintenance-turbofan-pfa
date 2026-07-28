import pandas as pd
import os

train_file = r"C:\Users\HP\Desktop\pfa_proj\train_FD003.csv"
test_file = r"C:\Users\HP\Desktop\pfa_proj\test_FD003.csv"
RUL_file = r"C:\Users\HP\Desktop\pfa_proj\RUL_FD003.csv"


def separer_premiers_moteurs(nom_fichier):
    print(f"Chargement du fichier : {nom_fichier}...")
    df = pd.read_csv(nom_fichier)
    
    if 'unit_number' not in df.columns:
        raise ValueError("La colonne 'unit_number' est introuvable dans le fichier CSV.")

    trois_premiers_moteurs = df['unit_number'].unique()[:3]
    print(f"Moteurs sélectionnés pour l'extraction : {trois_premiers_moteurs}")

    dossier_destination = os.path.dirname(nom_fichier)
    
    # NOUVEAU : Récupérer le nom du fichier d'origine (ex: "FD001_test") pour l'inclure dans le nom de sortie
    nom_base = os.path.splitext(os.path.basename(nom_fichier))[0]

    for moteur in trois_premiers_moteurs:
        df_moteur = df[df['unit_number'] == moteur]
        
        # NOUVEAU : Le fichier s'appellera par exemple "FD001_test_moteur_1.csv"
        nom_fichier_sortie = os.path.join(dossier_destination, f'{nom_base}_moteur_{moteur}.csv')
        
        df_moteur.to_csv(nom_fichier_sortie, index=False)
        print(f"-> Fichier généré : {nom_fichier_sortie} ({len(df_moteur)} lignes enregistrées)")

# Tu peux maintenant exécuter les deux à la suite sans conflit !
separer_premiers_moteurs(train_file)
separer_premiers_moteurs(test_file)