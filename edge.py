import pandas as pd
import serial
import time
import json
import re
import requests
from datetime import datetime

# --- CONFIGURATION ---
PORT_SERIE = 'COM9'
BAUD_RATE = 115200

# L'IP de ton PC Cloud (celui avec FastAPI)
IP_PC_CLOUD = "127.0.0.1" 
URL_FASTAPI = f"http://{IP_PC_CLOUD}:8000/api/maintenance/upload"

DROP_COLS = [
    'unit_number', 'time_cycles', 'setting_3',
    'sensor_1', 'sensor_5', 'sensor_6',
    'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'
]
DOSSIER = r"C:\Users\HP\Desktop\pfa_proj"
FICHIER_JSON = f"{DOSSIER}\\resultats_rul.json"

df1 = pd.read_csv(r"C:\Users\HP\Desktop\pfa_proj\test_FD003_moteur_1.csv")
df2 = pd.read_csv(r"C:\Users\HP\Desktop\pfa_proj\test_FD003_moteur_2.csv")
df3 = pd.read_csv(r"C:\Users\HP\Desktop\pfa_proj\test_FD003_moteur_3.csv")

max_lignes = max(len(df1), len(df2), len(df3))

resultats = {
    "session": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "moteurs": {
        "1": [],
        "2": [],
        "3": []
    }
}

def parser_reponse(ligne_texte):
    pattern = r"Moteur\s+(\d+)\s*\|\s*RUL:\s*([\d.]+)\s*->\s*(\w+)"
    match = re.search(pattern, ligne_texte)
    if match:
        return {
            "moteur_id": int(match.group(1)),
            "rul":       float(match.group(2)),
            "statut":    match.group(3)
        }
    return None

# ◄ MODIFICATION ICI : On ajoute le paramètre 'capteurs_dict'
def lire_et_enregistrer(port, index_ligne, capteurs_dict, timeout_ms=200):
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if port.in_waiting > 0:
            raw = port.readline()
            texte = raw.decode('utf-8', errors='ignore').strip()
            if texte:
                print(f"   [RÉPONSE EDGE IA] : {texte}")
                parsed = parser_reponse(texte)
                if parsed:
                    mid = str(parsed["moteur_id"])
                    resultats["moteurs"][mid].append({
                        "ligne":    index_ligne,
                        "capteurs": capteurs_dict,  # ◄ AJOUT DES CAPTEURS DANS LE JSON
                        "rul":      parsed["rul"],
                        "statut":   parsed["statut"]
                    })
        else:
            time.sleep(0.01)

def sauvegarder_json():
    with open(FICHIER_JSON, 'w', encoding='utf-8') as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print(f"\n💾 JSON sauvegardé en local : {FICHIER_JSON}")

def envoyer_au_cloud():
    print(f"\n🚀 Connexion au serveur Cloud FastAPI ({URL_FASTAPI})...")
    try:
        reponse = requests.post(URL_FASTAPI, json=resultats, timeout=5)
        if reponse.status_code == 200:
            print("✅ Succès ! Le fichier JSON a été transféré.")
        else:
            print(f"❌ Échec de réception. Code erreur HTTP : {reponse.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Impossible de joindre le serveur Cloud : {e}")

# --- EXECUTION PRINCIPALE ---
try:
    print(f"Connexion à l'ESP32 sur le port {PORT_SERIE}...")
    esp32 = serial.Serial(PORT_SERIE, BAUD_RATE, timeout=0.1, dsrdtr=False, rtscts=False)
    esp32.dtr = False
    esp32.rts = False
    time.sleep(2.5)
    esp32.reset_input_buffer()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if esp32.in_waiting > 0:
            msg_boot = esp32.readline().decode('utf-8', errors='ignore').strip()
            if msg_boot:
                print(f"[BOOT ESP32] : {msg_boot}")
        time.sleep(0.01)

    print(" Connexion établie ! Début de l'envoi temps réel...\n")

    moteurs = [(1, df1), (2, df2), (3, df3)]

    for index_ligne in range(max_lignes):
        for id_moteur, df in moteurs:
            if index_ligne < len(df):
# 1. Envoi Série (format CSV) - On filtre les 16 features pour le CNN
                features_cols = [c for c in df.columns if c not in DROP_COLS]
                valeurs_liste = df.iloc[index_ligne][features_cols].astype(str).tolist()
                msg = f"{id_moteur},{','.join(valeurs_liste)}\n"
                esp32.write(msg.encode('utf-8'))
                print(f"➤ Ligne {index_ligne} envoyée (Moteur {id_moteur})")

                # 2. ◄ MODIFICATION ICI : On extrait la ligne sous forme de dictionnaire {nom_colonne: valeur}
                capteurs_dict = df.iloc[index_ligne].to_dict()

                time.sleep(0.1)
                
                # 3. ◄ MODIFICATION ICI : On passe ce dictionnaire à la fonction d'enregistrement
                lire_et_enregistrer(esp32, index_ligne, capteurs_dict, timeout_ms=150)

except serial.SerialException as e:
    print(f" Erreur port série : {e}")
except KeyboardInterrupt:
    print("\nSimulation arrêtée manuellement.")
finally:
    if 'esp32' in locals() and esp32.is_open:
        esp32.close()
        print("Port série fermé.")
    
    sauvegarder_json()
    envoyer_au_cloud()