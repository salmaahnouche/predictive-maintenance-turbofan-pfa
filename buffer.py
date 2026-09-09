# etape2_buffer.py
from collections import deque
import numpy as np

TIME_STEPS = 30
NB_FEATURES = 16


class BufferMoteur:
    
    def __init__(self):
        self.buffer = deque(maxlen=TIME_STEPS)

    def ajouter(self, valeurs_normalisees):
       
        self.buffer.append(valeurs_normalisees)
        est_plein = len(self.buffer) == TIME_STEPS
        return est_plein, len(self.buffer)

    def get_fenetre(self):
        if len(self.buffer) < TIME_STEPS:
            return None
        return np.array(self.buffer, dtype=np.float32)


if __name__ == "__main__":
    buf = BufferMoteur()

    # On simule l'envoi de 35 lignes bidon (16 valeurs chacune)
    for i in range(35):
        ligne_bidon = [i * 0.01] * NB_FEATURES
        pret, taille = buf.ajouter(ligne_bidon)
        print(f"Ligne {i} -> buffer {taille}/{TIME_STEPS}, prêt: {pret}")

    fenetre = buf.get_fenetre()
    print("\nForme de la fenêtre finale :", fenetre.shape)
    print("Première ligne du buffer (devrait être ligne 5, car 0-4 éjectées) :", fenetre[0])
    print("Dernière ligne du buffer (devrait être ligne 34) :", fenetre[-1])