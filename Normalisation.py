# etape1_normalisation.py

NB_FEATURES = 16

RUL_SEUIL_CRITIQUE = 30.0
RUL_SEUIL_DEGRADE = 80.0

# Copiés EXACTEMENT depuis le .ino (SCALER_MIN / SCALER_RANGE)
SCALER_MIN = [
    -0.0086, -0.0006, 640.84, 1564.3,
    1377.06, 549.61, 2386.9, 9017.98,
    46.69, 517.77, 2386.93, 8099.68,
    8.1563, 388.0, 38.17, 22.8726
]
SCALER_RANGE = [
    0.0172, 0.0013, 4.2700, 51.0900,
    64.1000, 20.8800, 1.7000, 216.3700,
    1.7500, 19.6300, 1.6800, 190.8700,
    0.4142, 11.0000, 1.6800, 1.0779
]


def normaliser(valeurs_brutes):
    """Reproduit exactement le calcul du firmware : (val - min) / range, clip [0,1]."""
    out = []
    for i in range(NB_FEATURES):
        if SCALER_RANGE[i] > 0.0:
            v = (valeurs_brutes[i] - SCALER_MIN[i]) / SCALER_RANGE[i]
        else:
            v = 0.0
        out.append(min(1.0, max(0.0, v)))
    return out


def statut_depuis_rul(rul):
    """Reproduit exactement afficherResultat() du firmware."""
    if rul <= RUL_SEUIL_CRITIQUE:
        return "CRITIQUE"
    elif rul <= RUL_SEUIL_DEGRADE:
        return "DEGRADE"
    else:
        return "SAIN"


if __name__ == "__main__":
    # Test avec des valeurs bidon (juste pour vérifier que ça tourne)
    fausses_valeurs = SCALER_MIN  # si on normalise le min lui-même, on doit obtenir des 0
    print("Normalisées (devrait être ~tout à 0) :", normaliser(fausses_valeurs))

    for rul_test in [10, 50, 150]:
        print(f"RUL={rul_test} -> {statut_depuis_rul(rul_test)}")