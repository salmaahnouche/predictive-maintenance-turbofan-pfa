#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ── 1. NOUVEAU HEADER DU MODELE CNN ──────────────────────────
#include "cerveau_cnn.h"
#define TIME_STEPS        30
#define NB_FEATURES       16
#define NUMBER_OF_INPUTS  (TIME_STEPS * NB_FEATURES)  // 30 * 16 = 480
#define NB_MOTEURS        3

#define RUL_SEUIL_CRITIQUE  30.0f
#define RUL_SEUIL_DEGRADE   80.0f

// ── 2. ARENA REDUITE POUR LE CNN (40 Ko suffisent largement) ─
constexpr int kTensorArenaSize = 40 * 1024; 
static uint8_t* tensor_arena = nullptr;

// ── Scalers en Flash (PROGMEM) → 0 octets de DRAM ────────────
const float SCALER_MIN[NB_FEATURES] PROGMEM = {
    -0.0086f, -0.0006f,  640.84f,  1564.3f,
     1377.06f,  549.61f, 2386.9f,  9017.98f,
       46.69f,  517.77f, 2386.93f, 8099.68f,
        8.1563f, 388.0f,   38.17f,  22.8726f
};
const float SCALER_RANGE[NB_FEATURES] PROGMEM = {
    0.0172f,  0.0013f,  4.2700f,  51.0900f,
   64.1000f, 20.8800f,  1.7000f, 216.3700f,
    1.7500f, 19.6300f,  1.6800f, 190.8700f,
    0.4142f, 11.0000f,  1.6800f,   1.0779f
};

// ── Copies locales en RAM (chargées une fois au setup) ────────
static float scaler_min[NB_FEATURES];
static float scaler_range[NB_FEATURES];

// ── Buffers moteurs alloués dynamiquement (hors BSS) ─────────
struct EtatMoteur {
    float buffer[TIME_STEPS][NB_FEATURES];
    int   index_ecriture;
    bool  buffer_rempli;
};
static EtatMoteur* moteurs = nullptr;

// ── Objets TFLite ─────────────────────────────────────────────
const tflite::Model* model       = nullptr;
tflite::AllOpsResolver    resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input       = nullptr;
TfLiteTensor* output      = nullptr;

// ── Déclarations ─────────────────────────────────────────────
void traiterNouvelleLigne(int id_moteur, float* capteurs_bruts);
void preparerTenseur(int idx, float* sortie);
void afficherResultat(int id_moteur, float rul);

// ============================================================
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n=== Demarrage ESP32 Edge AI (Modele CNN) ===");

    // ── Charger scalers depuis Flash ───────────────────────
    for (int i = 0; i < NB_FEATURES; i++) {
        scaler_min[i]   = pgm_read_float(&SCALER_MIN[i]);
        scaler_range[i] = pgm_read_float(&SCALER_RANGE[i]);
    }
    Serial.println("Scalers charges depuis Flash OK");

    // ── Allouer tensor_arena hors DRAM ─────────────────────
    if (psramFound()) {
        tensor_arena = (uint8_t*) ps_malloc(kTensorArenaSize);
        Serial.println("Arena allouee en PSRAM");
    } else {
        tensor_arena = (uint8_t*) malloc(kTensorArenaSize);
        Serial.println("Pas de PSRAM — Arena allouee en heap");
    }

    if (!tensor_arena) {
        Serial.println("ERREUR : impossible d'allouer l'arena !");
        while (true) { delay(1000); }
    }

    // ── Allouer les buffers moteurs dynamiquement ──────────
    moteurs = (EtatMoteur*) malloc(NB_MOTEURS * sizeof(EtatMoteur));
    if (!moteurs) {
        Serial.println("ERREUR : impossible d'allouer les buffers moteurs !");
        while (true) { delay(1000); }
    }
    for (int i = 0; i < NB_MOTEURS; i++) {
        moteurs[i].index_ecriture = 0;
        moteurs[i].buffer_rempli  = false;
        memset(moteurs[i].buffer, 0, sizeof(moteurs[i].buffer));
    }
    Serial.println("Buffers moteurs OK");

    Serial.print("RAM libre apres allocs : ");
    Serial.print(ESP.getFreeHeap() / 1024);
    Serial.println(" Ko");

    // ── 3. CHARGER LE NOUVEAU MODELE CNN ───────────────────────
    model = tflite::GetModel(cerveau_cnn_tflite);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("ERREUR schema : ");
        Serial.println(model->version());
        while (true) { delay(1000); }
    }
    Serial.println("Modele OK");

    // ── Interpréteur ───────────────────────────────────────
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    Serial.print("AllocateTensors (arena=");
    Serial.print(kTensorArenaSize / 1024);
    Serial.println(" Ko)...");

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERREUR AllocateTensors !");
        Serial.print("Necessaire : ");
        Serial.print(interpreter->arena_used_bytes());
        Serial.println(" octets");
        while (true) { delay(1000); }
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    Serial.print("Arena utilisee : ");
    Serial.print(interpreter->arena_used_bytes() / 1024);
    Serial.println(" Ko OK");

    Serial.print("Input shape : [");
    for (int i = 0; i < input->dims->size; i++) {
        Serial.print(input->dims->data[i]);
        if (i < input->dims->size - 1) Serial.print(", ");
    }
    Serial.println("]");

    Serial.print("RAM restante : ");
    Serial.print(ESP.getFreeHeap() / 1024);
    Serial.println(" Ko");
    Serial.println("=========================================");
    Serial.println("ESP32 Pret : En attente des donnees...");
}

// ============================================================
void loop() {
    if (!Serial.available()) return;

    String message = Serial.readStringUntil('\n');
    message.trim();
    if (message.length() == 0) return;

    int firstComma = message.indexOf(',');
    if (firstComma == -1) return;

    int id_moteur = message.substring(0, firstComma).toInt();
    if (id_moteur < 1 || id_moteur > NB_MOTEURS) return;

    float capteurs[NB_FEATURES];
    int   pos      = firstComma + 1;
    bool  parse_ok = true;

    for (int i = 0; i < NB_FEATURES; i++) {
        int    nextComma = message.indexOf(',', pos);
        String token     = (nextComma == -1)
            ? message.substring(pos)
            : message.substring(pos, nextComma);
        if (nextComma != -1) pos = nextComma + 1;
        if (token.length() == 0) { parse_ok = false; break; }
        capteurs[i] = token.toFloat();
    }

    if (!parse_ok) { Serial.println("ERREUR parsing"); return; }
    traiterNouvelleLigne(id_moteur, capteurs);
}

// ============================================================
void traiterNouvelleLigne(int id_moteur, float* capteurs_bruts) {
    int         idx = id_moteur - 1;
    EtatMoteur& m   = moteurs[idx];

    for (int i = 0; i < NB_FEATURES; i++) {
        float val = (scaler_range[i] > 0.0f)
            ? (capteurs_bruts[i] - scaler_min[i]) / scaler_range[i]
            : 0.0f;
        m.buffer[m.index_ecriture][i] = constrain(val, 0.0f, 1.0f);
    }

    m.index_ecriture = (m.index_ecriture + 1) % TIME_STEPS;
    if (m.index_ecriture == 0) m.buffer_rempli = true;

    if (!m.buffer_rempli) {
        Serial.print("Moteur "); Serial.print(id_moteur);
        Serial.print(" | Buffer : ");
        Serial.print(m.index_ecriture);
        Serial.print("/"); Serial.println(TIME_STEPS);
        return;
    }

    // Préparer et lancer l'inférence
    float entree_locale[NUMBER_OF_INPUTS];
    preparerTenseur(idx, entree_locale);

    for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
        input->data.f[i] = entree_locale[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.print("ERREUR inference moteur ");
        Serial.println(id_moteur);
        return;
    }

    float rul = constrain(output->data.f[0], 0.0f, 999.0f);
    afficherResultat(id_moteur, rul);
}

// ============================================================
void preparerTenseur(int idx, float* sortie) {
    EtatMoteur& m   = moteurs[idx];
    int         plat = 0;
    for (int i = 0; i < TIME_STEPS; i++) {
        int chrono = (m.index_ecriture + i) % TIME_STEPS;
        for (int j = 0; j < NB_FEATURES; j++) {
            sortie[plat++] = m.buffer[chrono][j];
        }
    }
}

// ============================================================
void afficherResultat(int id_moteur, float rul) {
    Serial.print("Moteur "); Serial.print(id_moteur);
    Serial.print(" | RUL: "); Serial.print(rul, 1);
    if      (rul <= RUL_SEUIL_CRITIQUE) Serial.println(" -> CRITIQUE");
    else if (rul <= RUL_SEUIL_DEGRADE)  Serial.println(" -> DEGRADE");
    else                                Serial.println(" -> SAIN");
}