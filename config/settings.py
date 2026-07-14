from pathlib import Path

# --- REPRODUCIBILIDAD ---
# Semilla fija: garantiza que el muestreo y el entrenamiento sean idénticos en cada ejecución
RANDOM_STATE = 42

# --- RUTAS DEL PROYECTO ---
# BASE_DIR: raíz del proyecto, se calcula automáticamente desde la ubicación de este archivo
BASE_DIR = Path(__file__).resolve().parent.parent
# Carpeta donde se guarda el dataset original sin modificar
DATA_RAW = BASE_DIR / "data" / "raw"
# Carpeta donde se guarda el CSV procesado y balanceado
DATA_PROCESSED = BASE_DIR / "data" / "processed"
# Carpeta raíz de todos los artefactos generados por el entrenamiento
ARTIFACTS = BASE_DIR / "artifacts"

# --- FUENTE DE DATOS ---
# Espejo oficial del Multilingual Amazon Reviews Corpus en español (MARC original está defunct)
DATASET_NAME = "SetFit/amazon_reviews_multi_es"

# --- ORDEN CONTRACTUAL DE CLASES (INMUTABLE) ---
# 0=NEGATIVO, 1=NEUTRO, 2=POSITIVO — este orden rige en todo el sistema sin excepción
LABEL_ID_TO_NAME = {0: "NEGATIVO", 1: "NEUTRO", 2: "POSITIVO"}
# Mapa inverso: convierte nombre de clase a su ID numérico
LABEL_NAME_TO_ID = {v: k for k, v in LABEL_ID_TO_NAME.items()}

# --- MAPEO DE ESTRELLAS A SENTIMIENTO ---
# Si label va de 0 a 4: 1★ y 2★ → NEGATIVO, 3★ → NEUTRO, 4★ y 5★ → POSITIVO
STAR_TO_SENTIMENT_0_4 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
# Si label va de 1 a 5: misma lógica pero con rango 1-5
STAR_TO_SENTIMENT_1_5 = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

# --- TAMAÑO DE LA MUESTRA POR SPLIT ---
# 3000 por clase en train, 500 por clase en validation y test → dataset balanceado
SAMPLE_SIZES = {
    "train":      {"NEGATIVO": 3000, "NEUTRO": 3000, "POSITIVO": 3000},
    "validation": {"NEGATIVO": 500,  "NEUTRO": 500,  "POSITIVO": 500},
    "test":       {"NEGATIVO": 500,  "NEUTRO": 500,  "POSITIVO": 500},
}

# --- RAMA A: PARÁMETROS DEL VECTORIZADOR TF-IDF ---
# ngram_range=(1,2): usa palabras sueltas y pares de palabras (bigramas)
# min_df=2: ignora términos que aparecen en menos de 2 documentos (ruido)
# max_df=0.95: ignora términos que aparecen en más del 95% de documentos (demasiado comunes)
# sublinear_tf=True: aplica logaritmo a la frecuencia para reducir el peso de términos muy frecuentes
TFIDF_PARAMS = {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "sublinear_tf": True}

# --- RAMA B: PARÁMETROS DE LA RED NEURONAL GRU ---
# max_vocab_size: tamaño máximo del vocabulario aprendido por el tokenizador
# embedding_dim: dimensión del vector que representa cada palabra
# gru_units: número de unidades en la capa GRU (memoria de la red)
# dropout: fracción de neuronas desactivadas aleatoriamente para evitar sobreajuste
# max_length: longitud máxima de secuencia - se fija con el percentil 75 de longitudes en train
# epochs: número máximo de épocas de entrenamiento
# batch_size: número de ejemplos procesados por paso de entrenamiento
# patience: épocas sin mejora antes de detener el entrenamiento (EarlyStopping)
GRU_CONFIG = {
    "max_vocab_size": 10000,  # reducido de 20000: criterio OC-02, entrenamiento CPU en plazo
    "embedding_dim": 100,
    "gru_units": 64,
    "dropout": 0.30,
    "max_length": 35,         # fijado en percentil 75 tras auditoría - criterio OC-02
    "epochs": 20,
    "batch_size": 64,
    "patience": 3,
}

# --- RAMA C: PARÁMETROS DEL MODELADO DE TEMAS LDA ---
# n_topics: número de temas a descubrir en el corpus
# max_features: tamaño máximo del vocabulario para CountVectorizer
# ngram_range=(1,2): incluye bigramas para capturar frases temáticas como "atención_cliente"
# min_df=5: ignora términos muy raros
# max_df=0.90: ignora términos demasiado frecuentes
LDA_CONFIG = {
    "n_topics": 5,
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.90,
}

# --- INTERFAZ ---
# Umbral de confianza: si la probabilidad máxima es menor a 0.60, se muestra advertencia
CONFIDENCE_THRESHOLD = 0.60