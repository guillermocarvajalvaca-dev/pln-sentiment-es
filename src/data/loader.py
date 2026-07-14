import pandas as pd                          # manipulación de tablas de datos
import re                                    # expresiones regulares para limpieza de texto
from datasets import load_dataset            # descarga el dataset desde Hugging Face Hub
from config.settings import (
    DATASET_NAME,                            # nombre del dataset en Hugging Face
    DATA_RAW,                                # ruta donde se guarda el dataset crudo
    DATA_PROCESSED,                          # ruta donde se guarda el CSV procesado
    SAMPLE_SIZES,                            # cantidad de ejemplos por clase y split
    LABEL_ID_TO_NAME,                        # mapa de ID numérico a nombre de clase
    RANDOM_STATE,                            # semilla para reproducibilidad
)


def clean_common(text: str) -> str:
    """Limpieza común aplicada a las tres ramas: elimina ruido sin alterar el contenido."""
    text = str(text)                         # convierte a string por si viene como otro tipo
    text = re.sub(r"<[^>]+>", " ", text)    # elimina etiquetas HTML como <br>, <p>, etc.
    text = re.sub(r"\s+", " ", text)        # colapsa múltiples espacios y saltos de línea en uno
    return text.strip()                      # elimina espacios al inicio y al final


def map_label_to_sentiment(label: int, label_range: str) -> int:
    """Convierte la calificación en estrellas al orden contractual: 0=NEG, 1=NEU, 2=POS."""
    if label_range == "0_4":                 # rango 0-4: 0★=NEG, 1★=NEG, 2★=NEU, 3★=POS, 4★=POS
        mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    else:                                    # rango 1-5: 1★=NEG, 2★=NEG, 3★=NEU, 4★=POS, 5★=POS
        mapping = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    return mapping[label]                    # devuelve el ID de sentimiento correspondiente


def download_and_save_raw(label_range: str = "0_4") -> pd.DataFrame:
    """Descarga el dataset, lo guarda en data/raw/ y devuelve un DataFrame combinado."""
    print(f"Descargando {DATASET_NAME} desde Hugging Face Hub...")
    ds = load_dataset(DATASET_NAME)          # descarga los tres splits: train, validation, test

    frames = []                              # lista para acumular los DataFrames de cada split
    for split_name in ["train", "validation", "test"]:
        df = ds[split_name].to_pandas()      # convierte el split a DataFrame de pandas
        df["split"] = split_name            # agrega columna para identificar a qué split pertenece
        frames.append(df)                    # acumula el DataFrame en la lista

    raw_df = pd.concat(frames, ignore_index=True)   # une los tres splits en un solo DataFrame
    DATA_RAW.mkdir(parents=True, exist_ok=True)      # crea la carpeta data/raw/ si no existe
    raw_df.to_csv(DATA_RAW / "amazon_es_raw.csv", index=False)  # guarda copia sin modificar
    print(f"Dataset crudo guardado en {DATA_RAW / 'amazon_es_raw.csv'}")
    print(f"Dimensiones: {raw_df.shape} | Columnas: {list(raw_df.columns)}")
    print(raw_df["label"].value_counts().sort_index())  # muestra distribución real de etiquetas
    return raw_df


def build_balanced_sample(raw_df: pd.DataFrame, label_range: str = "0_4") -> pd.DataFrame:
    """Construye la muestra balanceada de 9000/1500/1500 registros desde los splits nativos."""
    records = []                             # lista para acumular los registros seleccionados

    for split_name, class_sizes in SAMPLE_SIZES.items():   # itera sobre train, validation, test
        split_df = raw_df[raw_df["split"] == split_name].copy()  # filtra solo ese split

        split_df["sentiment_id"] = split_df["label"].apply(
            lambda x: map_label_to_sentiment(x, label_range)  # convierte estrella a sentimiento
        )

        sampled_parts = []                   # partes muestreadas por clase para este split
        for sent_name, n in class_sizes.items():             # itera sobre NEG, NEU, POS
            sent_id = {"NEGATIVO": 0, "NEUTRO": 1, "POSITIVO": 2}[sent_name]
            class_df = split_df[split_df["sentiment_id"] == sent_id]  # filtra por clase
            sampled = class_df.sample(n=n, random_state=RANDOM_STATE)  # muestrea n ejemplos
            sampled_parts.append(sampled)    # acumula la muestra de esta clase

        split_sample = pd.concat(sampled_parts, ignore_index=True)  # une las 3 clases del split
        records.append(split_sample)         # acumula este split

    df = pd.concat(records, ignore_index=True)   # une los tres splits en el DataFrame final

    df["record_id"] = ["r{:06d}".format(i) for i in range(len(df))]  # ID único por registro
    df["text_raw"] = df["text"].apply(clean_common)       # aplica limpieza común al texto
    df["text_classic"] = df["text_raw"]                   # base para Rama A (se procesa más en classic.py)
    df["text_lda"] = df["text_raw"]                       # base para Rama C (se procesa más en lda.py)
    df["label_original"] = df["label"]                    # conserva la etiqueta original en estrellas
    df["sentiment_label"] = df["sentiment_id"].map(LABEL_ID_TO_NAME)  # nombre legible de la clase

    result = df[[                            # selecciona y ordena las columnas del esquema contractual
        "record_id", "split", "text_raw", "text_classic",
        "text_lda", "label_original", "sentiment_id", "sentiment_label"
    ]]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)     # crea data/processed/ si no existe
    result.to_csv(DATA_PROCESSED / "reviews_es_balanced.csv", index=False)  # guarda el CSV
    print(f"CSV procesado guardado: {result.shape[0]} registros, {result.shape[1]} columnas")
    print(result["sentiment_label"].value_counts())        # verifica distribución final
    return result