import pandas as pd                          # manipulación de tablas de datos
import json                                  # para guardar y cargar el tokenizador en formato JSON
from tensorflow.keras.preprocessing.text import Tokenizer          # convierte texto a secuencias de números
from tensorflow.keras.preprocessing.sequence import pad_sequences  # iguala la longitud de todas las secuencias
from config.settings import ARTIFACTS, GRU_CONFIG


def build_tokenizer(train_texts: list) -> Tokenizer:
    """
    Crea y ajusta el tokenizador SOLO con los textos de entrenamiento.
    Regla anti-fuga: el tokenizador nunca ve validation ni test durante el ajuste.
    """
    tokenizer = Tokenizer(
        num_words=GRU_CONFIG["max_vocab_size"],  # limita el vocabulario a las N palabras más frecuentes
        oov_token="<OOV>"                        # token especial para palabras fuera del vocabulario
    )
    tokenizer.fit_on_texts(train_texts)          # aprende el vocabulario únicamente desde train
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer) -> None:
    """Guarda el tokenizador en disco como JSON para reutilizarlo en inferencia."""
    tokenizer_path = ARTIFACTS / "tokenizer" / "tokenizer.json"
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)  # crea la carpeta si no existe
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())             # serializa el tokenizador completo a JSON
    print(f"Tokenizador guardado en {tokenizer_path}")


def load_tokenizer() -> Tokenizer:
    """Carga el tokenizador desde disco para usarlo en inferencia sin reentrenar."""
    tokenizer_path = ARTIFACTS / "tokenizer" / "tokenizer.json"
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        return tokenizer_from_json(f.read())     # reconstruye el tokenizador desde JSON


def texts_to_padded_sequences(texts: list, tokenizer: Tokenizer) -> object:
    """
    Convierte textos a secuencias numéricas de longitud uniforme.
    max_length=35: criterio OC-02, percentil 75 de longitudes en train.
    Secuencias más cortas se rellenan con ceros; las más largas se truncan.
    """
    sequences = tokenizer.texts_to_sequences(texts)   # convierte cada texto a lista de IDs numéricos
    padded = pad_sequences(
        sequences,
        maxlen=GRU_CONFIG["max_length"],         # longitud fija: 35 palabras
        padding="post",                          # rellena con ceros al final de la secuencia
        truncating="post"                        # trunca por el final si supera max_length
    )
    return padded                                # matriz numpy de shape (n_textos, 35)


def apply_neural_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza mínima para la Rama B — el tokenizador maneja la representación."""
    df = df.copy()                               # no modifica el DataFrame original
    df["text_neural"] = df["text_raw"].str.lower().str.strip()  # solo minúsculas y espacios
    return df


from tensorflow.keras.preprocessing.text import tokenizer_from_json  # importa al final para evitar conflicto circular