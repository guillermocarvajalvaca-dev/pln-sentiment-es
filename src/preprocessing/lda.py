import re                                    # expresiones regulares para limpieza de texto
import pandas as pd                          # manipulación de tablas de datos
import nltk                                  # librería de PLN para stopwords en español
from config.settings import LDA_CONFIG

# Descarga las stopwords de NLTK si no están disponibles localmente
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Carga las stopwords en español — palabras vacías que no aportan significado temático
# Ejemplo: "de", "la", "que", "en" — se eliminan para que LDA se enfoque en términos relevantes
STOPWORDS_ES = set(stopwords.words("spanish"))


def preprocess_lda(text: str) -> str:
    """
    Preprocesamiento para la Rama C (LDA).
    Elimina stopwords para que el modelo temático se enfoque en términos con significado.
    Usa bigramas para capturar frases como 'atención cliente' o 'buena calidad'.
    """
    text = str(text).lower()                 # convierte a minúsculas para normalizar el vocabulario
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)   # elimina caracteres especiales, conserva tildes
    text = re.sub(r"\s+", " ", text)         # colapsa espacios múltiples en uno solo
    tokens = text.strip().split()            # divide el texto en palabras individuales

    # Elimina stopwords: filtra las palabras que no aportan información temática
    tokens = [t for t in tokens if t not in STOPWORDS_ES and len(t) > 2]

    return " ".join(tokens)                  # reconstruye el texto limpio como string


def apply_lda_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el preprocesamiento LDA a la columna text_lda del DataFrame."""
    df = df.copy()                           # no modifica el DataFrame original
    df["text_lda"] = df["text_raw"].apply(preprocess_lda)   # aplica función fila por fila

    # Manejo del caso M1: texto vacío tras eliminar stopwords (reseñas muy cortas)
    # En lugar de fallar, devuelve un string vacío que el predictor maneja con fallback
    empty_count = (df["text_lda"].str.strip() == "").sum()
    if empty_count > 0:
        print(f"⚠️  {empty_count} textos quedaron vacíos tras preprocesamiento LDA — se manejarán con fallback M1")

    return df