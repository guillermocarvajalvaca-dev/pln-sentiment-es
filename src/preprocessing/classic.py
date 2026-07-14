import re                                    # expresiones regulares para manipulación de texto
import pandas as pd                          # manipulación de tablas de datos

# Negaciones en español que se conservan explícitamente
# Motivo: "no funciona" debe tratarse distinto a "funciona" — la negación cambia el sentimiento
NEGATIONS = ["no", "nunca", "ni", "sin", "tampoco", "jamás", "nada", "nadie"]


def preprocess_classic(text: str) -> str:
    """
    Preprocesamiento para la Rama A (TF-IDF + Naive Bayes).
    Conserva negaciones porque invierten el sentimiento de la frase.
    No elimina stopwords masivamente: TF-IDF ya pondera su importancia.
    """
    text = str(text).lower()                 # convierte a minúsculas para normalizar el vocabulario

    # Marca las negaciones con prefijo NEG_ para que el vectorizador las trate como términos especiales
    # Ejemplo: "no funciona" → "NEG_no funciona" preservando el contexto negativo
    for neg in NEGATIONS:
        text = re.sub(
            rf"\b{neg}\b",                   # busca la negación como palabra completa
            f"NEG_{neg}",                    # la reemplaza con el prefijo NEG_
            text
        )

    text = re.sub(r"[^a-záéíóúüñ\s_]", " ", text)  # elimina caracteres especiales, conserva tildes
    text = re.sub(r"\s+", " ", text)         # colapsa espacios múltiples en uno solo
    return text.strip()                      # elimina espacios al inicio y al final


def apply_classic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el preprocesamiento clásico a la columna text_classic del DataFrame."""
    df = df.copy()                           # no modifica el DataFrame original
    df["text_classic"] = df["text_raw"].apply(preprocess_classic)  # aplica función fila por fila
    return df