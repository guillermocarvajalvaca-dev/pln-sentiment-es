import pandas as pd                          # manipulación de tablas de datos
import numpy as np                           # cálculos numéricos para percentiles
from config.settings import RANDOM_STATE     # semilla de reproducibilidad


def audit_dataframe(df: pd.DataFrame, split_name: str = "completo") -> dict:
    """Ejecuta la auditoría completa del DataFrame según el checklist §6.6 del contrato."""
    print(f"\n{'='*60}")
    print(f"AUDITORIA - split: {split_name}")
    print(f"{'='*60}")

    # Dimensiones del dataset
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

    # Tipos de columnas
    print(f"\nTipos de columnas:\n{df.dtypes}")

    # Valores nulos por columna
    nulls = df.isnull().sum()
    print(f"\nValores nulos:\n{nulls[nulls > 0] if nulls.sum() > 0 else 'Ninguno'}")

    # Textos vacíos en text_raw
    empty_texts = (df["text_raw"].str.strip() == "").sum()
    print(f"\nTextos vacíos en text_raw: {empty_texts}")

    # Duplicados exactos por texto
    duplicates = df.duplicated(subset=["text_raw"]).sum()
    print(f"Duplicados exactos en text_raw: {duplicates}")

    # Distribución de clases
    print(f"\nDistribución de clases:")
    print(df["sentiment_label"].value_counts().sort_index())

    # Longitud de textos en palabras
    word_lengths = df["text_raw"].str.split().str.len()
    p75 = int(np.percentile(word_lengths, 75))   # percentil 75: criterio OC-02 para max_length
    p95 = int(np.percentile(word_lengths, 95))   # percentil 95: referencia original del contrato
    print(f"\nLongitud en palabras:")
    print(f"  Media:        {word_lengths.mean():.1f}")
    print(f"  Mediana:      {word_lengths.median():.1f}")
    print(f"  Percentil 75: {p75}  <- max_length para GRU (criterio OC-02)")
    print(f"  Percentil 95: {p95}  <- referencia original del contrato")
    print(f"  Máximo:       {word_lengths.max()}")

    # Longitud de textos en caracteres
    char_lengths = df["text_raw"].str.len()
    print(f"\nLongitud en caracteres:")
    print(f"  Media:   {char_lengths.mean():.1f}")
    print(f"  Máximo:  {char_lengths.max()}")

    # Rango real de label_original
    print(f"\nRango real de label_original: {df['label_original'].min()} - {df['label_original'].max()}")
    print(f"Distribución:\n{df['label_original'].value_counts().sort_index()}")

    results = {
        "shape": df.shape,
        "nulls": nulls.sum(),
        "empty_texts": empty_texts,
        "duplicates": duplicates,
        "p75_words": p75,                    # se usa para fijar GRU_CONFIG["max_length"]
        "p95_words": p95,
        "label_range": f"{df['label_original'].min()}-{df['label_original'].max()}",
    }

    print(f"\n{'='*60}")
    print("AUDITORÍA COMPLETADA")
    print(f"{'='*60}\n")
    return results


def check_no_leakage(df: pd.DataFrame) -> bool:
    """Verifica que no haya textos duplicados entre train y test (fuga de datos)."""
    train_texts = set(df[df["split"] == "train"]["text_raw"])   # textos del split train
    test_texts  = set(df[df["split"] == "test"]["text_raw"])    # textos del split test
    overlap = train_texts & test_texts                           # intersección entre ambos sets
    if overlap:
        print(f"ALERTA: {len(overlap)} textos duplicados entre train y test - STOP THE LINE")
        return False
    print("OK - Sin fuga de datos: train y test no comparten textos")
    return True