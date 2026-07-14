"""
main.py — Orquestador del pipeline completo según el manual §4 Paso 4.
Orden obligatorio: Rama A (NB) → Rama B (GRU) → Rama C (LDA) → Evaluación → champion.json
Ejecutar una sola vez para entrenar todos los modelos y generar todos los artefactos.
"""

import sys                                   # permite agregar la raíz al path de importación
sys.path.insert(0, ".")                      # agrega la raíz del proyecto al path de Python

import pandas as pd                          # manipulación del CSV procesado
import numpy as np                           # operaciones numéricas
from config.settings import DATA_PROCESSED, RANDOM_STATE

# Módulos de preprocesamiento por rama
from src.preprocessing.classic import apply_classic_preprocessing   # Rama A
from src.preprocessing.neural  import (                             # Rama B
    build_tokenizer, save_tokenizer, texts_to_padded_sequences
)
from src.preprocessing.lda     import apply_lda_preprocessing       # Rama C

# Módulos de modelos
from src.models.naive_bayes_model import train_naive_bayes, evaluate_naive_bayes, save_naive_bayes
from src.models.gru_model         import train_gru, evaluate_gru, save_gru
from src.models.lda_model         import train_lda, get_topic_words, save_lda, save_topic_names

# Módulos de evaluación
from src.evaluation.metrics import plot_confusion_matrix, plot_training_curves, save_comparison


def main():
    print("=" * 60)
    print("PIPELINE PLN — OBSERVATORIO DE OPINIONES E-COMMERCE")
    print("=" * 60)

    # ── PASO 1: Carga del CSV procesado ──────────────────────────
    print("\n[1/8] Cargando CSV procesado...")
    df = pd.read_csv(DATA_PROCESSED / "reviews_es_balanced.csv")  # lee el CSV balanceado
    print(f"Registros cargados: {df.shape[0]}")

    # Separa los splits según su columna 'split'
    train_df = df[df["split"] == "train"].copy()       # 9.000 registros para entrenar
    val_df   = df[df["split"] == "validation"].copy()  # 1.500 registros para validar
    test_df  = df[df["split"] == "test"].copy()        # ~998 registros para evaluar

    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

    # ── PASO 2: Preprocesamiento Rama A ──────────────────────────
    print("\n[2/8] Preprocesamiento Rama A (clásica)...")
    train_df = apply_classic_preprocessing(train_df)   # aplica minúsculas + negaciones + limpieza
    val_df   = apply_classic_preprocessing(val_df)
    test_df  = apply_classic_preprocessing(test_df)

    # ── PASO 3: Rama A — TF-IDF + Naive Bayes ────────────────────
    print("\n[3/8] Entrenando Naive Bayes (Rama A)...")
    vectorizer_nb, model_nb = train_naive_bayes(       # ajusta TF-IDF y entrena NB solo con train
        train_df["text_classic"].tolist(),
        train_df["sentiment_id"].tolist(),
    )
    save_naive_bayes(vectorizer_nb, model_nb, {})      # guarda artefactos en disco
    print("Naive Bayes entrenado y guardado.")

    # ── PASO 4: Preprocesamiento Rama B + GRU ────────────────────
    print("\n[4/8] Preprocesamiento y entrenamiento Rama B (GRU)...")
    # Tokenizador: se ajusta SOLO con train para evitar fuga de datos
    tokenizer = build_tokenizer(train_df["text_raw"].str.lower().str.strip().tolist())
    save_tokenizer(tokenizer)                          # persiste el tokenizador en disco

    # Convierte textos a secuencias padded de longitud fija (35 palabras)
    X_train_gru = texts_to_padded_sequences(train_df["text_raw"].str.lower().str.strip().tolist(), tokenizer)
    X_val_gru   = texts_to_padded_sequences(val_df["text_raw"].str.lower().str.strip().tolist(),   tokenizer)
    X_test_gru  = texts_to_padded_sequences(test_df["text_raw"].str.lower().str.strip().tolist(),  tokenizer)

    y_train = np.array(train_df["sentiment_id"].tolist())   # etiquetas de entrenamiento (array para Keras)
    y_val   = np.array(val_df["sentiment_id"].tolist())     # etiquetas de validación
    y_test  = np.array(test_df["sentiment_id"].tolist())    # etiquetas de test

    # Entrena la GRU con EarlyStopping sobre validation
    model_gru, history = train_gru(X_train_gru, y_train, X_val_gru, y_val)
    plot_training_curves(history)                      # genera curvas de entrenamiento PNG

    # ── PASO 5: Rama C — LDA ─────────────────────────────────────
    print("\n[5/8] Entrenando LDA (Rama C)...")
    train_df = apply_lda_preprocessing(train_df)       # elimina stopwords para LDA
    vectorizer_lda, lda = train_lda(                   # ajusta CountVectorizer y LDA solo con train
        train_df["text_lda"].tolist()
    )
    topics = get_topic_words(lda, vectorizer_lda)      # extrae top-10 palabras por tema
    save_lda(vectorizer_lda, lda, topics)              # guarda artefactos LDA en disco

    # Nombres provisionales — DEBEN revisarse manualmente tras inspeccionar las palabras
    # El contrato §13 G3 exige que al menos 4 de 5 temas tengan nombre interpretable
    topic_names_provisional = {
        "topic_0": "Revisar tras inspección",
        "topic_1": "Revisar tras inspección",
        "topic_2": "Revisar tras inspección",
        "topic_3": "Revisar tras inspección",
        "topic_4": "Revisar tras inspección",
    }
    save_topic_names(topic_names_provisional)          # guarda nombres provisionales

    # ── PASO 6: Evaluación sobre el mismo test ───────────────────
    print("\n[6/8] Evaluando ambos modelos sobre el test...")
    # Naive Bayes: transforma test con el vectorizador ya ajustado
    nb_metrics = evaluate_naive_bayes(
        vectorizer_nb, model_nb,
        test_df["text_classic"].tolist(),
        y_test,
    )
    save_naive_bayes(vectorizer_nb, model_nb, nb_metrics)  # sobreescribe con métricas reales

    # GRU: evalúa sobre las secuencias padded del test
    gru_metrics = evaluate_gru(model_gru, X_test_gru, y_test)
    save_gru(model_gru, gru_metrics)                   # guarda modelo GRU y métricas

    # ── PASO 7: Matrices de confusión y champion ─────────────────
    print("\n[7/8] Generando matrices y determinando champion...")
    plot_confusion_matrix(nb_metrics["confusion_matrix"],  "Naive Bayes")  # PNG matriz NB
    plot_confusion_matrix(gru_metrics["confusion_matrix"], "GRU")          # PNG matriz GRU
    save_comparison(nb_metrics, gru_metrics)           # escribe comparison.json y champion.json

    # ── PASO 8: Smoke test final ──────────────────────────────────
    print("\n[8/8] Smoke test final — verificando los 4 ejemplos de demo...")
    from src.inference.predictor import Predictor      # importa aquí para usar artefactos recién creados
    predictor = Predictor()                            # carga todos los artefactos desde disco

    ejemplos = [
        "El producto llegó antes de lo previsto y funciona perfectamente.",
        "No funciona, llegó roto y el vendedor nunca respondió.",
        "Cumple con lo básico, pero esperaba una mejor calidad por el precio.",
        "El producto funciona muy bien, pero llegó dos semanas tarde y la caja estaba dañada.",
        "Mal",                                         # ejemplo de borde para verificar Señal de Duda
    ]

    for texto in ejemplos:
        r = predictor.predict(texto)
        if "error" in r:
            print(f"  [{texto[:40]}] -> ERROR: {r['error']}")
        else:
            duda = "DUDA" if r["flag_duda"] else "OK"
            print(f"  [{texto[:40]}...] -> {r['sentimiento']} ({r['confianza']*100:.1f}%) | NB:{r['prediccion_nb']} GRU:{r['prediccion_gru']} | {duda} | Tema:{r['tema_nombre']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO — Todos los artefactos generados")
    print("=" * 60)
    print("Siguiente paso: revisar lda_topics.json y asignar nombres reales a los temas")
    print("Luego ejecutar: python src/app/gradio_app.py")


if __name__ == "__main__":
    main()