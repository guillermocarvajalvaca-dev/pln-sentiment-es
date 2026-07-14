import json                                  # guarda métricas en formato JSON
import time                                  # mide el tiempo de inferencia
import numpy as np                           # operaciones numéricas
import tensorflow as tf                      # framework de deep learning
from tensorflow.keras.models import Sequential        # modelo secuencial capa por capa
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout  # capas de la red neuronal
from tensorflow.keras.callbacks import EarlyStopping  # detiene el entrenamiento si no mejora
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config.settings import ARTIFACTS, GRU_CONFIG, LABEL_ID_TO_NAME, RANDOM_STATE

# Fija semillas para reproducibilidad total del entrenamiento neuronal
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_gru_model(vocab_size: int) -> tf.keras.Model:
    """
    Construye la arquitectura GRU según el contrato §8.2.
    Embedding: convierte cada ID de palabra en un vector denso de 100 dimensiones.
    GRU: procesa la secuencia palabra por palabra manteniendo memoria del contexto anterior.
    Dropout: desactiva aleatoriamente 30% de neuronas para evitar sobreajuste.
    Dense+Softmax: convierte la salida de la GRU en probabilidades por clase.
    """
    model = Sequential([
        # Capa de embeddings: aprende representaciones vectoriales de las palabras durante el entrenamiento
        Embedding(
            input_dim=vocab_size,            # tamaño del vocabulario conocido por el tokenizador
            output_dim=GRU_CONFIG["embedding_dim"],   # cada palabra se representa con 100 números
            input_length=GRU_CONFIG["max_length"],    # longitud fija de secuencia: 35 palabras
        ),
        # Capa GRU: lee la secuencia palabra por palabra y aprende dependencias de orden
        GRU(
            units=GRU_CONFIG["gru_units"],   # 64 unidades de memoria en la capa recurrente
            dropout=GRU_CONFIG["dropout"],   # 30% de dropout durante entrenamiento
        ),
        # Capa de salida: 3 neuronas (una por clase) con activación softmax
        # Softmax: convierte los valores en probabilidades que suman 1.0
        Dense(3, activation="softmax"),      # 3 clases: NEGATIVO, NEUTRO, POSITIVO
    ])

    # Compila el modelo con optimizador Adam y función de pérdida para clasificación multiclase
    model.compile(
        optimizer="adam",                    # Adam: ajusta la tasa de aprendizaje automáticamente
        loss="sparse_categorical_crossentropy",  # función de pérdida para etiquetas enteras (0,1,2)
        metrics=["accuracy"],                # métrica a monitorear durante el entrenamiento
    )
    model.summary()                          # imprime la arquitectura completa del modelo
    return model


def train_gru(X_train, y_train, X_val, y_val) -> tuple:
    """
    Entrena la GRU con EarlyStopping sobre validation.
    EarlyStopping: detiene el entrenamiento si val_loss no mejora en 3 épocas consecutivas.
    restore_best_weights: recupera los pesos de la mejor época al terminar.
    """
    vocab_size = GRU_CONFIG["max_vocab_size"] + 1  # +1 por el token OOV reservado
    model = build_gru_model(vocab_size)

    early_stopping = EarlyStopping(
        monitor="val_loss",                  # observa la pérdida en validation
        patience=GRU_CONFIG["patience"],     # espera 3 épocas sin mejora antes de detener
        restore_best_weights=True,           # recupera los pesos de la época con menor val_loss
    )

    history = model.fit(
        X_train, y_train,                    # datos de entrenamiento
        validation_data=(X_val, y_val),      # datos de validación para EarlyStopping
        epochs=GRU_CONFIG["epochs"],         # máximo 20 épocas
        batch_size=GRU_CONFIG["batch_size"], # 64 ejemplos por paso de gradiente
        callbacks=[early_stopping],          # activa EarlyStopping durante el entrenamiento
        verbose=1,                           # muestra progreso por época
    )

    print(f"Entrenamiento detenido en época {len(history.history['loss'])}")
    return model, history


def evaluate_gru(model, X_test, test_labels: list) -> dict:
    """
    Evalúa la GRU sobre el mismo test que Naive Bayes.
    Garantiza comparación justa: mismo conjunto, mismas métricas.
    """
    start = time.time()
    probs = model.predict(X_test, verbose=0)         # probabilidades por clase shape (n, 3)
    inference_time = (time.time() - start) / len(test_labels)  # tiempo medio por reseña

    predictions = np.argmax(probs, axis=1)           # clase con mayor probabilidad

    metrics = {
        "model": "GRU",
        "accuracy":          round(float(accuracy_score(test_labels, predictions)), 4),
        "precision_macro":   round(float(precision_score(test_labels, predictions, average="macro")), 4),
        "recall_macro":      round(float(recall_score(test_labels, predictions, average="macro")), 4),
        "f1_macro":          round(float(f1_score(test_labels, predictions, average="macro")), 4),
        "f1_weighted":       round(float(f1_score(test_labels, predictions, average="weighted")), 4),
        "f1_per_class":      {
            LABEL_ID_TO_NAME[i]: round(float(v), 4)
            for i, v in enumerate(f1_score(test_labels, predictions, average=None))
        },
        "inference_time_s":  round(inference_time, 6),
        "confusion_matrix":  confusion_matrix(test_labels, predictions).tolist(),
    }

    print(f"GRU — Accuracy: {metrics['accuracy']} | F1 macro: {metrics['f1_macro']}")
    return metrics


def save_gru(model, metrics: dict) -> None:
    """Persiste el modelo GRU y sus métricas en artifacts/ para la interfaz."""
    model_path   = ARTIFACTS / "models"  / "gru_sentiment.keras"
    metrics_path = ARTIFACTS / "metrics" / "gru_metrics.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)                   # guarda arquitectura + pesos en formato .keras
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Modelo GRU guardado en {model_path}")


def load_gru():
    """Carga el modelo GRU desde disco para inferencia — sin reentrenar.
    Usa formato SavedModel para compatibilidad entre versiones de TensorFlow."""
    import os
    savedmodel_path = ARTIFACTS / "models" / "gru_sentiment_savedmodel"
    keras_path      = ARTIFACTS / "models" / "gru_sentiment.keras"

    if savedmodel_path.exists():                 # prioriza SavedModel — más compatible
        model = tf.saved_model.load(str(savedmodel_path))
        print("GRU cargada desde SavedModel")
        return model
    elif keras_path.exists():                    # fallback al formato .keras
        model = tf.keras.models.load_model(str(keras_path))
        print("GRU cargada desde .keras")
        return model
    else:
        raise FileNotFoundError("No se encontró el modelo GRU en artifacts/models/")