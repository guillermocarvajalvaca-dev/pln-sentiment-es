import joblib                                # guarda y carga modelos entrenados en disco
import json                                  # guarda métricas en formato JSON legible
import time                                  # mide el tiempo de inferencia
import numpy as np                           # operaciones numéricas
from sklearn.feature_extraction.text import TfidfVectorizer    # convierte texto a matriz TF-IDF
from sklearn.naive_bayes import MultinomialNB                  # clasificador probabilístico bayesiano
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config.settings import ARTIFACTS, TFIDF_PARAMS, LABEL_ID_TO_NAME


def train_naive_bayes(train_texts: list, train_labels: list) -> tuple:
    """
    Entrena el pipeline TF-IDF + Naive Bayes SOLO con datos de entrenamiento.
    Regla anti-fuga: el vectorizador aprende el vocabulario únicamente desde train.
    TF-IDF: convierte cada texto en un vector de pesos que refleja la importancia de cada palabra.
    Naive Bayes: calcula la probabilidad de cada clase dado el vector de palabras.
    """
    # Crea el vectorizador con los parámetros del contrato §8.1
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

    # fit_transform: ajusta el vocabulario con train Y transforma train a matriz TF-IDF
    # Nunca se llama fit sobre validation o test — solo transform
    X_train = vectorizer.fit_transform(train_texts)  # matriz dispersa (n_textos, n_términos)

    # Entrena Naive Bayes: aprende P(clase|palabras) desde la matriz TF-IDF
    model = MultinomialNB()
    model.fit(X_train, train_labels)         # ajusta las probabilidades de cada clase

    print(f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} términos")
    print(f"Clases aprendidas: {model.classes_}")
    return vectorizer, model                 # devuelve ambos objetos para persistirlos


def evaluate_naive_bayes(vectorizer, model, test_texts: list, test_labels: list) -> dict:
    """
    Evalúa el modelo sobre el conjunto de test — el mismo test que usará GRU.
    Mide tiempo de inferencia para compararlo con GRU en la tabla del contrato §9.
    """
    # Transforma test con el vectorizador ya ajustado — solo transform, nunca fit
    X_test = vectorizer.transform(test_texts)

    # Mide tiempo de inferencia sobre todo el test
    start = time.time()
    predictions = model.predict(X_test)      # predice la clase para cada texto
    probs = model.predict_proba(X_test)      # probabilidades por clase en orden contractual
    inference_time = (time.time() - start) / len(test_texts)  # tiempo medio por reseña en segundos

    # Calcula todas las métricas obligatorias del contrato §9
    metrics = {
        "model": "Naive Bayes",
        "accuracy":          round(accuracy_score(test_labels, predictions), 4),
        "precision_macro":   round(precision_score(test_labels, predictions, average="macro"), 4),
        "recall_macro":      round(recall_score(test_labels, predictions, average="macro"), 4),
        "f1_macro":          round(f1_score(test_labels, predictions, average="macro"), 4),
        "f1_weighted":       round(f1_score(test_labels, predictions, average="weighted"), 4),
        "f1_per_class":      {                # F1 por clase en orden contractual
            LABEL_ID_TO_NAME[i]: round(v, 4)
            for i, v in enumerate(f1_score(test_labels, predictions, average=None))
        },
        "inference_time_s":  round(inference_time, 6),
        "confusion_matrix":  confusion_matrix(test_labels, predictions).tolist(),
    }

    print(f"Naive Bayes — Accuracy: {metrics['accuracy']} | F1 macro: {metrics['f1_macro']}")
    return metrics


def save_naive_bayes(vectorizer, model, metrics: dict) -> None:
    """Persiste el vectorizador, el modelo y las métricas en artifacts/ para la interfaz."""
    vec_path   = ARTIFACTS / "vectorizers" / "tfidf_vectorizer.joblib"
    model_path = ARTIFACTS / "models"      / "naive_bayes.joblib"
    metrics_path = ARTIFACTS / "metrics"   / "naive_bayes_metrics.json"

    vec_path.parent.mkdir(parents=True, exist_ok=True)      # crea carpeta si no existe
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, vec_path)        # guarda el vectorizador ajustado
    joblib.dump(model, model_path)           # guarda el modelo entrenado
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)  # guarda métricas legibles

    print(f"Artefactos Naive Bayes guardados en artifacts/")


def load_naive_bayes():
    """Carga el vectorizador y el modelo desde disco para inferencia — sin reentrenar."""
    vectorizer = joblib.load(ARTIFACTS / "vectorizers" / "tfidf_vectorizer.joblib")
    model      = joblib.load(ARTIFACTS / "models"      / "naive_bayes.joblib")
    return vectorizer, model