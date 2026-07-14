import joblib                                # guarda y carga modelos en disco
import json                                  # guarda temas y nombres en formato JSON
import numpy as np                           # operaciones numéricas
from sklearn.feature_extraction.text import CountVectorizer        # convierte texto a matriz de conteos
from sklearn.decomposition import LatentDirichletAllocation        # modelo de temas no supervisado
from config.settings import ARTIFACTS, LDA_CONFIG, RANDOM_STATE


def train_lda(train_texts: list) -> tuple:
    """
    Entrena el modelo LDA SOLO con textos de entrenamiento.
    LDA: descubre temas latentes en el corpus encontrando grupos de palabras que co-ocurren.
    CountVectorizer: usa conteos crudos (no TF-IDF) porque LDA necesita frecuencias absolutas.
    Regla anti-fuga: CountVectorizer se ajusta únicamente con train.
    """
    # CountVectorizer con bigramas para capturar frases temáticas como "atención cliente"
    vectorizer = CountVectorizer(
        ngram_range=LDA_CONFIG["ngram_range"],   # unigramas y bigramas
        max_features=LDA_CONFIG["max_features"], # máximo 5000 términos más frecuentes
        min_df=LDA_CONFIG["min_df"],             # ignora términos que aparecen en menos de 5 documentos
        max_df=LDA_CONFIG["max_df"],             # ignora términos en más del 90% de documentos
    )

    # fit_transform: aprende vocabulario desde train Y transforma train a matriz de conteos
    X_train = vectorizer.fit_transform(train_texts)   # matriz (n_textos, n_términos)

    # LDA: encuentra n_topics grupos de palabras que co-ocurren frecuentemente
    lda = LatentDirichletAllocation(
        n_components=LDA_CONFIG["n_topics"],     # número de temas a descubrir
        random_state=RANDOM_STATE,               # semilla para reproducibilidad
        max_iter=10,                             # iteraciones de optimización
    )
    lda.fit(X_train)                             # ajusta el modelo sobre la matriz de conteos

    print(f"LDA entrenado: {LDA_CONFIG['n_topics']} temas, vocabulario {len(vectorizer.vocabulary_)} términos")
    return vectorizer, lda


def get_topic_words(lda, vectorizer, n_words: int = 10) -> dict:
    """
    Extrae las top-N palabras más representativas de cada tema.
    Estas palabras son la base para asignar nombres humanos a los temas.
    """
    feature_names = vectorizer.get_feature_names_out()  # lista de todos los términos del vocabulario
    topics = {}

    for topic_id, topic_weights in enumerate(lda.components_):
        # Obtiene los índices de las N palabras con mayor peso en este tema
        top_indices = topic_weights.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]   # palabras más representativas
        topics[f"topic_{topic_id}"] = top_words
        print(f"Tema {topic_id}: {', '.join(top_words)}")

    return topics


def assign_topic(text: str, vectorizer, lda) -> tuple:
    """
    Asigna el tema dominante a un texto nuevo durante la inferencia.
    Manejo M1: si el texto queda vacío tras preprocesamiento, devuelve fallback sin excepción.
    """
    if not text or text.strip() == "":       # fallback M1: texto vacío tras eliminar stopwords
        return -1, "no determinable (texto insuficiente)", []

    X = vectorizer.transform([text])         # transforma el texto a vector de conteos
    topic_probs = lda.transform(X)[0]        # probabilidades de pertenencia a cada tema
    topic_id = int(np.argmax(topic_probs))   # tema con mayor probabilidad
    return topic_id, topic_probs.tolist()


def save_lda(vectorizer, lda, topics: dict) -> None:
    """Persiste el vectorizador LDA, el modelo y los temas en artifacts/."""
    vec_path    = ARTIFACTS / "vectorizers" / "count_vectorizer.joblib"
    model_path  = ARTIFACTS / "models"      / "lda_model.joblib"
    topics_path = ARTIFACTS / "metrics"     / "lda_topics.json"

    vec_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    topics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, vec_path)        # guarda el CountVectorizer ajustado
    joblib.dump(lda, model_path)             # guarda el modelo LDA entrenado
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)  # guarda palabras por tema

    print(f"Artefactos LDA guardados en artifacts/")


def load_lda() -> tuple:
    """Carga el CountVectorizer y el modelo LDA desde disco para inferencia."""
    vectorizer = joblib.load(ARTIFACTS / "vectorizers" / "count_vectorizer.joblib")
    lda        = joblib.load(ARTIFACTS / "models"      / "lda_model.joblib")
    return vectorizer, lda


def save_topic_names(topic_names: dict) -> None:
    """
    Guarda los nombres humanos asignados a cada tema tras inspección manual.
    Ejemplo: {"topic_0": "Envío y logística", "topic_1": "Calidad del producto"}
    """
    path = ARTIFACTS / "metrics" / "topic_names.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(topic_names, f, indent=2, ensure_ascii=False)
    print(f"Nombres de temas guardados en {path}")


def load_topic_names() -> dict:
    """Carga los nombres humanos de los temas para mostrarlos en la interfaz."""
    path = ARTIFACTS / "metrics" / "topic_names.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)