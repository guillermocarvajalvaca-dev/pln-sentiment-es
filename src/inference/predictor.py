import json                                  # lee champion.json y topic_names.json
import numpy as np                           # operaciones numéricas sobre probabilidades
from config.settings import ARTIFACTS, CONFIDENCE_THRESHOLD, LABEL_ID_TO_NAME
from src.preprocessing.classic import preprocess_classic       # preprocesamiento Rama A
from src.preprocessing.neural import texts_to_padded_sequences, load_tokenizer  # preprocesamiento Rama B
from src.preprocessing.lda import preprocess_lda               # preprocesamiento Rama C
from src.models.naive_bayes_model import load_naive_bayes      # carga NB desde disco
from src.models.gru_model import load_gru                      # carga GRU desde disco
from src.models.lda_model import load_lda, assign_topic, load_topic_names  # carga LDA desde disco


class Predictor:
    """
    Orquestador de inferencia — implementa el flujo contractual §10.
    Carga todos los artefactos UNA sola vez al inicializar.
    Nunca reentrena: solo lee modelos desde disco.
    """

    def __init__(self):
        """Carga todos los artefactos desde disco al iniciar la aplicación."""
        print("Cargando artefactos desde disco...")

        # Carga el vectorizador TF-IDF y el modelo Naive Bayes
        self.vectorizer_nb, self.model_nb = load_naive_bayes()

        # Carga el tokenizador neuronal y el modelo GRU
        self.tokenizer = load_tokenizer()
        self.model_gru = load_gru()

        # Carga el CountVectorizer y el modelo LDA
        self.vectorizer_lda, self.model_lda = load_lda()

        # Carga los nombres humanos asignados a los temas
        self.topic_names = load_topic_names()

        # Lee el champion desde champion.json — nunca hardcodeado (contrato C3)
        champion_path = ARTIFACTS / "metrics" / "champion.json"
        with open(champion_path, "r", encoding="utf-8") as f:
            self.champion_data = json.load(f)
        self.champion = self.champion_data["champion"]  # "Naive Bayes" o "GRU"

        print(f"Artefactos cargados. Champion: {self.champion}")

    def predict(self, text: str) -> dict:
        """
        Ejecuta el flujo completo de inferencia sobre un texto nuevo.
        Devuelve sentimiento, probabilidades, tema, Señal de Duda y advertencias.
        """
        # Validación de entrada: rechaza texto vacío o solo espacios
        if not text or text.strip() == "":
            return {"error": "Texto vacío — ingrese una reseña para analizar"}

        # --- RAMA A: Naive Bayes ---
        text_classic = preprocess_classic(text)          # preprocesa para TF-IDF
        X_nb = self.vectorizer_nb.transform([text_classic])  # vectoriza con TF-IDF
        pred_nb = int(self.model_nb.predict(X_nb)[0])        # clase predicha por NB
        probs_nb = self.model_nb.predict_proba(X_nb)[0].tolist()  # probabilidades por clase

        # --- RAMA B: GRU ---
        text_neural = text.lower().strip()               # preprocesamiento mínimo para GRU
        X_gru = texts_to_padded_sequences([text_neural], self.tokenizer)  # secuencia padded
        # SavedModel usa infer() en lugar de predict() — compatible con ambos formatos
        if hasattr(self.model_gru, 'predict'):
            probs_gru_raw = self.model_gru.predict(X_gru, verbose=0)[0]  # modelo Keras
        else:
            infer = self.model_gru.signatures['serving_default']          # SavedModel
            import tensorflow as tf
            input_tensor = tf.constant(X_gru, dtype=tf.float32)          # convierte a tensor
            output = infer(input_tensor)                                  # inferencia
            probs_gru_raw = list(output.values())[0][0].numpy()          # extrae probabilidades
        probs_gru = [float(p) for p in probs_gru_raw]   # convierte a lista Python
        pred_gru = int(np.argmax(probs_gru))             # clase con mayor probabilidad

        # --- RAMA C: LDA ---
        text_lda = preprocess_lda(text)                  # preprocesa eliminando stopwords
        topic_id, topic_probs = assign_topic(            # asigna tema dominante
            text_lda, self.vectorizer_lda, self.model_lda
        )

        # Nombre humano del tema — fallback M1 si texto insuficiente
        if topic_id == -1:
            topic_name = "no determinable (texto insuficiente)"
            topic_words = []
        else:
            topic_key = f"topic_{topic_id}"
            topic_name = self.topic_names.get(topic_key, f"Tema {topic_id}")
            topic_words = []                             # se puede extender con top words

        # --- CAPA DE DECISIÓN ---
        # Selecciona probabilidades del champion para confianza y predicción final
        if self.champion == "Naive Bayes":
            probs_champion = probs_nb
            pred_champion = pred_nb
        else:
            probs_champion = probs_gru
            pred_champion = pred_gru

        confianza = max(probs_champion)                  # probabilidad máxima del champion

        # Señal de Duda: se activa cuando NB y GRU discrepan — revisión humana recomendada
        flag_duda = (pred_nb != pred_gru)

        # Advertencia de baja confianza: se activa si la confianza es menor al umbral
        flag_baja_confianza = (confianza < CONFIDENCE_THRESHOLD)

        return {
            "sentimiento":         LABEL_ID_TO_NAME[pred_champion],  # clase final del champion
            "confianza":           round(confianza, 4),              # probabilidad máxima
            "probabilidades":      {                                  # probabilidades por clase
                LABEL_ID_TO_NAME[i]: round(p, 4)
                for i, p in enumerate(probs_champion)
            },
            "prediccion_nb":       LABEL_ID_TO_NAME[pred_nb],        # predicción de Naive Bayes
            "prediccion_gru":      LABEL_ID_TO_NAME[pred_gru],       # predicción de GRU
            "probs_nb":            {LABEL_ID_TO_NAME[i]: round(p, 4) for i, p in enumerate(probs_nb)},
            "probs_gru":           {LABEL_ID_TO_NAME[i]: round(p, 4) for i, p in enumerate(probs_gru)},
            "champion":            self.champion,                     # modelo ganador
            "tema_id":             topic_id,                         # ID numérico del tema
            "tema_nombre":         topic_name,                       # nombre humano del tema
            "flag_duda":           flag_duda,                        # True si NB ≠ GRU
            "flag_baja_confianza": flag_baja_confianza,              # True si confianza < 0.60
        }