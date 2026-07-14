# Observatorio de Opiniones de E-commerce en Español
**Proyecto Final — Módulo de PLN · Maestría en IA**

---

## Descripción
Sistema de PLN de punta a punta que analiza reseñas de productos en español.
Clasifica sentimiento (NEGATIVO / NEUTRO / POSITIVO), descubre temas con LDA
y compara dos enfoques: Naive Bayes (clásico) vs. GRU (neuronal).

---

## Resultados obtenidos

| Modelo | F1 macro |
|---|---|
| Naive Bayes | 0.6388 |
| GRU + FastText (champion) | 0.6391 |

Champion: GRU — embeddings preentrenados en español fueron determinantes.

---

## Dataset
- Fuente: SetFit/amazon_reviews_multi_es (Hugging Face Hub)
- Muestra balanceada: 9.000 train · 1.500 validation · ~1.000 test
- Mapeo: 1-2 estrellas NEGATIVO · 3 estrellas NEUTRO · 4-5 estrellas POSITIVO

---

## Decisiones técnicas
- NB sobre LogReg: la consigna exige Naive Bayes explícitamente
- GRU sobre Transformer: factibilidad en el plazo disponible
- CPU local + GPU Colab: entrenamiento GRU en Colab T4
- max_vocab=20.000 y max_length=72: percentil 95 del corpus

---

## Limitaciones
1. LDA sobre textos cortos: coherencia potencialmente baja
2. Etiquetas de estrellas: proxy imperfecto del sentimiento
3. Recall de NEUTRO típicamente menor
4. Sin sentimiento por aspecto
5. Sin Transformer afinado
6. Interfaz local sin despliegue

---

## Instalación y ejecución

1. Crear y activar entorno virtual:
   python -m venv .venv
   .venv\Scripts\Activate.ps1

2. Instalar dependencias:
   pip install -r requirements.txt

3. Entrenar todos los modelos:
   python main.py

4. Lanzar la interfaz (solo local):
   python src/app/gradio_app.py
   Abrir en navegador: http://127.0.0.1:7860

---

## Estructura del proyecto

config/settings.py         configuración central
data/raw/                  dataset crudo (descargado por el pipeline)
data/processed/            CSV balanceado (generado por el pipeline)
artifacts/                 modelos, vectorizadores, métricas y figuras
src/data/                  carga y validación del dataset
src/preprocessing/         pipelines por rama (classic, neural, lda)
src/models/                Naive Bayes, GRU, LDA
src/evaluation/            métricas y gráficos
src/inference/             predictor de inferencia
src/app/                   interfaz Gradio (localhost únicamente)
main.py                    entrenamiento end-to-end
requirements.txt           dependencias del proyecto
