# Observatorio de Opiniones de E-commerce en Español
**Proyecto Final — Módulo de PLN · Maestría en IA · UCB "San Pablo"**
**Entrega:** 13/07/2026 · **Autor:** Guillermo Carvajal Vaca

---

## Descripción
Sistema de PLN de punta a punta que analiza reseñas de productos en español.
Clasifica sentimiento (NEGATIVO / NEUTRO / POSITIVO), descubre temas con LDA
y compara dos enfoques: Naive Bayes (clásico) vs. GRU (neuronal).

---

## Resultados obtenidos

| Modelo | F1 macro |
|---|---|
| Naive Bayes (champion) | 0.6388 |
| GRU | 0.5181 |

Champion: Naive Bayes — el contexto vence a la moda (Clase 12).

---

## Dataset
- Fuente: SetFit/amazon_reviews_multi_es (Hugging Face Hub)
- Muestra balanceada: 9.000 train · 1.500 validation · 998 test
- Mapeo: 1-2 estrellas NEGATIVO · 3 estrellas NEUTRO · 4-5 estrellas POSITIVO

---

## Decisiones técnicas
- NB sobre LogReg: la consigna exige Naive Bayes explícitamente
- GRU sobre Transformer: factibilidad en el plazo disponible
- CPU sobre GPU: TF 2.15 incompatible con CUDA 13.3
- max_vocab=10.000 y max_length=35: criterio OC-02, percentil 75

---

## Limitaciones
1. LDA sobre textos cortos: coherencia potencialmente baja
2. Etiquetas de estrellas: proxy imperfecto del sentimiento
3. Recall de NEUTRO típicamente menor
4. Sin sentimiento por aspecto
5. Sin Transformer afinado
6. Interfaz local sin despliegue

---

## Versiones principales
- tensorflow==2.15.1
- gradio==4.44.1
- scikit-learn==1.9.0
- pandas==2.3.3
- numpy==1.26.4

---

## Instalación y ejecución

1. Crear y activar entorno virtual:
   python -m venv .venv
   .venv\Scripts\Activate.ps1

2. Instalar dependencias:
   pip install -r requirements.txt

3. Entrenar todos los modelos:
   python main.py

4. Lanzar la interfaz:
   python src/app/gradio_app.py
   Abrir en el navegador: http://127.0.0.1:7860

---

## Estructura del proyecto

config/settings.py         configuración central del proyecto
data/raw/                  dataset crudo descargado de Hugging Face
data/processed/            CSV balanceado 11.998 registros
artifacts/                 modelos vectorizadores métricas y figuras
src/data/                  carga y validación del dataset
src/preprocessing/         pipelines por rama classic neural lda
src/models/                Naive Bayes GRU LDA
src/evaluation/            métricas y gráficos
src/inference/             predictor de inferencia
src/app/                   interfaz Gradio
main.py                    entrenamiento end-to-end
requirements.txt           dependencias del proyecto