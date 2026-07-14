# Fuente para NotebookLM — Proyecto Final PLN
## Observatorio de Opiniones de E-commerce en Español
**Maestría en IA · UCB San Pablo · Autor: Guillermo Carvajal Vaca · 13/07/2026**

---

## 1. El problema y el caso de uso

El sistema analiza reseñas de productos en español de e-commerce de forma automática.
Resuelve tres preguntas sin intervención humana:
- ¿El cliente está satisfecho? (sentimiento: NEGATIVO, NEUTRO, POSITIVO)
- ¿De qué habla la reseña? (tema dominante por LDA)
- ¿Qué modelo predice mejor? (comparación Naive Bayes vs GRU)

Usuario objetivo: equipos de atención al cliente que procesan miles de reseñas diariamente.
Caso de uso: Voz del Cliente — E-commerce (Caso 1 de la consigna).

---

## 2. Los datos

- Fuente: SetFit/amazon_reviews_multi_es (espejo del MARC en Hugging Face)
- El dataset original amazon_reviews_multi fue retirado (defunct) — decisión documentada en el contrato
- Splits nativos: train 200.000, validation 5.000, test 5.000
- Muestra balanceada final: 9.000 train, 1.500 validation, 998 test
- Mapeo de etiquetas: 1-2 estrellas = NEGATIVO, 3 estrellas = NEUTRO, 4-5 estrellas = POSITIVO
- Auditoría: sin nulos, 2 duplicados eliminados entre train y test (Stop-the-Line activado)
- Longitud media de reseñas: 28 palabras, percentil 95: 72 palabras

---

## 3. El pipeline paso a paso

### Preprocesamiento — 3 ramas diferenciadas
- Rama A (clásica): minúsculas + marcado de negaciones (NEG_no, NEG_nunca) + limpieza
- Rama B (neuronal): minúsculas + tokenizador Keras ajustado solo con train + padding a 72 tokens
- Rama C (LDA): minúsculas + eliminación de stopwords NLTK español + bigramas

Regla anti-fuga aplicada: TfidfVectorizer, Tokenizer y CountVectorizer
se ajustaron exclusivamente con el split de entrenamiento. Nunca se usó test para tuning.

### Representación numérica
- Rama A: TF-IDF con unigramas y bigramas (ngram_range 1,2) — bolsa de palabras ponderada
- Rama B: embeddings FastText preentrenados en español (cc.es.300.vec, 300 dimensiones)
  Cobertura del vocabulario: 98.3% de las 15.193 palabras únicas del corpus

### Modelado de temas LDA
- 5 temas descubiertos sobre el corpus de entrenamiento
- Temas identificados:
  - Tema 0: Valoración general (bien, precio, cumple, calidad, mal)
  - Tema 1: Rendimiento y batería (funciona, luz, batería, carga, problema)
  - Tema 2: Relación calidad-precio (calidad, precio, buena calidad, esperaba, foto)
  - Tema 3: Envío y logística (llegado, llegó, caja, amazon, pedido, vendedor)
  - Tema 4: Accesorios y pantalla (pantalla, cristal, protector, funda, poner)
- Limitación: reseñas cortas producen temas menos coherentes — mitigado con bigramas

### Clasificación de sentimiento
- Modelo clásico: TF-IDF + Multinomial Naive Bayes (baseline primero — principio Clase 12)
- Modelo neuronal: Embedding FastText (300d, trainable=False) + SpatialDropout1D(0.2) + GRU(64) + Dense(3,softmax)
- Entrenado en Google Colab con GPU T4 — 20 épocas, EarlyStopping patience=3, ReduceLROnPlateau

---

## 4. Comparación de modelos — métricas reales

| Métrica | Naive Bayes (CPU) | GRU + FastText (GPU T4) |
|---|---|---|
| Accuracy | 0.6362 | 0.6362 |
| F1 macro | 0.6388 | 0.6391 |
| F1 NEGATIVO | 0.6578 | 0.6575 |
| F1 NEUTRO | 0.5451 | 0.5297 |
| F1 POSITIVO | 0.7135 | 0.7300 |
| Tiempo inferencia | ~0 s | 0.001 s |
| Dispositivo | CPU local | GPU Colab T4 |

Champion: GRU por F1 macro 0.6391 vs 0.6388 (diferencia de 0.0003 — empate técnico).

Interpretación clave: sin embeddings preentrenados la GRU obtenía F1 de 0.19.
Con FastText en español la GRU alcanzó al Naive Bayes — confirma que los embeddings
preentrenados son esenciales para PLN en español con datasets pequeños.

---

## 5. Desafíos y decisiones tomadas

- Dataset original MARC defunct: se usó el espejo SetFit en Hugging Face
- TF 2.15 incompatible con CUDA 13.3 local: se entrenó en Google Colab GPU T4
- GRU sin embeddings no convergía (F1=0.19): se incorporaron embeddings FastText preentrenados
- TF 2.18 vs TF 2.15 incompatibilidad de formato .keras: se usó formato SavedModel
- Clase NEUTRO tiene F1 menor (0.53): esperado — reseñas de 3 estrellas son ambiguas por naturaleza

---

## 6. La interfaz Gradio

- Modo individual: el usuario pega una reseña y obtiene sentimiento, probabilidades, tema y comparación de modelos
- Señal de Duda: banner visible cuando Naive Bayes y GRU discrepan — revisión humana recomendada
- Advertencia de baja confianza: cuando la probabilidad máxima es menor al 60%
- 4 ejemplos precargados para la demo en vivo
- Champion leído desde champion.json — nunca hardcodeado

---

## 7. Conclusiones, limitaciones y mejoras futuras

### Conclusiones
- Los embeddings preentrenados en español son imprescindibles para la GRU con datasets pequeños
- El Naive Bayes es competitivo con la GRU — confirma el principio de la Clase 12: contexto vence a la moda
- La Señal de Duda es el mecanismo más valioso para casos ambiguos en producción
- F1 NEUTRO es el más bajo en ambos modelos — característica del dominio, no del modelo

### Limitaciones
1. Sin sentimiento por aspecto (precio, envío, calidad por separado)
2. Sin Transformer afinado (BETO/RoBERTuito) — GRU cumple el requisito de la consigna
3. Interfaz local sin despliegue en nube
4. Etiquetas derivadas de estrellas: proxy imperfecto del sentimiento real

### Mejoras futuras
- Afinar BETO o RoBERTuito para comparación con Transformer
- Implementar sentimiento por aspecto
- Desplegar en Hugging Face Spaces
- Agregar modo lote CSV en la interfaz
- Embeddings entrenables (trainable=True) con más datos
