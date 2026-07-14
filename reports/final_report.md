# Informe Final — Observatorio de Opiniones de E-commerce en Español
**Módulo de PLN · Maestría en IA · UCB "San Pablo"**
**Autor:** Guillermo Carvajal Vaca · **Fecha:** 13/07/2026

---

## 1. Problema y caso de uso

El sistema analiza reseñas de productos en español de e-commerce.
Resuelve tres preguntas automáticamente:
- ¿El cliente está satisfecho? (sentimiento)
- ¿De qué habla la reseña? (tema dominante)
- ¿Qué modelo predice mejor? (comparación clásico vs. neuronal)

Usuario objetivo: equipos de atención al cliente y gestión de productos
que necesitan procesar grandes volúmenes de reseñas sin leerlas manualmente.

---

## 2. Datos

- Fuente: SetFit/amazon_reviews_multi_es (espejo del MARC en Hugging Face)
- El dataset original amazon_reviews_multi fue retirado (defunct) — se usó el espejo oficial
- Splits nativos: train 200.000 · validation 5.000 · test 5.000
- Muestra balanceada: 9.000 train · 1.500 validation · 998 test
- Balance: 20% por estrella por construcción del MARC original
- Mapeo de etiquetas: 1-2 estrellas NEGATIVO · 3 estrellas NEUTRO · 4-5 estrellas POSITIVO
- Se eliminaron 2 registros del test por duplicación con train (Stop-the-Line)
- Rango real de label: 0-4 confirmado en auditoría

---

## 3. Pipeline

### Preprocesamiento
Tres ramas diferenciadas según el modelo:

- Rama A (clásica): minúsculas + marcado de negaciones (NEG_no, NEG_nunca) + limpieza
- Rama B (neuronal): minúsculas + tokenizador Keras ajustado solo con train + padding a 35 tokens
- Rama C (LDA): minúsculas + eliminación de stopwords NLTK + bigramas

Regla anti-fuga aplicada: TfidfVectorizer, Tokenizer y CountVectorizer
se ajustaron exclusivamente con el split de entrenamiento.

### Representación numérica
- Rama A: TF-IDF con unigramas y bigramas (ngram_range=1,2)
- Rama B: embeddings aprendidos de dimensión 100 por la red neuronal

### Modelado de temas (LDA)
- 5 temas descubiertos sobre el corpus de entrenamiento
- Temas identificados: Valoración general · Rendimiento y batería ·
  Relación calidad-precio · Envío y logística · Accesorios y pantalla
- Limitación: reseñas cortas producen temas menos coherentes

### Clasificación de sentimiento
- Modelo clásico: TF-IDF + Multinomial Naive Bayes
- Modelo neuronal: Embedding(10000,100) + GRU(64,dropout=0.30) + Dense(3,softmax)
- EarlyStopping con patience=3 sobre val_loss

---

## 4. Comparación de modelos

| Modelo | F1 macro | F1 NEGATIVO | F1 NEUTRO | F1 POSITIVO |
|---|---|---|---|---|
| Naive Bayes (champion) | 0.6388 | ver JSON | ver JSON | ver JSON |
| GRU | 0.5181 | ver JSON | ver JSON | ver JSON |

Champion: Naive Bayes por F1 macro superior.
Interpretación: el modelo clásico bien entrenado en el dominio supera
a la red neuronal — principio "contexto vence a la moda" de la Clase 12.
Las métricas exactas por clase se encuentran en artifacts/metrics/.

---

## 5. Análisis de errores

Casos difíciles identificados en el smoke test:
- Reseña mixta: "funciona muy bien pero llegó tarde" → NEGATIVO (56%)
  El modelo captura el aspecto negativo del envío con mayor peso
- Reseña muy corta: "Mal" → Señal de Duda activada (NB y GRU discrepan)
- Reseña neutra: confianza 58.7% — cerca del umbral de baja confianza (60%)

Patrones de error esperados:
- Ironía: "tan bueno que lo devolví" — el modelo no detecta ironía
- Negaciones complejas: se mitigan con el marcado NEG_ en Rama A
- Textos muy cortos: tema LDA no determinable (fallback M1 activado)

---

## 6. Limitaciones

1. LDA sobre textos cortos: coherencia potencialmente baja; mitigado con bigramas
2. Etiquetas derivadas de estrellas: proxy imperfecto del sentimiento
3. Recall de NEUTRO típicamente menor por ambigüedad de la clase
4. Sin sentimiento por aspecto: fuera del alcance temporal
5. Sin Transformer afinado: GRU cumple el requisito; BETO como mejora futura
6. Interfaz local sin despliegue en nube
7. GPU no disponible: TF 2.15 incompatible con CUDA 13.3; entrenado en CPU

---

## 7. Mejoras futuras

- Afinar BETO o RoBERTuito para comparación con Transformer
- Implementar sentimiento por aspecto (precio, envío, calidad)
- Desplegar la interfaz en Hugging Face Spaces
- Agregar modo lote CSV en la interfaz
- Calcular coherencia Cv para validar cuantitativamente los temas LDA