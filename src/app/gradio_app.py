import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # raíz del proyecto

import gradio as gr                          # framework para construir la interfaz web
from src.inference.predictor import Predictor  # orquestador de inferencia

# Carga todos los artefactos UNA sola vez al importar el módulo
# Nunca se reentrena aquí — Gradio solo lee modelos desde disco
predictor = Predictor()

# Ejemplos precargados del contrato §11 para la demo en vivo
EJEMPLOS = [
    ["El producto llegó antes de lo previsto y funciona perfectamente."],       # positivo
    ["No funciona, llegó roto y el vendedor nunca respondió."],                 # negativo
    ["Cumple con lo básico, pero esperaba una mejor calidad por el precio."],   # neutro
    ["El producto funciona muy bien, pero llegó dos semanas tarde y la caja estaba dañada."],  # mixto
]


def analizar(texto: str) -> tuple:
    """
    Función principal de la interfaz: recibe un texto y devuelve todos los análisis.
    Esta función es llamada por Gradio cada vez que el usuario presiona Analizar.
    """
    resultado = predictor.predict(texto)     # ejecuta el flujo completo de inferencia

    # Manejo de error: texto vacío
    if "error" in resultado:
        return resultado["error"], "", "", "", "", "", ""

    # Sentimiento principal con confianza
    sentimiento_txt = (
        f"{resultado['sentimiento']} "
        f"(confianza: {resultado['confianza']*100:.1f}%)"
    )

    # Probabilidades por clase formateadas
    probs_txt = "\n".join([
        f"  {clase}: {prob*100:.1f}%"
        for clase, prob in resultado["probabilidades"].items()
    ])

    # Predicciones individuales de cada modelo
    modelos_txt = (
        f"Naive Bayes:  {resultado['prediccion_nb']}\n"
        f"GRU:          {resultado['prediccion_gru']}\n"
        f"Champion:     {resultado['champion']}"
    )

    # Señal de Duda — visible solo cuando NB ≠ GRU
    if resultado["flag_duda"]:
        duda_txt = "⚠️ SEÑAL DE DUDA: los modelos discrepan — revisión humana recomendada"
    else:
        duda_txt = "✅ Modelos coinciden"

    # Advertencia de baja confianza
    if resultado["flag_baja_confianza"]:
        confianza_txt = "⚠️ Confianza baja (< 60%) — resultado poco confiable"
    else:
        confianza_txt = "✅ Confianza aceptable"

    # Tema dominante identificado por LDA
    tema_txt = f"Tema {resultado['tema_id']}: {resultado['tema_nombre']}"

    return (
        sentimiento_txt,                     # salida 1: sentimiento + confianza
        probs_txt,                           # salida 2: probabilidades por clase
        modelos_txt,                         # salida 3: predicciones de cada modelo
        duda_txt,                            # salida 4: Señal de Duda
        confianza_txt,                       # salida 5: advertencia de confianza
        tema_txt,                            # salida 6: tema dominante
    )


# Construcción de la interfaz Gradio
with gr.Blocks(title="Observatorio de Opiniones E-commerce") as demo:

    gr.Markdown("# 🔍 Observatorio de Opiniones de E-commerce en Español")
    gr.Markdown("Analiza reseñas de productos: sentimiento, tema dominante y comparación de modelos.")

    with gr.Row():
        with gr.Column():
            # Entrada de texto
            texto_input = gr.Textbox(
                label="Reseña del producto",
                placeholder="Pega aquí una reseña en español...",
                lines=4,
            )
            btn_analizar = gr.Button("Analizar", variant="primary")  # botón principal

        with gr.Column():
            # Salidas del análisis
            out_sentimiento = gr.Textbox(label="Sentimiento principal")
            out_probs       = gr.Textbox(label="Probabilidades por clase")
            out_modelos     = gr.Textbox(label="Predicción por modelo")
            out_duda        = gr.Textbox(label="Señal de Duda")        # contrato §11
            out_confianza   = gr.Textbox(label="Nivel de confianza")
            out_tema        = gr.Textbox(label="Tema dominante (LDA)")

    # Conecta el botón con la función de análisis
    btn_analizar.click(
        fn=analizar,                         # función a ejecutar al presionar el botón
        inputs=[texto_input],                # entrada: el texto del usuario
        outputs=[                            # salidas: todos los componentes de resultado
            out_sentimiento, out_probs, out_modelos,
            out_duda, out_confianza, out_tema
        ],
    )

    # Ejemplos precargados para la demo en vivo
    gr.Examples(
        examples=EJEMPLOS,
        inputs=[texto_input],
        label="Ejemplos de la demo",
    )


if __name__ == "__main__":
    demo.launch(                             # inicia el servidor local de Gradio
        server_name="127.0.0.1",            # localhost: evita el error de acceso en entornos restringidos
        server_port=7860,                    # puerto por defecto de Gradio
        share=False,                         # no genera enlace público (demo offline)
    )