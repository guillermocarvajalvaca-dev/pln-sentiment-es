import json                                  # guarda métricas en formato JSON
import numpy as np                           # operaciones numéricas
import matplotlib.pyplot as plt              # generación de gráficos
import matplotlib                            # configuración del backend
matplotlib.use("Agg")                        # backend sin pantalla: genera PNG sin abrir ventana
from config.settings import ARTIFACTS, LABEL_ID_TO_NAME

# Lista ordenada de nombres de clase según el orden contractual
CLASS_NAMES = [LABEL_ID_TO_NAME[i] for i in sorted(LABEL_ID_TO_NAME.keys())]


def plot_confusion_matrix(cm: list, model_name: str) -> None:
    """
    Genera y guarda la matriz de confusión como PNG en artifacts/figures/.
    La matriz muestra cuántas predicciones fueron correctas e incorrectas por clase.
    Filas = clase real, Columnas = clase predicha.
    """
    cm_array = np.array(cm)                  # convierte la lista a array numpy para operar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # dos subgráficos: absoluto y normalizado

    for ax, data, title, fmt in zip(
        axes,
        [cm_array, cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)],  # absoluto y normalizado
        [f"{model_name} — Matriz absoluta", f"{model_name} — Matriz normalizada"],
        ["d", ".2f"]                         # formato entero para absoluta, decimal para normalizada
    ):
        im = ax.imshow(data, cmap="Blues")   # mapa de color azul: más oscuro = más predicciones
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")  # etiquetas en eje X rotadas
        ax.set_yticklabels(CLASS_NAMES)      # etiquetas en eje Y
        ax.set_xlabel("Clase predicha")      # eje X = lo que predijo el modelo
        ax.set_ylabel("Clase real")          # eje Y = la etiqueta verdadera
        ax.set_title(title)
        plt.colorbar(im, ax=ax)              # barra de color para interpretar la intensidad

        # Escribe el valor numérico dentro de cada celda de la matriz
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black")

    plt.tight_layout()                       # ajusta el espaciado entre subgráficos
    out_path = ARTIFACTS / "figures" / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  # guarda con alta resolución
    plt.close()                              # libera memoria cerrando la figura
    print(f"Matriz de confusión guardada: {out_path}")


def plot_training_curves(history) -> None:
    """
    Genera y guarda las curvas de entrenamiento de la GRU como PNG.
    Muestra accuracy y loss en train y validation por época.
    Sirve para verificar que EarlyStopping funcionó correctamente.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curva de accuracy: sube si el modelo aprende correctamente
    axes[0].plot(history.history["accuracy"],     label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("GRU — Accuracy por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Curva de loss: baja si el modelo mejora; si sube en validation hay sobreajuste
    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title("GRU — Loss por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    out_path = ARTIFACTS / "figures" / "training_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curvas de entrenamiento guardadas: {out_path}")


def save_comparison(nb_metrics: dict, gru_metrics: dict) -> None:
    """
    Guarda la tabla comparativa de ambos modelos y determina el champion por F1 macro.
    El champion es el modelo con mayor F1 macro — métrica principal del contrato §9.
    """
    # Determina el champion comparando F1 macro de ambos modelos
    champion = "Naive Bayes" if nb_metrics["f1_macro"] >= gru_metrics["f1_macro"] else "GRU"

    comparison = {
        "Naive Bayes": nb_metrics,
        "GRU":         gru_metrics,
    }

    # Guarda la tabla comparativa completa
    comp_path = ARTIFACTS / "metrics" / "comparison.json"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Guarda champion.json — archivo que gobierna la interfaz (contrato C3)
    # Prohibido hardcodear el champion: siempre se lee desde este archivo
    champion_path = ARTIFACTS / "metrics" / "champion.json"
    with open(champion_path, "w", encoding="utf-8") as f:
        json.dump({
            "champion":       champion,
            "f1_macro_nb":    nb_metrics["f1_macro"],
            "f1_macro_gru":   gru_metrics["f1_macro"],
        }, f, indent=2, ensure_ascii=False)

    print(f"\nCampeón: {champion}")
    print(f"F1 macro NB:  {nb_metrics['f1_macro']}")
    print(f"F1 macro GRU: {gru_metrics['f1_macro']}")
    print(f"champion.json guardado en artifacts/metrics/")