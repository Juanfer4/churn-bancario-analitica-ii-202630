"""Figuras del modelado y de la evaluación final en test.

Usan el estilo único definido en `graficos.py`. Cada función devuelve la figura para
guardarla con `graficos.guardar_figura`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn.graficos import (COLOR_EJE, COLOR_SE_QUEDA, COLOR_SE_VA, MAPA_SECUENCIAL, TINTA_PRIMARIA, TINTA_SECUNDARIA,
                            TINTA_TENUE, decimal, entero, porcentaje, rejilla, texto)

COLORES_ESTRATEGIA = {"ninguna": COLOR_SE_QUEDA, "pesos": COLOR_SE_VA, "SMOTE": "#1baf7a"}


def auc_cv_por_modelo(tabla: pd.DataFrame, titulo: str) -> plt.Figure:
    """AUC de validación cruzada (media ± 1 desviación entre pliegues) y AUC de train por
    modelo. La distancia entre ambos puntos es la brecha que delata sobreajuste."""
    datos_ = tabla.sort_values("auc_cv")
    posiciones = np.arange(len(datos_))
    fig, eje = plt.subplots(figsize=(8.5, 0.5 * len(datos_) + 1.4))
    eje.errorbar(datos_["auc_cv"], posiciones, xerr=datos_["auc_cv_desv"], fmt="o", color=COLOR_SE_QUEDA,
                 ecolor=COLOR_SE_QUEDA, elinewidth=1.5, capsize=3, markersize=7, label="Validación cruzada (± 1 desv.)")
    eje.scatter(datos_["auc_train"], posiciones, marker="o", s=45, facecolors="none",
                edgecolors=TINTA_SECUNDARIA, linewidths=1.3, label="Train", zorder=3)
    for i, (cv, desv) in enumerate(zip(datos_["auc_cv"], datos_["auc_cv_desv"])):
        eje.text(cv, i + 0.22, decimal(cv, 4), ha="center", va="bottom", fontsize=8, color=TINTA_SECUNDARIA)
    eje.set_yticks(posiciones, datos_.index)
    eje.set_xlabel("AUC")
    eje.set_title(titulo, loc="left")
    eje.grid(axis="y", visible=False)
    eje.legend(loc="lower right")
    eje.set_ylim(-0.6, len(datos_) - 0.2)
    fig.tight_layout()
    return fig


def efecto_hiperparametros(paneles: dict[str, tuple[pd.DataFrame, str]], titulo: str,
                           ncols: int = 4) -> plt.Figure:
    """Para cada modelo, el mejor AUC de CV que se logra con cada valor de su hiperparámetro
    principal (optimizando el resto), con una línea por estrategia de desbalance."""
    fig, ejes = rejilla(len(paneles), ncols, ancho=3.9, alto=3.0)
    for eje, (nombre, (resultados, parametro)) in zip(ejes, paneles.items()):
        res = resultados.copy()
        res["_valor"] = res[parametro].map(lambda v: "sin límite" if _es_nulo(v) else _formatear_valor(v))
        res["_clave"] = _orden_valor(res[parametro])
        orden = list(dict.fromkeys(res.sort_values("_clave")["_valor"]))
        for estrategia, grupo in res.groupby("estrategia"):
            mejor = grupo.groupby("_valor")["auc_cv"].max().reindex(orden)
            eje.plot(range(len(orden)), mejor.values, marker="o", markersize=5, linewidth=2,
                     color=COLORES_ESTRATEGIA[estrategia], label=estrategia)
        eje.set_xticks(range(len(orden)), [texto(v) for v in orden], rotation=30, ha="right")
        eje.set_title(nombre, loc="left", fontsize=10)
        eje.set_xlabel(parametro, fontsize=8.5)
        eje.set_ylabel("AUC CV")
        eje.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: decimal(v, 4)))
    manijas = [plt.Line2D([0], [0], color=c, marker="o", linewidth=2) for c in COLORES_ESTRATEGIA.values()]
    fig.legend(manijas, [f"Desbalance: {e}" for e in COLORES_ESTRATEGIA], loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(titulo, y=1.06)
    fig.tight_layout()
    return fig


def _orden_valor(serie: pd.Series) -> pd.Series:
    """Clave de orden para valores de hiperparámetros mezclados (números, textos, tuplas)."""
    def clave(v):
        if _es_nulo(v):
            return -2.0  # "sin límite" (None) va primero
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, tuple):
            return float(sum(v)) + len(v) / 1000
        return -1.0  # textos como "scale" van primero
    return serie.map(clave)


def _es_nulo(v) -> bool:
    """None o NaN (pandas convierte None en NaN en columnas numéricas)."""
    return v is None or (isinstance(v, float) and np.isnan(v))


def _formatear_valor(v) -> str:
    """Valor de hiperparámetro legible: enteros sin decimales y coma decimal."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:g}".replace(".", ",")
    return str(v)


# ---------------------------------------------------------------------------
# Gráficos de la evaluación final
# ---------------------------------------------------------------------------
COLORES_MODELO = ["#2a78d6", "#eb6834", "#1baf7a"]  # hasta tres modelos principales (validados)


def estilos_modelos(nombres: list[str], referencia_gris: str | None):
    """Color fijo por posición para los modelos principales; el de referencia, gris discontinuo."""
    estilos, k = {}, 0
    for nombre in nombres:
        if nombre == referencia_gris:
            estilos[nombre] = {"color": TINTA_TENUE, "linestyle": "--"}
        else:
            estilos[nombre] = {"color": COLORES_MODELO[k], "linestyle": "-"}
            k += 1
    return estilos


def curvas_roc_pr(y_real, probabilidades: dict, titulo: str, referencia_gris: str | None = None) -> plt.Figure:
    """Curva ROC (sensibilidad frente a tasa de falsas alarmas) y curva precisión-captura
    de la clase churn, lado a lado, para todos los modelos en test."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    estilos = estilos_modelos(list(probabilidades), referencia_gris)
    fig, (eje_roc, eje_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for nombre, prob in probabilidades.items():
        fpr, tpr, _ = roc_curve(y_real, prob)
        eje_roc.plot(fpr, tpr, linewidth=2, label=f"{nombre} (AUC {decimal(roc_auc_score(y_real, prob), 4)})", **estilos[nombre])
        precision, captura, _ = precision_recall_curve(y_real, prob)
        eje_pr.plot(captura, precision, linewidth=2, label=f"{nombre} (AP {decimal(average_precision_score(y_real, prob), 4)})", **estilos[nombre])
    eje_roc.plot([0, 1], [0, 1], color=COLOR_EJE, linewidth=1)
    eje_roc.set_xlabel("Tasa de falsas alarmas (1 - especificidad)")
    eje_roc.set_ylabel("Sensibilidad (churners detectados)")
    eje_roc.set_title("Curva ROC", loc="left")
    eje_roc.legend(loc="lower right")
    tasa = float(np.mean(y_real))
    eje_pr.axhline(tasa, color=COLOR_EJE, linewidth=1)
    eje_pr.text(0.98, tasa + 0.02, f"azar: {porcentaje(tasa)}", fontsize=8, ha="right", color=TINTA_SECUNDARIA)
    eje_pr.set_xlabel("Captura de churners (sensibilidad)")
    eje_pr.set_ylabel("Precisión (churners entre los señalados)")
    eje_pr.set_title("Curva precisión-captura", loc="left")
    eje_pr.legend(loc="lower left", bbox_to_anchor=(0.0, 0.12))
    fig.suptitle(titulo, y=1.02)
    fig.tight_layout()
    return fig


def matrices_confusion(resultados: dict[str, dict], titulo: str) -> plt.Figure:
    """Matrices de confusión en conteos, una por modelo, con las métricas clave debajo."""
    fig, ejes = rejilla(len(resultados), min(4, len(resultados)), ancho=3.6, alto=3.6)
    for eje, (nombre, m) in zip(ejes, resultados.items()):
        matriz = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        eje.imshow(matriz, cmap=MAPA_SECUENCIAL, vmin=0, vmax=matriz.max())
        for i in range(2):
            for j in range(2):
                eje.text(j, i, entero(matriz[i, j]), ha="center", va="center", fontsize=12,
                         color="white" if matriz[i, j] > 0.55 * matriz.max() else TINTA_PRIMARIA)
        eje.set_xticks([0, 1], ["Se queda", "Se va"])
        eje.set_yticks([0, 1], ["Se queda", "Se va"])
        eje.set_xlabel("Predicción")
        eje.set_ylabel("Realidad")
        eje.grid(False)
        eje.set_title(nombre, loc="left", fontsize=10)
        eje.text(0.5, -0.42, f"Sensibilidad {porcentaje(m['sensibilidad'])} | Precisión {porcentaje(m['precision'])}\n"
                             f"Umbral {decimal(m['umbral'], 3)}", transform=eje.transAxes, ha="center",
                 va="top", fontsize=8.5, color=TINTA_SECUNDARIA)
    fig.suptitle(titulo, y=1.03)
    fig.tight_layout()
    return fig


def diagrama_calibracion(tablas: dict[str, pd.DataFrame], titulo: str, referencia_gris: str | None = None) -> plt.Figure:
    """Probabilidad promedio predicha frente a tasa real de churn por decil. Sobre la
    diagonal, el modelo está bien calibrado."""
    estilos = estilos_modelos(list(tablas), referencia_gris)
    fig, eje = plt.subplots(figsize=(6.5, 6))
    eje.plot([0, 1], [0, 1], color=COLOR_EJE, linewidth=1)
    for nombre, tabla in tablas.items():
        eje.plot(tabla["prob_media"], tabla["tasa_real"], marker="o", markersize=5, linewidth=2,
                 label=nombre, **estilos[nombre])
    eje.set_xlabel("Probabilidad de churn predicha (promedio del decil)")
    eje.set_ylabel("Tasa real de churn en el decil")
    eje.set_xlim(-0.02, 1.02)
    eje.set_ylim(-0.02, 1.02)
    eje.set_title(titulo, loc="left")
    eje.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gráficos de negocio
# ---------------------------------------------------------------------------
