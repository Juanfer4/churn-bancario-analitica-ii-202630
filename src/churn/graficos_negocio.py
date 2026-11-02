"""Figuras de negocio (curvas de ganancia y beneficio) y de explicabilidad del modelo.

Usan el estilo único definido en `graficos.py` y los colores de modelo de `graficos_modelado.py`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from churn.graficos import (COLOR_EJE, COLOR_GRILLA, COLOR_SE_QUEDA, COLOR_SE_VA, SUPERFICIE, TINTA_PRIMARIA,
                            TINTA_SECUNDARIA, TINTA_TENUE, decimal, entero, porcentaje, rejilla)
from churn.graficos_modelado import estilos_modelos

RAMPA_ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # de menor a mayor valor


def curvas_ganancia(curvas: dict[str, pd.DataFrame], tasa_churn: float, capacidades, titulo: str,
                    principal: str, referencia_gris: str | None = None) -> plt.Figure:
    """Curva de ganancia: qué fracción de churners se captura al contactar a una fracción de
    clientes ordenados por riesgo. Incluye el azar (diagonal) y el modelo perfecto."""
    estilos = estilos_modelos(list(curvas), referencia_gris)
    fig, eje = plt.subplots(figsize=(8.5, 5.5))
    x = np.linspace(0, 1, 200)
    eje.plot(x, x, color=COLOR_EJE, linewidth=1)
    eje.plot(x, np.minimum(x / tasa_churn, 1), color=COLOR_EJE, linewidth=1)
    eje.text(0.55, 0.50, "azar", fontsize=8, color=TINTA_SECUNDARIA)
    eje.text(tasa_churn, 1.025, f"modelo perfecto: todos los churners en el {porcentaje(tasa_churn, 0)} de mayor riesgo",
             fontsize=8, color=TINTA_SECUNDARIA, va="bottom")
    eje.set_ylim(-0.03, 1.09)
    for nombre, curva in curvas.items():
        eje.plot(curva["fraccion_contactada"], curva["captura"], linewidth=2, label=nombre, **estilos[nombre])
    curva = curvas[principal]
    for k in capacidades:
        fila = curva.iloc[(curva["fraccion_contactada"] - k).abs().argmin()]
        eje.scatter([fila["fraccion_contactada"]], [fila["captura"]], s=45, zorder=4,
                    color=estilos[principal]["color"], edgecolors=SUPERFICIE, linewidths=1.5)
        eje.annotate(f"{porcentaje(k, 0)}: {porcentaje(fila['captura'])}", (fila["fraccion_contactada"], fila["captura"]),
                     xytext=(8, -22), textcoords="offset points", fontsize=8.5, color=TINTA_PRIMARIA,
                     bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 1, "alpha": 0.85})
    eje.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: porcentaje(v, 0)))
    eje.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: porcentaje(v, 0)))
    eje.set_xlabel("Clientes contactados (de mayor a menor riesgo)")
    eje.set_ylabel("Churners capturados")
    eje.set_title(titulo, loc="left")
    eje.legend(loc="lower right")
    fig.tight_layout()
    return fig


def curvas_beneficio(curvas: dict[float, pd.DataFrame], capacidades, titulo: str,
                     max_fraccion: float = 0.5) -> plt.Figure:
    """Beneficio neto por cliente de la base (en unidades del costo del beneficio) según la
    fracción contactada, para varios valores supuestos de s * r. El punto marca el óptimo."""
    fig, eje = plt.subplots(figsize=(8.5, 5.2))
    for k in capacidades:
        eje.axvline(k, color=COLOR_GRILLA, linewidth=1.2, zorder=0)
        eje.text(k, 1.0, f" {porcentaje(k, 0)}", transform=eje.get_xaxis_transform(), fontsize=8,
                 color=TINTA_SECUNDARIA, va="top")
    eje.axhline(0, color=COLOR_EJE, linewidth=1)
    for color, (rs, curva) in zip(RAMPA_ORDINAL, curvas.items()):
        tramo = curva[curva["fraccion_contactada"] <= max_fraccion]
        eje.plot(tramo["fraccion_contactada"], tramo["beneficio_por_cliente"], color=color, linewidth=2,
                 label=f"s · r = {decimal(rs, 1)}")
        optimo = curva.loc[curva["beneficio_por_cliente"].idxmax()]
        eje.scatter([optimo["fraccion_contactada"]], [optimo["beneficio_por_cliente"]], color=color, s=40,
                    zorder=4, edgecolors=SUPERFICIE, linewidths=1.5)
    eje.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: porcentaje(v, 0)))
    eje.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: decimal(v, 2)))
    eje.set_xlabel("Clientes contactados (de mayor a menor riesgo)")
    eje.set_ylabel("Beneficio neto por cliente de la base\n(en unidades del costo del beneficio)")
    eje.set_title(titulo, loc="left")
    eje.legend(loc="upper right", title="Supuesto")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gráficos de explicabilidad (Fase 5, 5.6 a 5.8)
# ---------------------------------------------------------------------------
MAPA_RIESGO = LinearSegmentedColormap.from_list("riesgo_naranja", ["#fdf0ea", "#f5a37f", COLOR_SE_VA, "#8a3412"])


def importancia_variables(ganancia: pd.Series, permutacion: pd.DataFrame, titulo: str) -> plt.Figure:
    """Dos lecturas de la importancia, con las variables en el mismo orden (el de la
    permutación): ganancia dentro de los árboles y caída del AUC al desordenar la variable."""
    orden = permutacion.sort_values("caida_auc_media").index
    posiciones = np.arange(len(orden))
    fig, (eje_g, eje_p) = plt.subplots(1, 2, figsize=(11.5, 0.36 * len(orden) + 1.6), sharey=True)
    valores_g = ganancia.reindex(orden).fillna(0)
    eje_g.barh(posiciones, valores_g, color=COLOR_SE_QUEDA, height=0.62)
    for i, v in enumerate(valores_g):
        eje_g.text(v + 0.4, i, porcentaje(v / 100, 1), va="center", fontsize=7.5, color=TINTA_SECUNDARIA)
    eje_g.set_xlim(0, valores_g.max() * 1.22)
    eje_g.set_xlabel("Ganancia en los árboles (% del total)")
    eje_g.set_title("Uso dentro del modelo", loc="left", fontsize=10)
    valores_p = permutacion.loc[orden, "caida_auc_media"]
    eje_p.barh(posiciones, valores_p, xerr=permutacion.loc[orden, "caida_auc_desv"], color=COLOR_SE_VA,
               height=0.62, error_kw={"elinewidth": 1, "ecolor": TINTA_SECUNDARIA, "capsize": 2})
    desviaciones = permutacion.loc[orden, "caida_auc_desv"]
    for i, (v, d) in enumerate(zip(valores_p, desviaciones)):  # la etiqueta va después de la barra de error
        eje_p.text(max(v, 0) + d + valores_p.max() * 0.03, i, decimal(v, 4), va="center", fontsize=7.5, color=TINTA_SECUNDARIA)
    eje_p.set_xlim(min(0, valores_p.min()) * 1.1, valores_p.max() * 1.3)
    eje_p.axvline(0, color=TINTA_TENUE, linewidth=1)
    eje_p.set_xlabel("Caída del AUC al desordenar la variable (media de 5 pliegues)")
    eje_p.set_title("Dependencia en clientes no vistos", loc="left", fontsize=10)
    eje_g.set_yticks(posiciones, orden)
    for eje in (eje_g, eje_p):
        eje.grid(axis="y", visible=False)
    fig.suptitle(titulo, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def dependencias_parciales(curvas: dict[str, pd.DataFrame], X: pd.DataFrame, prob_media: float,
                           titulo: str, ncols: int = 3) -> plt.Figure:
    """Dependencia parcial de varias variables. Las marcas inferiores son los deciles de la
    variable en los datos: donde no hay marcas hay pocos clientes y la curva es menos confiable."""
    fig, ejes = rejilla(len(curvas), ncols, ancho=4.3, alto=3.1)
    for eje, (variable, curva) in zip(ejes, curvas.items()):
        eje.axhline(prob_media, color=TINTA_TENUE, linewidth=1, linestyle="--")
        escalones = curva["valor"].nunique() <= 10
        eje.plot(curva["valor"], curva["prob_churn_media"], color=COLOR_SE_VA, linewidth=2,
                 marker="o" if escalones else None, markersize=4)
        deciles = np.quantile(X[variable], np.linspace(0.1, 0.9, 9))
        limites = eje.get_ylim()  # vlines amplía los límites aunque se dibuje en coordenadas del eje
        eje.vlines(deciles, 0, 0.05, color=TINTA_SECUNDARIA, linewidth=1.2, transform=eje.get_xaxis_transform())
        eje.set_ylim(limites)
        eje.set_title(variable, loc="left", fontsize=10)
        # Con rangos pequeños se usa un decimal para que las marcas del eje no se repitan
        decimales = 1 if curva["prob_churn_media"].max() - curva["prob_churn_media"].min() < 0.05 else 0
        eje.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=decimales: porcentaje(v, d)))
        eje.set_ylabel("Prob. de churn promedio")
    ejes[0].text(0.98, prob_media, "promedio", transform=ejes[0].get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=7.5, color=TINTA_TENUE)
    fig.suptitle(titulo, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def dependencia_parcial_conjunta(tabla: pd.DataFrame, variable_x: str, variable_y: str, titulo: str,
                                 minimo_clientes: int = 10) -> plt.Figure:
    """Mapa de la probabilidad de churn promedio según dos variables. Las celdas con menos de
    `minimo_clientes` clientes reales se dejan en blanco: ahí el modelo extrapola."""
    malla = tabla.pivot(index=variable_y, columns=variable_x, values="prob_churn_media")
    conteo = tabla.pivot(index=variable_y, columns=variable_x, values="clientes_en_celda")
    valores = np.ma.masked_where(conteo.to_numpy() < minimo_clientes, malla.to_numpy())
    fig, eje = plt.subplots(figsize=(9, 5.6))
    eje.set_facecolor(SUPERFICIE)
    imagen = eje.pcolormesh(malla.columns, malla.index, valores, cmap=MAPA_RIESGO, vmin=0, vmax=1, shading="nearest")
    barra = fig.colorbar(imagen, ax=eje, pad=0.02)
    barra.set_label("Probabilidad de churn promedio")
    barra.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: porcentaje(v, 0)))
    barra.outline.set_visible(False)
    eje.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: entero(v)))
    eje.set_xlabel(variable_x)
    eje.set_ylabel(variable_y)
    eje.grid(False)
    eje.set_title(titulo, loc="left")
    eje.text(0.99, 0.02, f"En blanco: menos de {minimo_clientes} clientes reales (extrapolación)",
             transform=eje.transAxes, ha="right", fontsize=8, color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return fig
