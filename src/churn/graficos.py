"""Gráficos base del proyecto (estilo, paleta y guardado) y figuras del EDA.

Paleta validada para daltonismo (azul y naranja: ΔE CVD 24,7 entre ellas) y tinta neutra
para textos. Cada función devuelve la figura para poder guardarla con `guardar_figura`.

Las figuras del modelado y de la evaluación en test están en `graficos_modelado.py`; las de
negocio y explicabilidad, en `graficos_negocio.py`. Ambas usan el estilo definido aquí.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from churn import config

# ---------------------------------------------------------------------------
# Paleta y estilo
# ---------------------------------------------------------------------------
COLOR_SE_QUEDA = "#2a78d6"   # azul: clase 0
COLOR_SE_VA = "#eb6834"      # naranja: clase 1 (churn)
COLORES_CLASE = {0: COLOR_SE_QUEDA, 1: COLOR_SE_VA}
ETIQUETAS_CLASE = {0: "Se queda", 1: "Se va (churn)"}

TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
COLOR_GRILLA = "#e1e0d9"
COLOR_EJE = "#c3c2b7"
SUPERFICIE = "#fcfcfb"
GRIS_NEUTRO = "#f0efec"
ROJO_DIVERGENTE = "#e34948"

MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_gris_rojo", [COLOR_SE_QUEDA, GRIS_NEUTRO, ROJO_DIVERGENTE])
MAPA_SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul_secuencial", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def texto(valor) -> str:
    """Texto seguro para matplotlib: escapa el signo $ (si no, lo lee como fórmula)."""
    return str(valor).replace("$", r"\$")


def porcentaje(valor: float, decimales: int = 1) -> str:
    """Porcentaje con coma decimal, como se escribe en español."""
    return f"{100 * valor:.{decimales}f} %".replace(".", ",")


def entero(valor: int) -> str:
    """Entero con punto de miles, como se escribe en español."""
    return f"{int(valor):,}".replace(",", ".")


def decimal(valor: float, decimales: int = 3) -> str:
    """Número con coma decimal."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def aplicar_estilo() -> None:
    """Estilo sobrio: fondo claro, grilla tenue sólida, sin bordes superior ni derecho."""
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": COLOR_EJE,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "axes.titlecolor": TINTA_PRIMARIA,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR_GRILLA,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TINTA_TENUE,
        "ytick.color": TINTA_TENUE,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": TINTA_PRIMARIA,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "figure.dpi": 100,
    })


def guardar_figura(fig: plt.Figure, nombre: str, carpeta: Path = config.RUTA_FIGURAS) -> Path:
    """Guarda la figura en PNG (reports/figures) para usarla en el informe."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return ruta


def rejilla(n: int, ncols: int, ancho: float = 4.2, alto: float = 3.0):
    """Crea una rejilla de subgráficos y oculta los que sobran."""
    nfilas = math.ceil(n / ncols)
    fig, ejes = plt.subplots(nfilas, ncols, figsize=(ancho * ncols, alto * nfilas), squeeze=False)
    ejes = ejes.ravel()
    for eje in ejes[n:]:
        eje.set_visible(False)
    return fig, ejes[:n]


def _leyenda_clases(fig: plt.Figure) -> None:
    """Leyenda única arriba de la figura para las dos clases."""
    manijas = [plt.Rectangle((0, 0), 1, 1, color=COLORES_CLASE[c], alpha=0.75) for c in (0, 1)]
    fig.legend(manijas, [ETIQUETAS_CLASE[0], ETIQUETAS_CLASE[1]], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.0))


# ---------------------------------------------------------------------------
# Gráficos del EDA
# ---------------------------------------------------------------------------
def barras_auc_univariada(resumen: pd.DataFrame, titulo: str) -> plt.Figure:
    """Barras horizontales del AUC univariado, coloreadas por el sentido de la relación."""
    datos = resumen.sort_values("auc_univariada")
    colores = [COLOR_SE_VA if s.startswith("más alto, más") else COLOR_SE_QUEDA
               for s in datos["sentido"]]
    fig, eje = plt.subplots(figsize=(8, 0.38 * len(datos) + 1.2))
    eje.barh(datos.index, datos["auc_univariada"] - 0.5, left=0.5, color=colores, height=0.62)
    eje.axvline(0.5, color=TINTA_TENUE, linewidth=1)
    for i, valor in enumerate(datos["auc_univariada"]):
        eje.text(valor + 0.004, i, decimal(valor), va="center", fontsize=8, color=TINTA_SECUNDARIA)
    eje.set_xlim(0.5, max(0.85, datos["auc_univariada"].max() + 0.05))
    eje.set_xlabel("AUC univariada (0,5 = no separa; 1 = separa perfecto)")
    eje.set_title(titulo, loc="left")
    eje.grid(axis="y", visible=False)
    manijas = [plt.Rectangle((0, 0), 1, 1, color=COLOR_SE_VA),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_SE_QUEDA)]
    eje.legend(manijas, ["Valores altos, más churn", "Valores altos, menos churn"],
               loc="lower right")
    fig.tight_layout()
    return fig


def densidades_por_clase(X: pd.DataFrame, y: pd.Series, columnas: list[str],
                         titulo: str, ncols: int = 3, bins: int = 40) -> plt.Figure:
    """Histogramas normalizados por clase (cada clase suma 1) para comparar formas,
    no tamaños: así el 16 % de churners no queda aplastado por el 84 % restante."""
    fig, ejes = rejilla(len(columnas), ncols)
    for eje, col in zip(ejes, columnas):
        bordes = np.histogram_bin_edges(X[col], bins=bins)
        for clase in (0, 1):
            eje.hist(X.loc[y == clase, col], bins=bordes, density=True, alpha=0.55,
                     color=COLORES_CLASE[clase], edgecolor=SUPERFICIE, linewidth=0.4)
        eje.set_title(col, loc="left", fontsize=10)
        eje.set_ylabel("Densidad")
        eje.set_yticklabels([])
    _leyenda_clases(fig)
    fig.suptitle(titulo, y=1.06)
    fig.tight_layout()
    return fig


def dispersion_por_clase(X: pd.DataFrame, y: pd.Series, eje_x: str, eje_y: str,
                         titulo: str) -> plt.Figure:
    """Dispersión de dos variables coloreada por clase (churners dibujados encima)."""
    fig, eje = plt.subplots(figsize=(8, 5.2))
    for clase in (0, 1):
        mascara = y == clase
        eje.scatter(X.loc[mascara, eje_x], X.loc[mascara, eje_y], s=9, alpha=0.45,
                    color=COLORES_CLASE[clase], label=ETIQUETAS_CLASE[clase], linewidths=0)
    eje.set_xlabel(eje_x)
    eje.set_ylabel(eje_y)
    eje.set_title(titulo, loc="left")
    eje.legend(loc="upper left", markerscale=2.5)
    fig.tight_layout()
    return fig


def tasas_por_nivel(tablas: dict[str, pd.DataFrame], tasa_global: float, titulo: str,
                    ncols: int = 2, rotar: bool = False) -> plt.Figure:
    """Tasa de churn por nivel con intervalo de confianza del 95 % y la tasa global como
    referencia. El tamaño de cada grupo va en la etiqueta para no sobreinterpretar
    grupos pequeños."""
    fig, ejes = rejilla(len(tablas), ncols, ancho=5.6, alto=3.4)
    for eje, (col, tabla) in zip(ejes, tablas.items()):
        posiciones = np.arange(len(tabla))
        errores = np.vstack([tabla["tasa_churn"] - tabla["ic95_inf"],
                             tabla["ic95_sup"] - tabla["tasa_churn"]])
        eje.bar(posiciones, tabla["tasa_churn"], color=COLOR_SE_VA, width=0.62, alpha=0.9)
        eje.errorbar(posiciones, tabla["tasa_churn"], yerr=errores, fmt="none",
                     ecolor=TINTA_SECUNDARIA, elinewidth=1, capsize=3)
        eje.axhline(tasa_global, color=TINTA_TENUE, linewidth=1)
        eje.annotate(f"global {porcentaje(tasa_global)}", xy=(1.0, tasa_global),
                     xycoords=("axes fraction", "data"), xytext=(0, 3), textcoords="offset points",
                     ha="right", va="bottom", fontsize=8, color=TINTA_SECUNDARIA,
                     bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 1, "alpha": 0.85})
        etiquetas = [f"{texto(nivel)}\nn={entero(n)}" for nivel, n in zip(tabla.index, tabla["clientes"])]
        eje.set_xticks(posiciones, etiquetas, rotation=30 if rotar else 0,
                       ha="right" if rotar else "center")
        eje.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: porcentaje(v, 0)))
        eje.set_ylim(0, min(1.05, max(0.4, tabla["ic95_sup"].max() * 1.1)))
        eje.set_ylabel("Tasa de churn")
        eje.set_title(col, loc="left", fontsize=10)
        eje.grid(axis="x", visible=False)
    fig.suptitle(titulo, y=1.02)
    fig.tight_layout()
    return fig


def mapa_correlacion(corr: pd.DataFrame, titulo: str, umbral_etiqueta: float = 0.5) -> plt.Figure:
    """Mapa de calor de correlación (triángulo inferior). Solo se rotulan los pares con
    |r| >= umbral para dirigir la lectura a las redundancias que importan."""
    mascara = np.triu(np.ones(corr.shape, dtype=bool))
    valores = corr.mask(mascara)
    fig, eje = plt.subplots(figsize=(9.5, 8))
    imagen = eje.imshow(valores, cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1)
    n = len(corr)
    eje.set_xticks(range(n), corr.columns, rotation=60, ha="right")
    eje.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(i):
            r = corr.iloc[i, j]
            if abs(r) >= umbral_etiqueta:
                eje.text(j, i, decimal(r, 2), ha="center", va="center", fontsize=8,
                         color="white" if abs(r) > 0.8 else TINTA_PRIMARIA)
    eje.grid(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_visible(False)
    barra = fig.colorbar(imagen, ax=eje, shrink=0.75)
    barra.set_label("Correlación de Spearman")
    eje.set_title(titulo, loc="left")
    fig.tight_layout()
    return fig


def barras_conteo(X: pd.DataFrame, columna: str, destacar, titulo: str) -> plt.Figure:
    """Conteo de clientes por valor, resaltando un valor anómalo."""
    conteo = X[columna].value_counts().sort_index()
    colores = [COLOR_SE_VA if v == destacar else COLOR_SE_QUEDA for v in conteo.index]
    fig, eje = plt.subplots(figsize=(9, 3.8))
    eje.bar(conteo.index, conteo.values, color=colores, width=0.8)
    eje.annotate(f"{destacar}: {entero(conteo[destacar])} clientes ({porcentaje(conteo[destacar] / conteo.sum())})",
                 xy=(destacar, conteo[destacar]), xytext=(10, -12), textcoords="offset points",
                 fontsize=9, color=TINTA_PRIMARIA)
    eje.set_xlabel(columna)
    eje.set_ylabel("Clientes")
    eje.set_title(titulo, loc="left")
    eje.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def tabla_cruzada(tabla: pd.DataFrame, titulo: str) -> plt.Figure:
    """Mapa de calor de conteos para una tabla cruzada pequeña, con todas las celdas rotuladas."""
    fig, eje = plt.subplots(figsize=(9, 2.8))
    imagen = eje.imshow(tabla.values, cmap=MAPA_SECUENCIAL, aspect="auto")
    eje.set_xticks(range(tabla.shape[1]), [texto(c) for c in tabla.columns])
    eje.set_yticks(range(tabla.shape[0]), [texto(i) for i in tabla.index])
    maximo = tabla.values.max()
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = tabla.iloc[i, j]
            eje.text(j, i, entero(v), ha="center", va="center", fontsize=9,
                     color="white" if v > 0.55 * maximo else TINTA_PRIMARIA)
    eje.grid(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_visible(False)
    fig.colorbar(imagen, ax=eje, label="Clientes")
    eje.set_title(titulo, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gráficos del modelado
# ---------------------------------------------------------------------------
