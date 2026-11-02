"""Guardado y presentación de resultados del notebook: una sola forma de guardar tablas y figuras.

Antes cada celda repetía `tabla.round(6).to_csv(config.RUTA_TABLAS / "nombre.csv")` y
`graficos.guardar_figura(...)` seguido de `plt.show()`. Estas dos funciones lo centralizan:
el nombre del archivo, el redondeo y la carpeta se definen en un solo lugar.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from churn import config, graficos


def guardar_tabla(tabla: pd.DataFrame | pd.Series, nombre: str, decimales: int | None = 6,
                  indice: bool = True, carpeta: Path = config.RUTA_TABLAS) -> Path:
    """Guarda `tabla` como `nombre.csv` en `reports/tables` y devuelve la ruta.

    decimales: redondeo antes de guardar (6 por defecto, suficiente para citar cifras en el
        informe); None guarda los valores sin redondear.
    indice: si es False no se escribe el índice (para tablas cuyo índice es solo un contador).
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}.csv"
    destino = tabla if decimales is None else tabla.round(decimales)
    destino.to_csv(ruta, index=indice)
    return ruta


def mostrar_figura(fig: plt.Figure, nombre: str) -> Path:
    """Guarda la figura en `reports/figures` y la muestra en el notebook."""
    ruta = graficos.guardar_figura(fig, nombre)
    plt.show()
    return ruta
