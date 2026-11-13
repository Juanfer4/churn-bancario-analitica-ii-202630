"""Las tablas de la Fase 4 corresponden a los modelos guardados en la caché.

Recalcula las tablas de modelado desde la caché de modelos y las compara con las que el
notebook guardó en reports/tables. Si las tablas o la caché no existen, la prueba se omite:
hay que ejecutar el notebook primero. Las cifras del texto se revisan en test_cifras_texto.py.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from churn import comparacion, config
from conftest import exigir_tablas

TABLAS = config.RUTA_TABLAS
NECESARIAS = ["modelado_candidatas.csv", "modelado_tabla_cv.csv", "modelado_estrategias.csv",
              "modelado_rondas_1_2.csv", "modelado_bordes_ronda_1.csv",
              "modelado_bordes_final.csv", "modelado_pareada.csv", "modelado_pycaret.csv"]


def _leer(nombre, **kwargs):
    return pd.read_csv(TABLAS / nombre, **kwargs)


@pytest.fixture(scope="module")
def tablas():
    exigir_tablas(NECESARIAS)
    return {
        "cand": _leer("modelado_candidatas.csv"),
        "cv": _leer("modelado_tabla_cv.csv", index_col=0),
        "estr": _leer("modelado_estrategias.csv", index_col=0),
        "r12": _leer("modelado_rondas_1_2.csv", index_col=0),
        "b1": _leer("modelado_bordes_ronda_1.csv"),
        "bf": _leer("modelado_bordes_final.csv"),
        "par": _leer("modelado_pareada.csv", index_col=0),
        "pyc": _leer("modelado_pycaret.csv", index_col=0),
    }


def test_tablas_corresponden_a_los_modelos_en_cache(tablas):
    """La tabla final de CV debe salir exactamente de los modelos guardados en la caché."""
    rutas = {}
    for prefijo in ("ronda_2", "ronda_1"):
        for nombre in tablas["cv"].index:
            ruta = comparacion._archivo_cache(config.RUTA_EXPERIMENTOS, prefijo, nombre)
            if nombre not in rutas and ruta.exists():
                rutas[nombre] = ruta
    if len(rutas) != len(tablas["cv"]):
        pytest.skip("No está completa la caché de modelos")
    for nombre, ruta in rutas.items():
        resumen = joblib.load(ruta)["modelo"].resumen()
        assert np.isclose(resumen["auc_cv"], tablas["cv"].loc[nombre, "auc_cv"], atol=5e-6), nombre
        assert np.isclose(resumen["auc_train"], tablas["cv"].loc[nombre, "auc_train"], atol=5e-6), nombre
