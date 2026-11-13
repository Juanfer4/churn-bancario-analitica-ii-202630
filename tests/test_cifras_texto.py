"""Cada cifra del texto del notebook debe salir de un resultado del propio notebook.

La prueba lee el notebook, toma todos los números escritos en las celdas de texto y exige
que cada uno aparezca en una salida de código del notebook o en una tabla de reports/tables,
con el mismo redondeo. Si al editar el texto se cambia un número por error, o si el código
cambia y el texto queda desactualizado, la prueba falla y dice qué cifra no tiene respaldo.

Límite conocido: comprueba que el número existe entre los resultados, no que esté citado en
la frase correcta. Las reglas de decisión se revisan aparte, en `test_decisiones_del_notebook`.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from churn import config, datos, produccion

RUTA_NOTEBOOK = config.RAIZ / "notebooks" / "churn_banco.ipynb"
NUMERO_TEXTO = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+|\d+")  # 1.275 | 0,9930 | 42
NUMERO_SALIDA = re.compile(r"\d+\.\d+(?:e-?\d+)?|\d+")

# Números del texto que no son resultados: referencias a secciones, años, parámetros fijados
# por el equipo (config.py y grillas) y cuentas simples sobre cifras que sí están respaldadas.
CONSTANTES = {
    2021, 2022,                      # años del caso
    3000, 3001, 6000, 6001, 11000,   # tramos de monto del EDA (exploracion.TRAMOS_MONTO)
    2000,                            # remuestras del bootstrap y clientes del mapa de 5.6
    0.5, 0.2, 0.1, 0.25, 1.5,        # umbral por defecto, valores de s y umbrales del PSI
    199,                             # 1.275 - 1.076 churners (LightGBM frente a la logística, 5.4)
    0.006, 0.994, 0.001,             # aproximaciones redondeadas de 4.5 y 4.7
    98.02,                           # captura máxima por tramo de edad (tabla con 4 decimales)
}


def _a_numero(texto: str) -> tuple[float, int]:
    """Convierte '1.275' o '97,93' al número y a su cantidad de decimales."""
    entero, _, decimales = texto.partition(",")
    return float(entero.replace(".", "") + ("." + decimales if decimales else "")), len(decimales)


@pytest.fixture(scope="module")
def notebook():
    if not RUTA_NOTEBOOK.exists():
        pytest.skip("No está el notebook")
    return json.loads(RUTA_NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _resultados(celdas) -> np.ndarray:
    """Todos los números que produjo el código: salidas de las celdas y tablas guardadas."""
    valores = []
    for celda in celdas:
        for salida in celda.get("outputs", []):
            texto = "".join(salida.get("text", "")) + "".join(salida.get("data", {}).get("text/plain", ""))
            valores += [float(n) for n in NUMERO_SALIDA.findall(texto.replace(",", ""))]
    for ruta in sorted(config.RUTA_TABLAS.glob("*.csv")):
        tabla = pd.read_csv(ruta)
        valores += tabla.select_dtypes("number").to_numpy(dtype=float).ravel().tolist()
    valores = np.array(valores)
    return np.abs(valores[np.isfinite(valores)])


def test_cada_cifra_del_texto_sale_de_un_resultado(notebook):
    resultados = _resultados(notebook)
    if len(resultados) < 1000:
        pytest.skip("El notebook no tiene salidas: ejecútelo primero")
    sin_respaldo = []
    for i, celda in enumerate(notebook):
        if celda["cell_type"] != "markdown":
            continue
        texto = re.sub(r"`[^`]*`|\(\d\.\d+[^)]*\)|#+ \d\.\d+|\b\d\.\d+ (?:a|y) \d\.\d+\b", " ", "".join(celda["source"]))
        for cita in NUMERO_TEXTO.findall(texto):
            numero, decimales = _a_numero(cita)
            if numero in CONSTANTES or (decimales == 0 and numero < 100):
                continue  # enteros pequeños: conteos de pliegues, modelos o variables
            tolerancia = 0.5 * 10 ** -decimales + 1e-9
            como_valor = np.abs(resultados - numero) <= tolerancia
            como_porcentaje = np.abs(100 * resultados - numero) <= tolerancia
            if not (como_valor.any() or como_porcentaje.any()):
                sin_respaldo.append(f"celda {i}: {cita}")
    assert not sin_respaldo, sin_respaldo


def test_decisiones_del_notebook():
    """Las decisiones escritas en el texto se cumplen en las tablas, sin fijar cifras."""
    tablas = config.RUTA_TABLAS
    necesarias = ["modelado_categoricas", "modelado_tabla_cv", "test_bootstrap_auc", "negocio_incremental_oof",
                  "modelado_combinacion_finalistas"]
    if not all((tablas / f"{n}.csv").exists() for n in necesarias):
        pytest.skip("Ejecute el notebook para generar las tablas")
    categoricas = pd.read_csv(tablas / "modelado_categoricas.csv", index_col=0)
    cv = pd.read_csv(tablas / "modelado_tabla_cv.csv", index_col=0)
    bootstrap = pd.read_csv(tablas / "test_bootstrap_auc.csv", index_col=0)
    incremental = pd.read_csv(tablas / "negocio_incremental_oof.csv", index_col=0)
    combinacion = pd.read_csv(tablas / "modelado_combinacion_finalistas.csv", index_col=0)

    # 4.3: el modelo oficial no usa las categóricas porque no cumplen la regla, y produccion.py lo refleja
    assert bool(categoricas.loc["LightGBM", "aporte_claro"]) == produccion.USA_CATEGORICAS_OFICIAL
    assert list(categoricas.index[categoricas["aporte_claro"]]) == ["Regresión logística"]
    # 4.7 y 5.2: LightGBM es el mejor en validación cruzada y ningún finalista lo supera en test
    assert cv.index[0] == produccion.NOMBRE_MODELO_OFICIAL
    finalistas = bootstrap.drop(index=["LightGBM", "Regresión logística"])
    assert (finalistas["dif_ic95_inf"] < 0).all() and (finalistas["dif_ic95_sup"] > 0).all()
    assert bootstrap.loc["Regresión logística", "dif_ic95_sup"] < 0
    # 5.4: pasar de 20 % a 30 % de contacto casi no encuentra churners
    assert incremental.loc[0.3, "precision_marginal"] < 0.05 < incremental.loc[0.2, "precision_marginal"]
    # 4.7: combinar a los finalistas no mejora a LightGBM por más de una desviación entre pliegues
    assert (combinacion["diferencia_vs_referencia"] <= combinacion["diferencia_desv"]).all()
    # 6.1: el modelo guardado no recibe las categóricas
    assert not set(datos.COLUMNAS_CATEGORICAS) & set(produccion.columnas_entrada())
