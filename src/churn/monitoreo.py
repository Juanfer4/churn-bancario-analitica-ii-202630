"""Monitoreo y control del modelo en uso: ¿los clientes nuevos se parecen a los de entrenamiento
y el modelo sigue capturando churners como se esperaba?

Tres indicadores, con semáforo (verde, amarillo, rojo):

1. PSI por variable (índice de estabilidad poblacional): cuánto se movió la distribución de cada
   variable frente a train. Detecta cambios en los datos (data drift) sin conocer el churn real,
   así que se revisa con cada lote de clientes.
2. PSI del puntaje: lo mismo, pero sobre la probabilidad de churn que da el modelo. Resume en un
   número los cambios conjuntos entre variables que el PSI por variable no ve (por ejemplo, que
   monto y número de transacciones dejen de moverse juntos). Tampoco necesita el churn real.
3. Captura de churners del 20 % de mayor riesgo frente a la línea base de validación cruzada.
   Detecta que la relación entre las variables y el churn cambió (concept drift). Necesita el
   churn real, que se conoce al cierre del año siguiente: se revisa una vez al año. Debe medirse
   en un grupo de control (clientes de alto riesgo que no reciben el beneficio); en los
   contactados, una campaña que funciona reduce el churn y haría parecer que el modelo empeoró.

Los umbrales están en `config.UMBRALES_MONITOREO`. Los del PSI (0,10 y 0,25) son los que se
usan por convención en la industria; los de captura son supuestos del equipo, ajustables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config, negocio

EPSILON = 1e-4  # evita dividir entre cero o tomar logaritmo de cero en rangos vacíos


def psi(referencia: pd.Series, nuevo: pd.Series, cortes: int = 10) -> float:
    """Índice de estabilidad poblacional entre la distribución de referencia (train) y la nueva.

    Variables numéricas con varios valores: se parten en `cortes` rangos con los percentiles de
    la referencia. Variables categóricas o con pocos valores: se comparan las proporciones por
    categoría. Valores cercanos a 0 indican que nada cambió; más de 0,25, que cambió mucho.
    """
    referencia, nuevo = pd.Series(referencia).dropna(), pd.Series(nuevo).dropna()
    if pd.api.types.is_numeric_dtype(referencia) and referencia.nunique() > cortes:
        bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, cortes + 1)))
        bordes[0], bordes[-1] = -np.inf, np.inf
        p_ref = pd.cut(referencia, bordes).value_counts(normalize=True, sort=False)
        p_nue = pd.cut(nuevo, bordes).value_counts(normalize=True, sort=False)
    else:
        categorias = sorted(set(referencia) | set(nuevo), key=str)
        p_ref = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0)
        p_nue = nuevo.value_counts(normalize=True).reindex(categorias, fill_value=0)
    a, b = np.clip(p_ref.to_numpy(float), EPSILON, None), np.clip(p_nue.to_numpy(float), EPSILON, None)
    return float(np.sum((b - a) * np.log(b / a)))


def estado_indicador(valor: float, amarillo: float, rojo: float) -> str:
    """Semáforo de un indicador en el que un valor más alto es peor."""
    if valor >= rojo:
        return "rojo"
    return "amarillo" if valor >= amarillo else "verde"


def psi_por_variable(X_referencia: pd.DataFrame, X_nuevo: pd.DataFrame,
                     umbrales: dict | None = None) -> pd.DataFrame:
    """PSI y semáforo de cada variable que comparten las dos tablas, de mayor a menor PSI."""
    amarillo, rojo = (umbrales or config.UMBRALES_MONITOREO)["psi"]
    columnas = [c for c in X_referencia.columns if c in X_nuevo.columns]
    tabla = pd.DataFrame({"psi": {c: psi(X_referencia[c], X_nuevo[c]) for c in columnas}})
    tabla["estado"] = [estado_indicador(v, amarillo, rojo) for v in tabla["psi"]]
    return tabla.sort_values("psi", ascending=False)


def captura_vs_base(y_real, probabilidad, captura_base: float, capacidad: float = 0.2,
                    umbrales: dict | None = None) -> dict:
    """Captura observada del top `capacidad` por riesgo, caída frente a la línea base y semáforo.

    captura_base: captura de referencia (por ejemplo, la de validación cruzada en train).
    La caída se mide en puntos de captura (0,03 = 3 puntos porcentuales).
    """
    amarillo, rojo = (umbrales or config.UMBRALES_MONITOREO)["caida_captura"]
    escenario = negocio.tabla_escenarios(y_real, probabilidad, capacidades=(capacidad,)).iloc[0]
    caida = float(captura_base - escenario["captura"])
    return {"captura": float(escenario["captura"]), "captura_base": float(captura_base),
            "caida": caida, "estado": estado_indicador(caida, amarillo, rojo)}


def psi_puntaje(puntaje_referencia, puntaje_nuevo, umbrales: dict | None = None) -> dict:
    """PSI de la probabilidad de churn entre la referencia y un lote nuevo, con su semáforo.

    La referencia debe ser del mismo tipo que el lote: puntajes del mismo modelo sobre los
    clientes con que se entrenó (o sobre el primer lote en operación). Comparar puntajes de
    entrenamiento contra puntajes fuera de pliegue daría una falsa alarma.
    """
    amarillo, rojo = (umbrales or config.UMBRALES_MONITOREO)["psi"]
    valor = psi(pd.Series(np.asarray(puntaje_referencia, dtype=float)), pd.Series(np.asarray(puntaje_nuevo, dtype=float)))
    return {"psi": valor, "estado": estado_indicador(valor, amarillo, rojo)}


def tabla_semaforos(pipeline, columnas: list[str], X_referencia: pd.DataFrame, lotes: dict[str, pd.DataFrame],
                    captura_base: float, y_lotes: dict[str, pd.Series] | None = None, capacidad: float = 0.2,
                    umbrales: dict | None = None) -> pd.DataFrame:
    """Los tres indicadores de monitoreo, con su semáforo, para uno o varios lotes de clientes.

    pipeline y columnas: el modelo en uso y las columnas que recibe.
    X_referencia: clientes con que se entrenó el modelo (referencia del PSI por variable y, puntuados
        por el mismo modelo, del PSI del puntaje).
    lotes: {nombre: clientes del lote}.
    captura_base: captura de referencia del top `capacidad` (por ejemplo, la de validación cruzada).
    y_lotes: churn real de cada lote, si ya se conoce. En operación se conoce un año después y se
        mide en el grupo de control; mientras tanto la captura queda pendiente y solo se leen los PSI.
    """
    clase = list(pipeline.classes_).index(1)  # columna de la clase churn en predict_proba
    puntaje_referencia = pipeline.predict_proba(X_referencia[columnas])[:, clase]
    etiqueta = f"captura {round(100 * capacidad)} %"
    filas = {}
    for nombre, X_lote in lotes.items():
        psi_variable = psi_por_variable(X_referencia, X_lote, umbrales).iloc[0]
        puntaje = pipeline.predict_proba(X_lote[columnas])[:, clase]
        psi_p = psi_puntaje(puntaje_referencia, puntaje, umbrales)
        fila = {"variable con más PSI": psi_variable.name, "PSI variable": psi_variable["psi"],
                "estado": psi_variable["estado"], "PSI puntaje": psi_p["psi"], "estado puntaje": psi_p["estado"]}
        y_lote = None if y_lotes is None else y_lotes.get(nombre)
        if y_lote is None:
            fila.update({etiqueta: np.nan, "caída": np.nan, "estado captura": "pendiente (sin churn real)"})
        else:
            captura = captura_vs_base(y_lote, puntaje, captura_base, capacidad, umbrales)
            fila.update({etiqueta: captura["captura"], "caída": captura["caida"], "estado captura": captura["estado"]})
        filas[nombre] = fila
    return pd.DataFrame(filas).T
