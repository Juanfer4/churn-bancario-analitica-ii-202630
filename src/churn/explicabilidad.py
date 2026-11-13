"""Explicabilidad y riesgos del modelo final (Fase 5, 5.6 a 5.8).

Todo se calcula con train (validación cruzada con los mismos pliegues): el test ya se usó
una vez para la evaluación final y no se vuelve a usar para sacar conclusiones nuevas.

La importancia de variables es relativa y no causal: dice cuánto se apoya el modelo en
cada variable para ordenar clientes, no que cambiar esa variable cambie el churn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import cross_val_predict

from churn import config, datos, evaluacion, exploracion, features, negocio


def _variable_original(nombre_salida: str) -> str:
    """Nombre de la variable original de una columna de salida (agrupa las One-Hot)."""
    for col in datos.COLUMNAS_CATEGORICAS:
        if nombre_salida.startswith(col + "_"):
            return col
    return nombre_salida


def importancia_ganancia(pipeline) -> pd.Series:
    """Importancia por pureza: ganancia total (reducción de la pérdida) que aportan los cortes
    de cada variable en los árboles del modelo, en porcentaje del total. Las columnas One-Hot
    se suman en su variable original para poder compararlas."""
    modelo = pipeline.named_steps["modelo"]
    nombres = features.nombres_salida(pipeline)
    if hasattr(modelo, "booster_"):  # LightGBM: ganancia total, no conteo de cortes
        valores = modelo.booster_.feature_importance(importance_type="gain")
    else:
        valores = modelo.feature_importances_
    serie = pd.Series(valores, index=nombres, dtype=float).groupby(_variable_original).sum()
    return (100 * serie / serie.sum()).sort_values(ascending=False).rename("importancia_ganancia_pct")


def importancia_permutacion_cv(pipeline, X: pd.DataFrame, y: pd.Series, cv, n_repeticiones: int = 5,
                               semilla: int = config.SEMILLA, n_jobs: int = -1) -> pd.DataFrame:
    """Importancia por permutación con validación cruzada: en cada pliegue se ajusta el pipeline
    con la parte de entrenamiento y, en la parte de validación, se desordena una variable
    original a la vez y se mide cuánto cae el AUC. Una caída grande significa que el modelo
    depende de esa variable para ordenar clientes que no vio."""
    caidas = []
    for k, (entrenar, validar) in enumerate(cv.split(X, y)):
        modelo = clone(pipeline).fit(X.iloc[entrenar], y.iloc[entrenar])
        resultado = permutation_importance(modelo, X.iloc[validar], y.iloc[validar], scoring="roc_auc",
                                           n_repeats=n_repeticiones, random_state=semilla + k, n_jobs=n_jobs)
        caidas.append(pd.Series(resultado.importances_mean, index=X.columns))
    # Solo se reportan las variables que llegan al modelo: las que el pipeline descarta
    # (avg_open_to_buy y, si el modelo no las usa, las categóricas) tienen caída 0 por construcción
    usadas = {_variable_original(nombre) for nombre in features.nombres_salida(modelo)}
    tabla = pd.concat(caidas, axis=1)
    tabla = tabla.loc[tabla.index.isin(usadas)]
    return pd.DataFrame({"caida_auc_media": tabla.mean(axis=1), "caida_auc_desv": tabla.std(axis=1, ddof=0),
                         "pliegues_con_caida": (tabla > 0).sum(axis=1)}).sort_values("caida_auc_media", ascending=False)


def dependencia_parcial(pipeline, X: pd.DataFrame, variables: list[str], puntos: int = 20,
                        percentiles: tuple[float, float] = (0.01, 0.99)) -> dict[str, pd.DataFrame]:
    """Dependencia parcial: probabilidad promedio de churn predicha cuando una variable toma
    cada valor de su rango (entre los percentiles indicados), dejando las demás como están
    en los datos. Supone que la variable se puede mover sola; con variables muy
    correlacionadas la lectura es aproximada."""
    salida = {}
    for variable in variables:
        resultado = partial_dependence(pipeline, X, [variable], kind="average", grid_resolution=puntos,
                                       percentiles=percentiles, response_method="predict_proba")
        salida[variable] = pd.DataFrame({"valor": _valores_grilla(resultado)[0],
                                         "prob_churn_media": resultado["average"][0]})
    return salida


def dependencia_parcial_2d(pipeline, X: pd.DataFrame, variable_x: str, variable_y: str, puntos: int = 16,
                           percentiles: tuple[float, float] = (0.01, 0.99), muestra: int | None = 2000,
                           semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Dependencia parcial conjunta de dos variables, más cuántos clientes reales caen cerca de
    cada punto de la grilla. Las combinaciones sin clientes son extrapolación del modelo y
    no deben leerse como evidencia.

    El promedio se calcula sobre una muestra aleatoria de clientes (`muestra`), porque cada
    punto de la grilla exige predecir a todos los clientes otra vez; el conteo de clientes
    por celda sí usa todos los datos."""
    base = X if muestra is None or muestra >= len(X) else X.sample(n=muestra, random_state=semilla)
    resultado = partial_dependence(pipeline, base, [variable_x, variable_y], kind="average", grid_resolution=puntos,
                                   percentiles=percentiles, response_method="predict_proba")
    valores_x, valores_y = _valores_grilla(resultado)
    # Cada punto de la grilla representa la celda entre los puntos medios con sus vecinos
    bordes_x, bordes_y = _bordes(valores_x), _bordes(valores_y)
    conteo, _, _ = np.histogram2d(X[variable_x], X[variable_y], bins=[bordes_x, bordes_y])
    malla_x, malla_y = np.meshgrid(valores_x, valores_y, indexing="ij")
    return pd.DataFrame({variable_x: malla_x.ravel(), variable_y: malla_y.ravel(),
                         "prob_churn_media": resultado["average"][0].ravel(), "clientes_en_celda": conteo.ravel().astype(int)})


def _valores_grilla(resultado) -> list[np.ndarray]:
    """Valores de la grilla (el nombre de la clave cambió entre versiones de scikit-learn)."""
    return list(resultado["grid_values"] if "grid_values" in resultado else resultado["values"])


def _bordes(valores: np.ndarray) -> np.ndarray:
    """Bordes de celda alrededor de cada valor de la grilla (puntos medios entre vecinos)."""
    valores = np.asarray(valores, dtype=float)
    medios = (valores[:-1] + valores[1:]) / 2
    return np.concatenate([[valores[0] - (medios[0] - valores[0])], medios, [valores[-1] + (valores[-1] - medios[-1])]])


def neutralizar(X: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Quita la información de una variable sin cambiar la estructura del pipeline:
    las numéricas se eliminan; las categóricas se reemplazan por un valor constante (el
    codificador espera la columna, pero una constante no aporta nada para ordenar)."""
    if variable not in X.columns:
        raise ValueError(f"La variable {variable} no está en los datos")
    if variable in datos.COLUMNAS_CATEGORICAS:
        copia = X.copy()
        copia[variable] = "sin_informacion"
        return copia
    if variable in ("total_revolving_bal", "total_trans_amt", "total_trans_ct", *features.COLUMNAS_A_ELIMINAR):
        raise ValueError(f"{variable} la necesita CreadorVariables; no se puede eliminar")
    return X.drop(columns=variable)


def sensibilidad_variables(pipeline, X: pd.DataFrame, y: pd.Series, cv, variantes: dict[str, list[str]],
                           n_jobs: int = -1) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """AUC de CV del mismo pipeline (mismos hiperparámetros y pliegues) sin ciertas variables.
    Cada variante se entrena una sola vez por pliegue: de las predicciones fuera de pliegue
    se calcula el AUC de cada pliegue, y las diferencias contra la versión completa son
    pareadas (mismo pliegue, mismos clientes). Devuelve la tabla y esas predicciones."""
    pliegues = list(cv.split(X, y))
    filas, oof, base = [], {}, None
    for nombre, variables in {"completo": [], **variantes}.items():
        X_variante = X
        for variable in variables:
            X_variante = neutralizar(X_variante, variable)
        prob = cross_val_predict(clone(pipeline), X_variante, y, cv=pliegues, method="predict_proba", n_jobs=n_jobs)[:, 1]
        auc = evaluacion.auc_por_pliegue(y, prob, pliegues)
        oof[nombre] = prob
        if base is None:
            base = auc
        diferencia = auc - base
        filas.append({"variante": nombre, "variables_quitadas": ", ".join(variables) or "ninguna",
                      "auc_cv": auc.mean(), "auc_cv_desv": auc.std(),
                      "diferencia_vs_completo": diferencia.mean(), "diferencia_desv": diferencia.std(),
                      "pliegues_por_debajo": int((diferencia < 0).sum())})
    return pd.DataFrame(filas).set_index("variante"), oof


def equidad_por_grupo(y_real, probabilidad, grupo: pd.Series, capacidad: float = 0.2) -> pd.DataFrame:
    """Cómo trata el escenario de contacto a cada grupo (por ejemplo, por sexo o por tramo de edad).

    Por grupo: tamaño, tasa real de churn, porcentaje contactado, captura y precisión dentro del
    grupo. Un trato proporcional contacta a cada grupo según su riesgo real: la razón entre el
    porcentaje contactado y la tasa de churn (`razon_contacto_riesgo`) debe ser parecida."""
    y = np.asarray(y_real).astype(int)
    p = np.asarray(probabilidad, dtype=float)
    grupo = np.asarray(grupo)
    n_contactar = negocio.contactados(len(y), capacidad)
    seleccionado = np.zeros(len(y), dtype=bool)
    seleccionado[negocio.orden_por_riesgo(p)[:n_contactar]] = True  # mismo orden que los escenarios
    filas = []
    for valor in pd.unique(grupo):
        m = grupo == valor
        churners = y[m].sum()
        tp = (y[m] & seleccionado[m]).sum()
        filas.append({"grupo": valor, "clientes": int(m.sum()), "churners": int(churners),
                      "tasa_churn_real": y[m].mean(), "pct_contactado": seleccionado[m].mean(),
                      "captura_en_grupo": tp / churners if churners else np.nan,
                      "precision_en_grupo": tp / seleccionado[m].sum() if seleccionado[m].sum() else np.nan})
    tabla = pd.DataFrame(filas).set_index("grupo").sort_index()
    tabla["razon_contacto_riesgo"] = tabla["pct_contactado"] / tabla["tasa_churn_real"]
    return tabla


def ic_brecha_captura(y_real, probabilidad, grupo: pd.Series, capacidad: float = 0.2,
                      n_remuestras: int = 2000, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Brecha de captura entre dos grupos (por ejemplo, mujeres menos hombres) en el escenario de
    contacto, con su IC 95 % por bootstrap.

    La selección (quién queda en el top `capacidad`) se fija con todos los clientes, como en
    `equidad_por_grupo`; luego se remuestrean churners con reposición dentro de cada grupo y se
    recalcula la captura de cada uno. Si el intervalo incluye el 0, la brecha no se distingue
    del azar con este tamaño de muestra.
    """
    y = np.asarray(y_real).astype(int)
    p = np.asarray(probabilidad, dtype=float)
    grupo = np.asarray(grupo)
    valores = sorted(pd.unique(grupo), key=str)
    if len(valores) != 2:
        raise ValueError("ic_brecha_captura compara exactamente dos grupos")
    seleccionado = np.zeros(len(y), dtype=bool)
    seleccionado[negocio.orden_por_riesgo(p)[:negocio.contactados(len(y), capacidad)]] = True
    capturado = {v: seleccionado[(grupo == v) & (y == 1)] for v in valores}  # 1 si el churner quedó contactado
    rng = np.random.default_rng(semilla)
    brechas = np.array([rng.choice(capturado[valores[0]], len(capturado[valores[0]])).mean()
                        - rng.choice(capturado[valores[1]], len(capturado[valores[1]])).mean()
                        for _ in range(n_remuestras)])
    observada = capturado[valores[0]].mean() - capturado[valores[1]].mean()
    inf, sup = np.percentile(brechas, [2.5, 97.5])
    return pd.DataFrame([{
        "comparacion": f"{valores[0]} menos {valores[1]}",
        f"churners_{valores[0]}": len(capturado[valores[0]]), f"churners_{valores[1]}": len(capturado[valores[1]]),
        "brecha_captura": observada, "ic95_inf": inf, "ic95_sup": sup,
        "incluye_cero": bool(inf <= 0 <= sup)}]).set_index("comparacion")


def seleccion_en_grupo(y_real, probabilidades: dict[str, np.ndarray], mascara, capacidad: float = 0.2) -> pd.DataFrame:
    """Para un grupo de clientes (por ejemplo, los de 6 contactos), cuántos quedan dentro del
    escenario de contacto y qué probabilidad les asigna cada variante del modelo."""
    y = np.asarray(y_real).astype(int)
    m = np.asarray(mascara, dtype=bool)
    filas = []
    for nombre, prob in probabilidades.items():
        p = np.asarray(prob, dtype=float)
        seleccionado = np.zeros(len(y), dtype=bool)
        seleccionado[negocio.orden_por_riesgo(p)[:negocio.contactados(len(y), capacidad)]] = True
        filas.append({"variante": nombre, "clientes": int(m.sum()), "churners": int(y[m].sum()),
                      "prob_media": p[m].mean(), "prob_minima": p[m].min(),
                      "dentro_del_escenario": int(seleccionado[m].sum())})
    return pd.DataFrame(filas).set_index("variante")


VARIANTES_SENSIBILIDAD = {"sin contactos": ["contacts_count_12_mon"]}

TRAMOS_MONTO = exploracion.TRAMOS_MONTO  # mismos tramos del EDA (2.4)
TRAMOS_EDAD = ([0, 35, 45, 55, 100], ["hasta 35", "36 a 45", "46 a 55", "más de 55"])  # para revisar el trato por edad (5.7)


def tabla_importancia(permutacion: pd.DataFrame, ganancia: pd.Series) -> pd.DataFrame:
    """Importancia por permutación (en su orden) junto con la de ganancia y los dos rankings."""
    tabla = permutacion.join(ganancia)
    tabla["ranking_permutacion"] = np.arange(1, len(tabla) + 1)
    tabla["ranking_ganancia"] = tabla["importancia_ganancia_pct"].rank(ascending=False).astype(int)
    return tabla


def resumir_tramos(conjunta: pd.DataFrame, minimo_clientes: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Celdas del mapa conjunto con al menos `minimo_clientes` y rango de la probabilidad por tramo de monto."""
    con_datos = conjunta[conjunta["clientes_en_celda"] >= minimo_clientes]
    tramos = pd.cut(con_datos["total_trans_amt"], TRAMOS_MONTO[0], labels=TRAMOS_MONTO[1])
    resumen = con_datos.groupby(tramos, observed=True).agg(
        celdas=("prob_churn_media", "size"), prob_min=("prob_churn_media", "min"),
        prob_max=("prob_churn_media", "max"), clientes=("clientes_en_celda", "sum"))
    return con_datos, resumen


def agregar_escenarios(sensibilidad: pd.DataFrame, y: pd.Series, oof_variantes: dict, capacidades=(0.1, 0.2)) -> pd.DataFrame:
    """Agrega a la tabla de sensibilidad churners capturados, captura y precisión por variante."""
    for nombre, prob in oof_variantes.items():
        escenarios = negocio.tabla_escenarios(y, prob)
        for k in capacidades:
            sensibilidad.loc[nombre, f"churners_{round(100 * k)}"] = escenarios.loc[k, "churners_capturados"]
            sensibilidad.loc[nombre, f"captura_{round(100 * k)}"] = escenarios.loc[k, "captura"]
            sensibilidad.loc[nombre, f"precision_{round(100 * k)}"] = escenarios.loc[k, "precision"]
    return sensibilidad
