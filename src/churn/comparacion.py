"""Comparación de modelos: variables candidatas, `Comparador`, rondas de búsqueda y caché.

La definición de cada modelo (clase `Modelo` y catálogo) vive en `modelos.py`. Este módulo
reúne lo que se hace con varios modelos a la vez: decidir variables candidatas, compararlos
con los mismos pliegues, ampliar las grillas por rondas, guardar cada modelo ajustado en
una caché con huella (si nada cambió no se repite el cómputo) y la referencia de PyCaret.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from churn import config, datos, evaluacion, features, negocio
from churn.modelos import ESTRATEGIAS, Modelo, validacion_cruzada

# ---------------------------------------------------------------------------
# Experimento de variables candidatas
# ---------------------------------------------------------------------------
COMBINACIONES_CANDIDATAS = {
    "base": (False, False),
    "+ ticket": (True, False),
    "+ saldo": (False, True),
    "+ ticket + saldo": (True, True),
}


def experimento_candidatas(modelos: list[Modelo], X: pd.DataFrame, y: pd.Series,
                           cv=None, n_jobs: int = -1) -> pd.DataFrame:
    """Evalúa cada modelo con sus valores por defecto (sin balanceo) en las 4 combinaciones
    de variables candidatas, con los mismos pliegues. Las diferencias contra "base" son
    pareadas: se comparan pliegue a pliegue, lo que elimina la variación entre pliegues."""
    cv = cv or validacion_cruzada()
    filas = []
    for modelo in modelos:
        auc_base = None
        for etiqueta, (ticket, saldo) in COMBINACIONES_CANDIDATAS.items():
            pipe = Pipeline(features.pasos_preparacion(modelo.escalar, ticket, saldo,
                                                       categoricas=None if modelo.usar_categoricas else [])
                            + [("balanceo", "passthrough"), ("modelo", clone(modelo.estimador))])
            auc = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)["test_score"]
            if etiqueta == "base":
                auc_base = auc
            diferencia = auc - auc_base
            filas.append({"modelo": modelo.nombre, "combinacion": etiqueta,
                          "auc_cv": auc.mean(), "auc_cv_desv": auc.std(),
                          "mejora_media": diferencia.mean(), "mejora_desv": diferencia.std(),
                          "pliegues_que_mejoran": int((diferencia > 0).sum())})
    return pd.DataFrame(filas)


def experimento_categoricas(modelos: list[Modelo], X: pd.DataFrame, y: pd.Series,
                            cv=None, n_jobs: int = -1) -> pd.DataFrame:
    """Evalúa cada modelo con sus valores por defecto, con y sin las cinco categóricas, en los
    mismos pliegues. `aporte_medio` es el AUC con categóricas menos el AUC sin ellas, pliegue
    a pliegue: si es positivo, las categóricas ayudan."""
    cv = cv or validacion_cruzada()
    filas = []
    for modelo in modelos:
        auc = {}
        for etiqueta, categoricas in (("con", list(datos.COLUMNAS_CATEGORICAS)), ("sin", [])):
            pipe = Pipeline(features.pasos_preparacion(modelo.escalar, categoricas=categoricas)
                            + [("balanceo", "passthrough"), ("modelo", clone(modelo.estimador))])
            auc[etiqueta] = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)["test_score"]
        aporte = auc["con"] - auc["sin"]
        filas.append({"modelo": modelo.nombre, "auc_con_categoricas": auc["con"].mean(),
                      "auc_sin_categoricas": auc["sin"].mean(), "aporte_medio": aporte.mean(),
                      "aporte_desv": aporte.std(), "pliegues_en_que_aportan": int((aporte > 0).sum())})
    tabla = pd.DataFrame(filas).set_index("modelo")
    tabla["aporte_claro"] = ((tabla["aporte_medio"] > 0) & (tabla["aporte_medio"] > tabla["aporte_desv"])
                             & (tabla["pliegues_en_que_aportan"] >= config.N_PLIEGUES - 1))
    return tabla


def aplicar_decision_categoricas(modelos_: list[Modelo], tabla: pd.DataFrame) -> pd.Series:
    """Misma regla que las variables candidatas: un modelo conserva las categóricas solo si su
    aporte es positivo, mayor que su desviación y aparece en al menos 4 de 5 pliegues
    (`aporte_claro`). Si no, deja de recibirlas (parsimonia). Modifica los modelos y devuelve
    la decisión."""
    for m in modelos_:
        m.usar_categoricas = bool(tabla.loc[m.nombre, "aporte_claro"])
    return tabla["aporte_claro"].rename("usa_categoricas")


def decidir_candidatas(tabla: pd.DataFrame) -> dict[str, tuple[bool, bool]]:
    """Regla de decisión por modelo: se adopta la combinación con mayor mejora media solo si
    esa mejora es positiva, supera la desviación de las diferencias por pliegue (es
    consistente) y mejora en al menos 4 de los 5 pliegues. Si ninguna cumple, se queda la
    versión base (parsimonia: no agregar variables que no aportan de forma clara)."""
    decisiones = {}
    for nombre, grupo in tabla.groupby("modelo", sort=False):
        candidatas = grupo[grupo["combinacion"] != "base"]
        validas = candidatas[(candidatas["mejora_media"] > 0)
                             & (candidatas["mejora_media"] > candidatas["mejora_desv"])
                             & (candidatas["pliegues_que_mejoran"] >= config.N_PLIEGUES - 1)]
        if validas.empty:
            decisiones[nombre] = COMBINACIONES_CANDIDATAS["base"]
        else:
            mejor = validas.sort_values("mejora_media", ascending=False).iloc[0]["combinacion"]
            decisiones[nombre] = COMBINACIONES_CANDIDATAS[mejor]
    return decisiones


# ---------------------------------------------------------------------------
# Comparador
# ---------------------------------------------------------------------------
class Comparador:
    """Entrena todos los modelos con los mismos pliegues y los compara."""

    def __init__(self, modelos: list[Modelo], cv=None):
        nombres = [m.nombre for m in modelos]
        if len(nombres) != len(set(nombres)):
            raise ValueError("Los nombres de los modelos deben ser únicos")
        self.modelos = {m.nombre: m for m in modelos}
        self.cv = cv or validacion_cruzada()

    def entrenar_todos(self, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1) -> "Comparador":
        total = len(self.modelos)
        for k, modelo in enumerate(self.modelos.values(), start=1):
            print(f"   {k}/{total} {modelo.nombre}: {modelo.n_candidatos()} candidatos x {config.N_PLIEGUES} pliegues...", end=" ")
            modelo.ajustar(X, y, cv=self.cv, n_jobs=n_jobs)
            r = modelo.resumen()
            print(f"AUC CV {r['auc_cv']:.4f} ({modelo.tiempo_busqueda_:.0f} s)")
        return self

    def tabla_cv(self) -> pd.DataFrame:
        """Mejor candidato de cada modelo, ordenado por AUC de validación cruzada."""
        tabla = pd.DataFrame([m.resumen() for m in self.modelos.values()]).set_index("modelo")
        return tabla.sort_values("auc_cv", ascending=False)

    def comparacion_pareada(self) -> pd.DataFrame:
        """Diferencia de AUC de cada modelo contra el mejor, pliegue a pliegue."""
        tabla = self.tabla_cv()
        mejor = self.modelos[tabla.index[0]].auc_por_pliegue()
        filas = []
        for nombre in tabla.index:
            diferencia = self.modelos[nombre].auc_por_pliegue() - mejor
            filas.append({"modelo": nombre, "diferencia_media": diferencia.mean(),
                          "diferencia_desv": diferencia.std(),
                          "pliegues_por_debajo_del_mejor": int((diferencia < 0).sum())})
        return pd.DataFrame(filas).set_index("modelo")

    def finalistas(self) -> list[str]:
        """Modelos cuyo AUC de CV queda a menos de una desviación estándar del mejor
        (empate estadístico práctico). Entre ellos se preferirá el más simple."""
        tabla = self.tabla_cv()
        mejor = tabla.iloc[0]
        return tabla.index[tabla["auc_cv"] >= mejor["auc_cv"] - mejor["auc_cv_desv"]].tolist()

    def hiperparametros_en_borde(self) -> pd.DataFrame:
        hallazgos = [h for m in self.modelos.values() for h in m.hiperparametros_en_borde()]
        return pd.DataFrame(hallazgos, columns=["modelo", "hiperparametro", "elegido", "probados", "extremo"])


# ---------------------------------------------------------------------------
# PyCaret como referencia AutoML (visto en la clase 10)
# ---------------------------------------------------------------------------
def referencia_pycaret(X_train: pd.DataFrame, y_train: pd.Series, n_seleccion: int = 3,
                       semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Ranking de PyCaret como punto de contraste, solo con train.

    PyCaret aparta internamente el 20 % de nuestro train como validación propia y hace su
    validación cruzada estratificada de 5 pliegues sobre el 80 % restante; nuestro test no
    se le entrega. Recibe los datos sin `avg_open_to_buy` (misma decisión del EDA) y aplica
    su propia preparación automática (codificación, Yeo-Johnson y normalización).
    Devuelve la tabla de `compare_models` ordenada por AUC.
    """
    from pycaret.classification import compare_models, pull, setup  # import pesado: solo si se usa

    datos_pycaret = X_train.drop(columns=features.COLUMNAS_A_ELIMINAR).copy()
    datos_pycaret["churn"] = y_train.values
    setup(data=datos_pycaret, target="churn", train_size=0.8, fold=config.N_PLIEGUES,
          fold_strategy="stratifiedkfold", session_id=semilla, normalize=True,
          transformation=True, verbose=False, html=False, n_jobs=-1)
    compare_models(sort="AUC", n_select=n_seleccion, verbose=False)
    return pull()


# ---------------------------------------------------------------------------
# Ronda 2: ampliación de grillas cuyo óptimo quedó en el borde
# ---------------------------------------------------------------------------
# Cada grilla contiene el óptimo de la ronda 1 y lo extiende en la dirección del borde.
# Solo con la estrategia "ninguna": en la ronda 1 las estrategias de desbalance no
# cambiaron el AUC más allá de una desviación estándar.
GRILLAS_RONDA_2 = {
    "SVM": {"modelo__kernel": ["rbf"], "modelo__C": [3, 10, 30], "modelo__gamma": [0.03, 0.1, 0.3]},
    "Red neuronal (MLP)": {"modelo__activation": ["tanh"],
                           "modelo__hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
                           "modelo__alpha": [1e-2, 1e-1, 0.3, 1.0]},
    "Árbol de decisión": {"modelo__max_depth": [6, 8, 10, 12], "modelo__min_samples_leaf": [20, 40, 80]},
    "Bagging de árboles": {"modelo__max_samples": [1.0], "modelo__max_features": [0.3, 0.5, 0.7]},
    "Random Forest": {"modelo__max_depth": [12, 20, None], "modelo__min_samples_leaf": [1, 2],
                      "modelo__max_features": [0.3, 0.5, 0.7]},
    "AdaBoost": {"modelo__estimator__max_depth": [2, 3], "modelo__n_estimators": [600, 1000],
                 "modelo__learning_rate": [0.5, 1.0, 1.5]},
    "Gradient Boosting": {"modelo__n_estimators": [400, 800], "modelo__learning_rate": [0.1, 0.2],
                          "modelo__max_depth": [3, 4]},
    "XGBoost": {"modelo__n_estimators": [600, 1000, 1500], "modelo__learning_rate": [0.01, 0.03, 0.05],
                "modelo__max_depth": [5, 7]},
    "LightGBM": {"modelo__n_estimators": [600, 1000, 1500], "modelo__learning_rate": [0.01, 0.03, 0.05],
                 "modelo__num_leaves": [31, 63]},
}


# Ronda 3: solo para modelos que en la ronda 2 mejoraron más de una desviación estándar
# y siguen con el óptimo en el borde (ver `modelos_a_ampliar`). Ninguno cumplió la regla.
GRILLAS_RONDA_3: dict = {}


def modelos_a_ampliar(anterior: "Comparador", nueva: "Comparador") -> list[str]:
    """Regla para seguir ampliando: el modelo tiene algún óptimo en el borde Y la ronda nueva
    mejoró su AUC de CV en más de una desviación estándar. Si la mejora es menor, ampliar más
    no cambia las conclusiones y se declara el borde en lugar de seguir buscando."""
    en_borde = set(nueva.hiperparametros_en_borde()["modelo"])
    elegidos = []
    for nombre, m in nueva.modelos.items():
        r_nueva, r_anterior = m.resumen(), anterior.modelos[nombre].resumen()
        if nombre in en_borde and r_nueva["auc_cv"] - r_anterior["auc_cv"] > r_nueva["auc_cv_desv"]:
            elegidos.append(nombre)
    return elegidos


def ampliar_grillas(comparador: "Comparador", grillas: dict) -> list[Modelo]:
    """Copias sin ajustar de los modelos con la grilla ampliada, la misma decisión de
    variables candidatas y solo la estrategia de desbalance "ninguna".
    Se usa para la ronda 2 (desde la ronda 1) y la ronda 3 (desde la ronda 2)."""
    nuevos = []
    for nombre, grilla in grillas.items():
        original = comparador.modelos[nombre]
        nuevos.append(replace(original, hiperparametros=grilla, pesos=None, smote=False,
                              busqueda_=None, tiempo_busqueda_=None))
    return nuevos


# ---------------------------------------------------------------------------
# Caché del experimento por modelo (evita repetir media hora de cómputo si nada cambió)
# ---------------------------------------------------------------------------
def versiones_librerias() -> tuple:
    """Versiones de las librerías que entrenan los modelos. Si alguna cambia, sus resultados
    pueden cambiar, así que la caché deja de servir."""
    return tuple((nombre, version(nombre)) for nombre in
                 ("scikit-learn", "imbalanced-learn", "lightgbm", "xgboost"))


def huella_modelo(X: pd.DataFrame, y: pd.Series, modelo: Modelo, cv) -> str:
    """Huella digital de todo lo que determina el resultado de un modelo: datos, grilla,
    estimador, variables candidatas, categóricas que usa el modelo, pliegues y versiones de scikit-learn, imbalanced-learn,
    LightGBM y XGBoost. Si algo cambia, la caché de ese modelo no se usa y se vuelve a entrenar."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    h.update(repr((modelo.nombre, modelo.familia,
                   sorted(modelo.estimador.get_params(deep=True).items(), key=str),
                   modelo.hiperparametros, modelo.escalar, modelo.pesos, modelo.smote,
                   modelo.ticket_promedio, modelo.indicadores_saldo,
                   modelo.usar_categoricas)).encode())
    h.update(repr((cv.n_splits, cv.shuffle, cv.random_state, versiones_librerias())).encode())
    return h.hexdigest()


def _archivo_cache(carpeta: Path, prefijo: str, nombre: str) -> Path:
    sin_tildes = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    simple = re.sub(r"[^a-z0-9]+", "_", sin_tildes.lower()).strip("_")
    return Path(carpeta) / f"{prefijo}_{simple}.joblib"


def entrenar_o_cargar(modelos_: list[Modelo], X: pd.DataFrame, y: pd.Series, carpeta: Path,
                      prefijo: str, reentrenar: bool = False, n_jobs: int = -1) -> "Comparador":
    """Entrena cada modelo o lo carga de disco si su caché tiene la misma huella."""
    comparador = Comparador(modelos_)
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    total = len(modelos_)
    for k, modelo in enumerate(modelos_, start=1):
        huella = huella_modelo(X, y, modelo, comparador.cv)
        ruta = _archivo_cache(carpeta, prefijo, modelo.nombre)
        if ruta.exists() and not reentrenar:
            try:
                guardado = joblib.load(ruta)
            except Exception as error:  # caché corrupta o de otra versión de librerías
                print(f"   {k}/{total} {modelo.nombre}: no se pudo leer la caché ({type(error).__name__}); se reentrena")
                guardado = {}
            if guardado.get("huella") == huella:
                comparador.modelos[modelo.nombre] = guardado["modelo"]
                print(f"   {k}/{total} {modelo.nombre}: cargado de la caché (misma huella)")
                continue
        print(f"   {k}/{total} {modelo.nombre}: {modelo.n_candidatos()} candidatos x {config.N_PLIEGUES} pliegues...", end=" ")
        modelo.ajustar(X, y, cv=comparador.cv, n_jobs=n_jobs)
        print(f"AUC CV {modelo.resumen()['auc_cv']:.4f} ({modelo.tiempo_busqueda_:.0f} s)")
        joblib.dump({"huella": huella, "modelo": modelo}, ruta)
    return comparador


# ---------------------------------------------------------------------------
# Tablas de apoyo para el notebook
# ---------------------------------------------------------------------------
def tabla_catalogo(modelos_: list[Modelo]) -> pd.DataFrame:
    """Resumen del diseño del experimento: qué se prueba en cada modelo."""
    filas = [{"modelo": m.nombre, "familia": m.familia,
              "preparacion": "con escalado" if m.escalar else "sin escalado",
              "estrategias_desbalance": ", ".join(m.estrategias()),
              "hiperparametros": ", ".join(k.replace("modelo__", "") for k in m.hiperparametros),
              "candidatos": m.n_candidatos(),
              "ajustes_cv": m.n_candidatos() * config.N_PLIEGUES}
             for m in modelos_]
    return pd.DataFrame(filas).set_index("modelo")


def auc_por_estrategia(comparador: "Comparador") -> pd.DataFrame:
    """Mejor AUC de CV que logra cada modelo con cada estrategia de desbalance."""
    filas = {}
    for nombre, m in comparador.modelos.items():
        filas[nombre] = m.resultados().groupby("estrategia")["auc_cv"].max()
    tabla = pd.DataFrame(filas).T.reindex(columns=list(ESTRATEGIAS))
    tabla["rango_entre_estrategias"] = tabla.max(axis=1) - tabla.min(axis=1)
    tabla["desv_mejor_candidato"] = [comparador.modelos[n].resumen()["auc_cv_desv"] for n in tabla.index]
    return tabla


def auc_por_valor(modelo: Modelo, hiperparametro: str) -> pd.DataFrame:
    """Mejor AUC de CV que logra un modelo con cada valor de un hiperparámetro (por ejemplo,
    cada kernel de SVM o cada función de activación de la red), con su desviación."""
    resultados = modelo.resultados()
    mejores = resultados.loc[resultados.groupby(hiperparametro)["auc_cv"].idxmax()]
    return (mejores.set_index(hiperparametro)[["auc_cv", "auc_cv_desv", "estrategia"]]
            .sort_values("auc_cv", ascending=False))


def combinar_finalistas(pipelines: dict, X: pd.DataFrame, y: pd.Series, referencia: str,
                        cv=None, n_jobs: int = -1) -> pd.DataFrame:
    """¿Combinar los finalistas mejora al mejor modelo solo? Compara, con los mismos pliegues:
    - votación: promedio de las probabilidades fuera de pliegue de los finalistas;
    - stacking: una regresión logística aprende a combinar sus probabilidades (con validación
      cruzada interna, para no usar predicciones de clientes ya vistos).
    Las diferencias contra `referencia` son pareadas, pliegue a pliegue. Solo usa train."""
    cv = cv or validacion_cruzada()
    pliegues = list(cv.split(X, y))
    oof = evaluacion.oof_de_modelos(pipelines, X, y, pliegues, n_jobs)
    apilado = StackingClassifier([(nombre, clone(pipe)) for nombre, pipe in pipelines.items()],
                                 final_estimator=LogisticRegression(max_iter=1000), cv=config.N_PLIEGUES,
                                 stack_method="predict_proba")
    prob = {referencia: oof[referencia],
            "Votación de finalistas": np.mean(list(oof.values()), axis=0),
            "Stacking de finalistas": evaluacion.predicciones_oof(apilado, X, y, pliegues, n_jobs)}
    base = evaluacion.auc_por_pliegue(y, prob[referencia], pliegues)
    filas = []
    for nombre, p in prob.items():
        auc = evaluacion.auc_por_pliegue(y, p, pliegues)
        filas.append({"combinacion": nombre, "auc_cv": auc.mean(), "auc_cv_desv": auc.std(),
                      "diferencia_vs_referencia": (auc - base).mean(), "diferencia_desv": (auc - base).std(),
                      "pliegues_que_mejoran": int((auc > base).sum()),
                      "churners_20": int(negocio.tabla_escenarios(y, p, capacidades=(0.2,)).iloc[0]["churners_capturados"])})
    return pd.DataFrame(filas).set_index("combinacion")


def comparar_rondas(anterior: "Comparador", nueva: "Comparador") -> pd.DataFrame:
    """AUC elegido en la ronda anterior y en la nueva para los modelos que se ampliaron,
    con la desviación de la ronda nueva para juzgar si el cambio es relevante."""
    filas = []
    for nombre, m_nueva in nueva.modelos.items():
        r_ant, r_nue = anterior.modelos[nombre].resumen(), m_nueva.resumen()
        filas.append({"modelo": nombre, "auc_cv_anterior": r_ant["auc_cv"], "auc_cv_nueva": r_nue["auc_cv"],
                      "cambio": r_nue["auc_cv"] - r_ant["auc_cv"], "desv_nueva": r_nue["auc_cv_desv"],
                      "hiperparametros_anterior": r_ant["mejores_hiperparametros"],
                      "hiperparametros_nueva": r_nue["mejores_hiperparametros"]})
    return pd.DataFrame(filas).set_index("modelo")


def combinar(*rondas: "Comparador") -> "Comparador":
    """Comparador final: para cada modelo, su versión de la ronda más reciente en que se ajustó."""
    final = {}
    for ronda in rondas:
        final.update(ronda.modelos)
    orden = list(rondas[0].modelos)
    return Comparador([final[n] for n in orden], cv=rondas[0].cv)


def tablas_candidatas(tabla: pd.DataFrame, modelos_: list[Modelo]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mejora media de AUC y número de pliegues (de 5) en que mejora cada combinación de
    variables candidatas, con los modelos en el orden del catálogo."""
    combinaciones = list(COMBINACIONES_CANDIDATAS)
    orden = [m.nombre for m in modelos_]
    mejora = tabla.pivot(index="modelo", columns="combinacion", values="mejora_media")[combinaciones].loc[orden]
    pliegues = tabla.pivot(index="modelo", columns="combinacion", values="pliegues_que_mejoran")[combinaciones].loc[orden]
    return mejora, pliegues


def aplicar_decision_candidatas(modelos_: list[Modelo], decision: dict[str, tuple[bool, bool]]) -> pd.DataFrame:
    """Asigna a cada modelo del catálogo las variables candidatas que decidió la validación
    cruzada (modifica los modelos) y devuelve la decisión como tabla."""
    for m in modelos_:
        m.ticket_promedio, m.indicadores_saldo = decision[m.nombre]
    return pd.DataFrame(decision, index=["ticket_promedio", "indicadores_saldo"]).T.loc[[m.nombre for m in modelos_]]
