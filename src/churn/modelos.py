"""Modelos del experimento: la clase genérica `Modelo` y el catálogo de ocho modelos.

La comparación entre modelos, las rondas de búsqueda y la caché viven en `comparacion.py`.

Agregar un modelo al experimento cuesta una línea en `catalogo_modelos()`: nombre,
estimador, grilla de hiperparámetros y cómo se le aplican los pesos de clase.

Cada modelo es un pipeline de imblearn con cuatro pasos planos:
    variables -> codificador -> balanceo -> modelo
`balanceo` es "passthrough" (sin balanceo) o SMOTE (clientes sintéticos de la clase
churn, creados solo dentro del pliegue de entrenamiento). La tercera estrategia, pesos
de clase, se aplica en el propio modelo (por ejemplo class_weight="balanced").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from churn import config, evaluacion, features

ESTRATEGIAS = ("ninguna", "pesos", "SMOTE")
PESOS_PROPORCION = "proporcion"  # se reemplaza por (clientes que se quedan / churners) de train


def validacion_cruzada(semilla: int = config.SEMILLA) -> StratifiedKFold:
    """Validación cruzada estratificada de 5 pliegues, con barajado fijo: todos los modelos
    se evalúan exactamente en los mismos pliegues, así sus resultados son comparables."""
    return StratifiedKFold(n_splits=config.N_PLIEGUES, shuffle=True, random_state=semilla)


@dataclass
class Modelo:
    """Un algoritmo con su grilla, su variante de preparación y su manejo del desbalance.

    nombre: nombre para tablas y gráficos.
    familia: lineal, kernel, red neuronal, árbol, bagging o boosting.
    estimador: el algoritmo de sklearn, XGBoost o LightGBM con sus valores fijos.
    hiperparametros: grilla de GridSearchCV, con claves del pipeline (por ejemplo "modelo__C").
    escalar: True si el modelo necesita Yeo-Johnson y estandarización.
    pesos: (parámetro, valor) para la estrategia de pesos de clase, o None si no la admite.
    smote: True si se prueba la estrategia SMOTE.
    usar_categoricas: False si el modelo no recibe las cinco categóricas (lo decide 4.3).
    """

    nombre: str
    familia: str
    estimador: BaseEstimator
    hiperparametros: dict
    escalar: bool
    pesos: tuple | None = None
    smote: bool = True
    ticket_promedio: bool = False
    indicadores_saldo: bool = False
    usar_categoricas: bool = True
    busqueda_: GridSearchCV | None = field(default=None, repr=False)
    tiempo_busqueda_: float | None = None

    # --- Construcción -----------------------------------------------------------------
    def pipeline(self) -> Pipeline:
        """Pipeline completo sin ajustar: preparación + balanceo + modelo."""
        return Pipeline(
            features.pasos_preparacion(self.escalar, self.ticket_promedio, self.indicadores_saldo,
                                       categoricas=None if self.usar_categoricas else [])
            + [("balanceo", "passthrough"), ("modelo", clone(self.estimador))]
        )

    def estrategias(self) -> list[str]:
        """Estrategias de desbalance que admite este modelo."""
        return ["ninguna"] + (["pesos"] if self.pesos else []) + (["SMOTE"] if self.smote else [])

    def grilla(self, y: pd.Series | None = None) -> list[dict]:
        """Grilla completa: cada combinación de hiperparámetros con cada estrategia."""
        grilla = []
        for estrategia in self.estrategias():
            bloque = {k: list(v) for k, v in self.hiperparametros.items()}
            if estrategia == "SMOTE":
                bloque["balanceo"] = [SMOTE(random_state=config.SEMILLA)]
            else:
                bloque["balanceo"] = ["passthrough"]
            if self.pesos:
                parametro, valor = self.pesos
                if estrategia == "pesos":
                    bloque[f"modelo__{parametro}"] = [self._resolver_peso(valor, y)]
                else:
                    bloque[f"modelo__{parametro}"] = [self._peso_neutro(parametro)]
            grilla.append(bloque)
        return grilla

    @staticmethod
    def _resolver_peso(valor, y):
        if valor == PESOS_PROPORCION:
            if y is None:
                raise ValueError("Se necesita y para calcular la proporción de clases")
            return float((y == 0).sum() / (y == 1).sum())
        return valor

    @staticmethod
    def _peso_neutro(parametro: str):
        return 1.0 if parametro == "scale_pos_weight" else None

    def n_candidatos(self) -> int:
        return sum(int(np.prod([len(v) for v in bloque.values()])) for bloque in self.grilla(pd.Series([0, 1])))

    # --- Ajuste -------------------------------------------------------------------------
    def ajustar(self, X: pd.DataFrame, y: pd.Series, cv=None, n_jobs: int = -1) -> "Modelo":
        """GridSearchCV sobre la grilla completa; elige por AUC y guarda también train."""
        cv = cv or validacion_cruzada()
        self.busqueda_ = GridSearchCV(
            self.pipeline(), self.grilla(y), scoring=evaluacion.METRICAS_CV,
            refit=self.seleccionar_candidato, cv=cv, n_jobs=n_jobs,
            return_train_score=True, error_score="raise")
        inicio = time.perf_counter()
        self.busqueda_.fit(X, y)
        self.tiempo_busqueda_ = time.perf_counter() - inicio
        return self

    def seleccionar_candidato(self, cv_results: dict) -> int:
        """Regla de elección del mejor candidato (se pasa a GridSearchCV como `refit`).

        1. Se toma el candidato con mayor AUC medio de CV y su desviación entre pliegues.
        2. Todos los candidatos a menos de esa desviación se consideran empatados.
        3. Entre los empatados se prefiere la estrategia de desbalance más simple
           (ninguna, luego pesos, luego SMOTE): no altera las probabilidades ni crea
           clientes sintéticos. Dentro de esa estrategia gana el de mayor AUC.
        """
        medias = np.asarray(cv_results["mean_test_auc"])
        desvios = np.asarray(cv_results["std_test_auc"])
        mejor = int(np.argmax(medias))
        empatados = np.flatnonzero(medias >= medias[mejor] - desvios[mejor])
        rango_estrategia = np.array([ESTRATEGIAS.index(self.estrategia_de(cv_results["params"][i]))
                                     for i in empatados])
        mas_simples = empatados[rango_estrategia == rango_estrategia.min()]
        return int(mas_simples[np.argmax(medias[mas_simples])])

    def _verificar_ajustado(self) -> None:
        if self.busqueda_ is None:
            raise RuntimeError(f"El modelo {self.nombre} no se ha ajustado")

    # --- Resultados ---------------------------------------------------------------------
    def estrategia_de(self, parametros: dict) -> str:
        """Traduce los parámetros de un candidato a su estrategia de desbalance."""
        if not isinstance(parametros.get("balanceo"), str):
            return "SMOTE"
        if self.pesos:
            valor = parametros.get(f"modelo__{self.pesos[0]}")
            if valor not in (None, 1.0):
                return "pesos"
        return "ninguna"

    def resultados(self) -> pd.DataFrame:
        """Todos los candidatos probados, con su estrategia y sus métricas de CV y train."""
        self._verificar_ajustado()
        r = self.busqueda_.cv_results_
        filas = []
        for i, parametros in enumerate(r["params"]):
            fila = {k.replace("modelo__", ""): v for k, v in parametros.items()
                    if k != "balanceo" and not (self.pesos and k == f"modelo__{self.pesos[0]}")}
            fila.update({
                "estrategia": self.estrategia_de(parametros),
                "auc_cv": r["mean_test_auc"][i], "auc_cv_desv": r["std_test_auc"][i],
                "ap_cv": r["mean_test_ap"][i], "auc_train": r["mean_train_auc"][i],
                "tiempo_ajuste_s": r["mean_fit_time"][i], "rango": r["rank_test_auc"][i],
            })
            filas.append(fila)
        return pd.DataFrame(filas).sort_values("rango").reset_index(drop=True)

    def auc_por_pliegue(self) -> np.ndarray:
        """AUC del mejor candidato en cada pliegue (para comparaciones pareadas)."""
        self._verificar_ajustado()
        i = self.busqueda_.best_index_
        return np.array([self.busqueda_.cv_results_[f"split{k}_test_auc"][i]
                         for k in range(self.busqueda_.n_splits_)])

    def resumen(self) -> dict:
        """Una fila por modelo para la tabla comparativa."""
        self._verificar_ajustado()
        i = self.busqueda_.best_index_
        r = self.busqueda_.cv_results_
        mejores = {k.replace("modelo__", ""): v for k, v in self.busqueda_.best_params_.items()
                   if k != "balanceo" and not (self.pesos and k == f"modelo__{self.pesos[0]}")}
        return {
            "modelo": self.nombre, "familia": self.familia,
            "auc_cv": r["mean_test_auc"][i], "auc_cv_desv": r["std_test_auc"][i],
            "ap_cv": r["mean_test_ap"][i], "auc_train": r["mean_train_auc"][i],
            "brecha_train_cv": r["mean_train_auc"][i] - r["mean_test_auc"][i],
            "tiempo_ajuste_s": r["mean_fit_time"][i], "tiempo_busqueda_s": self.tiempo_busqueda_,
            "candidatos": len(r["params"]),
            "estrategia": self.estrategia_de(self.busqueda_.best_params_),
            "ticket_promedio": self.ticket_promedio, "indicadores_saldo": self.indicadores_saldo,
            "mejores_hiperparametros": mejores,
        }

    def hiperparametros_en_borde(self) -> list[dict]:
        """Hiperparámetros numéricos cuyo mejor valor quedó en un extremo de la grilla.

        Si el óptimo está en el borde, quizá el mejor valor está fuera de lo probado:
        hay que ampliar la grilla o declararlo.
        """
        self._verificar_ajustado()
        hallazgos = []
        for clave, valores in self.hiperparametros.items():
            # Las arquitecturas de red (tuplas de capas) se ordenan por número total de neuronas
            tamano = {v: (sum(v) if isinstance(v, tuple) else v) for v in valores}
            numericos = [v for v in valores if isinstance(tamano[v], (int, float)) and not isinstance(v, bool)]
            if len(numericos) < 2 or len(numericos) != len(valores):
                continue
            elegido = self.busqueda_.best_params_[clave]
            minimo = min(numericos, key=tamano.get)
            maximo = max(numericos, key=tamano.get)
            if elegido in (minimo, maximo):
                hallazgos.append({"modelo": self.nombre, "hiperparametro": clave.replace("modelo__", ""),
                                  "elegido": elegido, "probados": sorted(numericos, key=tamano.get),
                                  "extremo": "mínimo" if elegido == minimo else "máximo"})
        return hallazgos


# ---------------------------------------------------------------------------
# Catálogo de modelos del experimento
# ---------------------------------------------------------------------------
def catalogo_modelos(semilla: int = config.SEMILLA) -> list[Modelo]:
    """Los diez modelos del experimento, uno por línea: los métodos de clasificación vistos
    en el curso (el porqué de cada uno está en 4.2 del notebook). Valores fijos
    (semilla, iteraciones) en el estimador; lo que se busca, en la grilla."""
    arbol_1 = DecisionTreeClassifier(max_depth=1, random_state=semilla)
    return [
        Modelo("Regresión logística", "lineal",
               LogisticRegression(solver="liblinear", max_iter=5000, random_state=semilla),
               {"modelo__C": [0.01, 0.1, 1, 10, 100], "modelo__penalty": ["l1", "l2"]},
               escalar=True, pesos=("class_weight", "balanced")),
        Modelo("SVM", "kernel",
               SVC(cache_size=1000, random_state=semilla),
               {"modelo__kernel": ["linear", "poly", "rbf", "sigmoid"], "modelo__C": [0.1, 1, 10],
                "modelo__gamma": ["scale", 0.1]},
               escalar=True, pesos=("class_weight", "balanced")),
        Modelo("Red neuronal (MLP)", "red neuronal",
               MLPClassifier(early_stopping=True, max_iter=1000, random_state=semilla),
               {"modelo__activation": ["relu", "tanh", "logistic"],
                "modelo__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
                "modelo__alpha": [1e-4, 1e-3, 1e-2]},
               escalar=True, pesos=None),
        Modelo("Árbol de decisión", "árbol",
               DecisionTreeClassifier(random_state=semilla),
               {"modelo__max_depth": [3, 5, 8, None], "modelo__min_samples_leaf": [1, 5, 20]},
               escalar=False, pesos=("class_weight", "balanced")),
        Modelo("Bagging de árboles", "bagging",
               BaggingClassifier(estimator=DecisionTreeClassifier(random_state=semilla), n_estimators=200,
                                 n_jobs=1, random_state=semilla),
               {"modelo__max_samples": [0.5, 1.0], "modelo__max_features": [0.5, 1.0]},
               escalar=False, pesos=None),
        Modelo("Random Forest", "bagging",
               RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=semilla),
               {"modelo__max_depth": [None, 12], "modelo__min_samples_leaf": [1, 5],
                "modelo__max_features": ["sqrt", 0.5]},
               escalar=False, pesos=("class_weight", "balanced_subsample")),
        Modelo("AdaBoost", "boosting",
               AdaBoostClassifier(estimator=arbol_1, algorithm="SAMME", random_state=semilla),
               {"modelo__estimator__max_depth": [1, 2], "modelo__n_estimators": [300, 600],
                "modelo__learning_rate": [0.1, 0.5, 1.0]},
               escalar=False, pesos=None),
        Modelo("Gradient Boosting", "boosting",
               GradientBoostingClassifier(subsample=0.8, random_state=semilla),
               {"modelo__n_estimators": [200, 400], "modelo__learning_rate": [0.05, 0.1],
                "modelo__max_depth": [2, 3]},
               escalar=False, pesos=None),
        Modelo("XGBoost", "boosting",
               XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             n_jobs=1, random_state=semilla),
               {"modelo__n_estimators": [300, 600], "modelo__learning_rate": [0.03, 0.1],
                "modelo__max_depth": [3, 5]},
               escalar=False, pesos=("scale_pos_weight", PESOS_PROPORCION)),
        Modelo("LightGBM", "boosting",
               LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              verbose=-1, n_jobs=1, random_state=semilla),
               {"modelo__n_estimators": [300, 600], "modelo__learning_rate": [0.03, 0.1],
                "modelo__num_leaves": [15, 31]},
               escalar=False, pesos=("class_weight", "balanced")),
    ]
