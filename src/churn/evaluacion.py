"""Métricas del proyecto: una sola definición que usan todos los modelos.

- `METRICAS_CV`: métricas de la validación cruzada. Solo usan el orden de los puntajes
  (no un umbral), por eso sirven igual para modelos con o sin balanceo de clases.
- `calcular_metricas`: métricas completas a un umbral dado (para la evaluación final).
  La clase positiva es siempre 1 = churn (el cliente se va).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score)

from churn import config

# AUC: probabilidad de que un churner reciba más riesgo que un cliente que se queda.
# AP (AUC-PR de la clase churn): resume precisión frente a captura de churners; su
# referencia de azar es la tasa de churn (16 %), no 0,5.
METRICAS_CV = {"auc": "roc_auc", "ap": "average_precision"}
METRICA_PRINCIPAL = "auc"


def calcular_metricas(y_real, probabilidad, umbral: float = 0.5) -> dict:
    """Métricas de ranking, calibración y clasificación a un umbral, para la clase churn.

    La matriz de confusión se desarma en el orden de sklearn (tn, fp, fn, tp) y de ahí
    salen sensibilidad = tp / (tp + fn) y especificidad = tn / (tn + fp).
    """
    y_real = np.asarray(y_real).astype(int)
    probabilidad = np.asarray(probabilidad, dtype=float)
    if not set(np.unique(y_real)) <= {0, 1}:
        raise ValueError("y_real debe ser binaria (0 = se queda, 1 = churn)")
    if probabilidad.min() < 0 or probabilidad.max() > 1:
        raise ValueError("probabilidad debe estar entre 0 y 1")
    prediccion = (probabilidad >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, prediccion, labels=[0, 1]).ravel()

    def dividir(a, b):
        return a / b if b > 0 else np.nan

    sensibilidad = dividir(tp, tp + fn)
    precision = dividir(tp, tp + fp)
    return {
        "auc": roc_auc_score(y_real, probabilidad),
        "ap": average_precision_score(y_real, probabilidad),
        "log_loss": log_loss(y_real, np.clip(probabilidad, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "brier": brier_score_loss(y_real, probabilidad),
        "umbral": umbral,
        "accuracy": dividir(tp + tn, tp + tn + fp + fn),
        "sensibilidad": sensibilidad,
        "especificidad": dividir(tn, tn + fp),
        "precision": precision,
        "vpn": dividir(tn, tn + fn),
        "f1": dividir(2 * precision * sensibilidad, precision + sensibilidad),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def tabla_metricas(resultados: dict[str, dict]) -> pd.DataFrame:
    """Une las métricas de varios modelos en una tabla (una fila por modelo)."""
    return pd.DataFrame(resultados).T


# ---------------------------------------------------------------------------
# Evaluación final
# ---------------------------------------------------------------------------
def predicciones_oof(pipeline, X: pd.DataFrame, y: pd.Series, cv, n_jobs: int = -1) -> np.ndarray:
    """Probabilidad de churn "fuera de pliegue": cada cliente de train es predicho por un
    modelo que no lo vio. Sirve para elegir umbrales sin tocar el test."""
    from sklearn.model_selection import cross_val_predict
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]


def auc_por_pliegue(y_real, probabilidad, pliegues) -> np.ndarray:
    """AUC de cada pliegue calculado con predicciones fuera de pliegue. Con los mismos
    pliegues, la resta entre dos modelos es una comparación pareada (mismos clientes)."""
    y = np.asarray(y_real).astype(int)
    p = np.asarray(probabilidad, dtype=float)
    return np.array([roc_auc_score(y[validar], p[validar]) for _, validar in pliegues])


def bootstrap_auc(y_real, probabilidades: dict[str, np.ndarray], referencia: str,
                  n_remuestras: int = 2000, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """Bootstrap pareado del test: se remuestrean clientes con reposición (estratificado por
    clase para conservar el 16 % de churners) y en cada remuestra se recalcula el AUC de
    todos los modelos sobre los mismos clientes. Da el IC 95 % del AUC de cada modelo y
    de su diferencia contra la referencia; si el IC de la diferencia incluye 0, los dos
    modelos son estadísticamente equivalentes en test."""
    y_real = np.asarray(y_real).astype(int)
    rng = np.random.default_rng(semilla)
    positivos, negativos = np.flatnonzero(y_real == 1), np.flatnonzero(y_real == 0)
    nombres = list(probabilidades)
    aucs = np.empty((n_remuestras, len(nombres)))
    for b in range(n_remuestras):
        indices = np.concatenate([rng.choice(positivos, len(positivos), replace=True),
                                  rng.choice(negativos, len(negativos), replace=True)])
        for j, nombre in enumerate(nombres):
            aucs[b, j] = roc_auc_score(y_real[indices], np.asarray(probabilidades[nombre])[indices])
    ref = nombres.index(referencia)
    filas = []
    for j, nombre in enumerate(nombres):
        diferencia = aucs[:, j] - aucs[:, ref]
        filas.append({
            "modelo": nombre,
            "auc_test": roc_auc_score(y_real, probabilidades[nombre]),
            "auc_ic95_inf": np.percentile(aucs[:, j], 2.5), "auc_ic95_sup": np.percentile(aucs[:, j], 97.5),
            "dif_vs_referencia": roc_auc_score(y_real, probabilidades[nombre]) - roc_auc_score(y_real, probabilidades[referencia]),
            "dif_ic95_inf": np.percentile(diferencia, 2.5), "dif_ic95_sup": np.percentile(diferencia, 97.5),
            "prop_remuestras_mejor_que_ref": float((diferencia > 0).mean()),
        })
    tabla = pd.DataFrame(filas).set_index("modelo")
    tabla["equivalente_a_referencia"] = (tabla["dif_ic95_inf"] <= 0) & (tabla["dif_ic95_sup"] >= 0)
    tabla.loc[referencia, "equivalente_a_referencia"] = True
    return tabla


def tabla_calibracion(y_real, probabilidad, n_grupos: int = 10) -> pd.DataFrame:
    """Calibración por deciles de probabilidad predicha: en cada grupo se compara la
    probabilidad promedio que da el modelo con la tasa de churn que realmente ocurrió.
    Un modelo bien calibrado tiene ambas columnas parecidas."""
    datos_ = pd.DataFrame({"y": np.asarray(y_real).astype(int), "p": np.asarray(probabilidad, dtype=float)})
    datos_["grupo"] = pd.qcut(datos_["p"].rank(method="first"), n_grupos, labels=False) + 1
    tabla = datos_.groupby("grupo").agg(clientes=("y", "size"), prob_media=("p", "mean"),
                                        tasa_real=("y", "mean"))
    tabla["diferencia"] = tabla["prob_media"] - tabla["tasa_real"]
    return tabla


def error_calibracion(tabla: pd.DataFrame) -> float:
    """Error medio de calibración: diferencia absoluta promedio entre la probabilidad
    predicha y la tasa real, ponderada por clientes de cada grupo (0 = calibración perfecta)."""
    return float((tabla["clientes"] * tabla["diferencia"].abs()).sum() / tabla["clientes"].sum())


def oof_de_modelos(pipelines: dict, X: pd.DataFrame, y: pd.Series, cv, n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Predicciones fuera de pliegue de cada pipeline con los mismos pliegues."""
    return {nombre: predicciones_oof(pipe, X, y, cv, n_jobs) for nombre, pipe in pipelines.items()}


def resumen_oof(oof: dict[str, np.ndarray], y: pd.Series, auc_cv_medio: dict[str, float]) -> pd.DataFrame:
    """Control de coherencia: AUC de las predicciones fuera de pliegue frente al AUC medio de
    la validación cruzada de la Fase 4. Deben coincidir: son los mismos pipelines y pliegues."""
    return pd.DataFrame({"auc_oof": {n: calcular_metricas(y, p)["auc"] for n, p in oof.items()},
                         "auc_cv_medio": {n: auc_cv_medio[n] for n in oof}})


def metricas_en_test(y_test, probabilidades: dict[str, np.ndarray], umbral: float = 0.5) -> pd.DataFrame:
    """Métricas en test de cada modelo con el mismo umbral (0,5, la vista clásica)."""
    return tabla_metricas({n: calcular_metricas(y_test, p, umbral) for n, p in probabilidades.items()})


def comparar_cv_test(resumen: pd.DataFrame, metricas_05: pd.DataFrame) -> pd.DataFrame:
    """AUC medio de validación cruzada frente al AUC en test y su diferencia."""
    tabla = pd.DataFrame({"auc_cv": resumen["auc_cv_medio"], "auc_test": metricas_05["auc"].astype(float)})
    tabla["diferencia_test_menos_cv"] = tabla["auc_test"] - tabla["auc_cv"]
    return tabla
