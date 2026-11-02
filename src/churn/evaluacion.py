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


def umbral_max_f1(y_real, probabilidad) -> float:
    """Umbral que maximiza el F1 (equilibrio entre precisión y captura de churners)."""
    from sklearn.metrics import precision_recall_curve
    precision, sensibilidad, umbrales = precision_recall_curve(y_real, probabilidad)
    f1 = 2 * precision[:-1] * sensibilidad[:-1] / np.clip(precision[:-1] + sensibilidad[:-1], 1e-12, None)
    return float(umbrales[int(np.argmax(f1))])


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


def tabla_calibracion_rangos(y_real, probabilidad,
                             cortes=(0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1)) -> pd.DataFrame:
    """Calibración por rangos fijos de probabilidad predicha (no por deciles).

    Con un modelo que da probabilidad casi 0 a la mayoría, los deciles juntan muchos grupos
    vacíos de churn y diluyen el error; los rangos fijos muestran dónde se equivoca la
    probabilidad (por ejemplo, en el riesgo intermedio)."""
    datos_ = pd.DataFrame({"y": np.asarray(y_real).astype(int), "p": np.asarray(probabilidad, dtype=float)})
    etiquetas = [f"{a:.0%} a {b:.0%}" for a, b in zip(cortes[:-1], cortes[1:])]
    datos_["rango"] = pd.cut(datos_["p"], list(cortes), labels=etiquetas, include_lowest=True)
    tabla = datos_.groupby("rango", observed=True).agg(clientes=("y", "size"), prob_media=("p", "mean"),
                                                       tasa_real=("y", "mean"))
    tabla["diferencia"] = tabla["prob_media"] - tabla["tasa_real"]
    return tabla


def error_calibracion(tabla: pd.DataFrame) -> float:
    """Error medio de calibración: diferencia absoluta promedio entre la probabilidad
    predicha y la tasa real, ponderada por clientes de cada grupo (0 = calibración perfecta)."""
    return float((tabla["clientes"] * tabla["diferencia"].abs()).sum() / tabla["clientes"].sum())


def puntos_de_operacion(y_real, probabilidad, capturas=(0.80, 0.90, 0.95),
                        falsas_alarmas_max=0.05) -> dict:
    """Lectura de las curvas en puntos concretos: la mejor precisión posible si se quiere
    capturar al menos cierta fracción de churners, y la mejor sensibilidad si se tolera como
    máximo cierta tasa de falsas alarmas."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    precision, captura, _ = precision_recall_curve(y_real, probabilidad)
    salida = {f"precision_con_captura_{int(round(100 * c))}": float(precision[captura >= c].max()) for c in capturas}
    fpr, tpr, _ = roc_curve(y_real, probabilidad)
    salida[f"sensibilidad_con_falsas_alarmas_{int(round(100 * falsas_alarmas_max))}"] = float(tpr[fpr <= falsas_alarmas_max].max())
    return salida


def oof_de_modelos(pipelines: dict, X: pd.DataFrame, y: pd.Series, cv, n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Predicciones fuera de pliegue de cada pipeline con los mismos pliegues."""
    return {nombre: predicciones_oof(pipe, X, y, cv, n_jobs) for nombre, pipe in pipelines.items()}


def resumen_oof(oof: dict[str, np.ndarray], y: pd.Series, auc_cv_medio: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    """Umbral de máximo F1 por modelo (fijado solo con train) y tabla que lo compara con el
    AUC de las predicciones fuera de pliegue y el AUC medio de la validación cruzada."""
    umbrales = {nombre: umbral_max_f1(y, prob) for nombre, prob in oof.items()}
    resumen = pd.DataFrame({
        "auc_oof": {n: calcular_metricas(y, p)["auc"] for n, p in oof.items()},
        "auc_cv_medio": {n: auc_cv_medio[n] for n in oof},
        "umbral_max_f1": umbrales,
    })
    return umbrales, resumen


def metricas_en_test(y_test, probabilidades: dict[str, np.ndarray], umbrales_f1: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas en test con umbral 0,5 para todos y con el umbral de máximo F1 fijado en train."""
    con_05 = tabla_metricas({n: calcular_metricas(y_test, p, 0.5) for n, p in probabilidades.items()})
    con_f1 = tabla_metricas({n: calcular_metricas(y_test, p, umbrales_f1[n]) for n, p in probabilidades.items()})
    return con_05, con_f1


def comparar_cv_test(resumen: pd.DataFrame, metricas_05: pd.DataFrame) -> pd.DataFrame:
    """AUC medio de validación cruzada frente al AUC en test y su diferencia."""
    tabla = pd.DataFrame({"auc_cv": resumen["auc_cv_medio"], "auc_test": metricas_05["auc"].astype(float)})
    tabla["diferencia_test_menos_cv"] = tabla["auc_test"] - tabla["auc_cv"]
    return tabla
