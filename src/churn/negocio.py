"""Lectura de negocio del modelo: a quién contactar y cuándo conviene (Fase 5, 5.4 y 5.5).

Supuestos explícitos (no son datos del caso):
- Capacidad de contacto: 10 %, 20 % o 30 % de los clientes (config.ESCENARIOS_CAPACIDAD).
- s, tasa de éxito: fracción de churners contactados que se quedan gracias al beneficio.
- r = V / c: cuántas veces vale un cliente retenido frente al costo del beneficio por cliente.

Selección: se ordena a los clientes por probabilidad de churn (de mayor a menor) y se
contacta al top k %. El beneficio se entrega a todos los contactados, se vayan o no.

Beneficio neto de un escenario, en unidades del costo del beneficio:
    TP * s * r - n          (TP churners contactados, n contactados)
Es positivo si r > 1 / (precisión * s): esa es la razón de equilibrio. Para pasar de un
escenario al siguiente se usa la precisión marginal de los contactos que se agregan.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from churn import config


def orden_por_riesgo(probabilidad: np.ndarray) -> np.ndarray:
    """Índices de mayor a menor probabilidad; los empates conservan el orden original."""
    return np.argsort(-np.asarray(probabilidad, dtype=float), kind="stable")


def contactados(n_clientes: int, capacidad: float) -> int:
    """Número de clientes que se contactan con una capacidad dada (redondeo al entero más cercano)."""
    if not 0 < capacidad <= 1:
        raise ValueError("La capacidad debe estar entre 0 y 1")
    return int(math.floor(capacidad * n_clientes + 0.5))


def tabla_escenarios(y_real, probabilidad, capacidades=config.ESCENARIOS_CAPACIDAD) -> pd.DataFrame:
    """Resultado de contactar al top k % por riesgo, para cada capacidad.

    captura: churners contactados / churners totales (qué parte del problema se cubre).
    precision: churners entre los contactados (qué parte de los beneficios llega a quien se iba).
    lift: precisión / tasa de churn (cuántas veces mejor que contactar al azar).
    umbral_implicito: probabilidad del último cliente contactado.
    """
    y = np.asarray(y_real).astype(int)
    p = np.asarray(probabilidad, dtype=float)
    orden = orden_por_riesgo(p)
    y_ordenado, p_ordenado = y[orden], p[orden]
    total_churners, tasa = int(y.sum()), float(y.mean())
    filas = []
    for k in capacidades:
        n = contactados(len(y), k)
        tp = int(y_ordenado[:n].sum())
        filas.append({
            "capacidad": k, "contactados": n, "churners_capturados": tp,
            "captura": tp / total_churners, "precision": tp / n, "lift": (tp / n) / tasa,
            "beneficios_a_no_churners": n - tp, "churners_sin_contactar": total_churners - tp,
            "umbral_implicito": float(p_ordenado[n - 1]),
            "captura_maxima_posible": min(n, total_churners) / total_churners,
        })
    return pd.DataFrame(filas).set_index("capacidad")


def tabla_incremental(escenarios: pd.DataFrame) -> pd.DataFrame:
    """Qué aporta cada escenario frente al anterior: contactos y churners adicionales y la
    precisión marginal de esos contactos (la que decide si ampliar vale la pena)."""
    filas, anterior = [], None
    for k, fila in escenarios.iterrows():
        n_prev, tp_prev = (0, 0) if anterior is None else (anterior["contactados"], anterior["churners_capturados"])
        dn, dtp = fila["contactados"] - n_prev, fila["churners_capturados"] - tp_prev
        filas.append({"capacidad": k, "contactos_adicionales": int(dn), "churners_adicionales": int(dtp),
                      "precision_marginal": dtp / dn if dn > 0 else np.nan})
        anterior = fila
    return pd.DataFrame(filas).set_index("capacidad")


def razon_equilibrio(precision: float | np.ndarray, tasa_exito: float) -> float | np.ndarray:
    """Valor mínimo de r = V / c para que un grupo de contactos no pierda dinero:
    r* = 1 / (precisión * s). Si la precisión es 0, nunca se recupera el costo (infinito)."""
    precision = np.asarray(precision, dtype=float)
    if not 0 < tasa_exito <= 1:
        raise ValueError("La tasa de éxito debe estar entre 0 y 1")
    with np.errstate(divide="ignore"):
        resultado = np.where(precision > 0, 1 / (precision * tasa_exito), np.inf)
    return float(resultado) if resultado.ndim == 0 else resultado


def tabla_equilibrio(escenarios: pd.DataFrame, incremental: pd.DataFrame,
                     tasas_exito=(0.2, 0.5, 1.0)) -> pd.DataFrame:
    """Razón de equilibrio promedio (todo el escenario) y marginal (solo los contactos que
    agrega frente al escenario anterior), para varias tasas de éxito supuestas."""
    filas = []
    for k in escenarios.index:
        for s in tasas_exito:
            filas.append({"capacidad": k, "tasa_exito": s,
                          "r_equilibrio_promedio": razon_equilibrio(escenarios.loc[k, "precision"], s),
                          "r_equilibrio_marginal": razon_equilibrio(incremental.loc[k, "precision_marginal"], s)})
    return pd.DataFrame(filas)


def beneficio_neto(escenarios: pd.DataFrame, tasa_exito: float, razon_valor_costo: float) -> pd.Series:
    """Beneficio neto de cada escenario en unidades del costo del beneficio: TP * s * r - n."""
    return escenarios["churners_capturados"] * tasa_exito * razon_valor_costo - escenarios["contactados"]


def tabla_deciles(y_real, probabilidad, n_grupos: int = 10) -> pd.DataFrame:
    """Tabla de ganancia por deciles de riesgo (decil 1 = el 10 % de mayor riesgo)."""
    y = np.asarray(y_real).astype(int)
    orden = orden_por_riesgo(probabilidad)
    grupos = np.array_split(orden, n_grupos)
    total, tasa = y.sum(), y.mean()
    filas, acumulado = [], 0
    for d, indices in enumerate(grupos, start=1):
        churners = int(y[indices].sum())
        acumulado += churners
        filas.append({"decil": d, "clientes": len(indices), "churners": churners,
                      "tasa_churn": churners / len(indices), "lift": (churners / len(indices)) / tasa,
                      "captura_acumulada": acumulado / total,
                      "prob_min": float(np.min(np.asarray(probabilidad)[indices])),
                      "prob_max": float(np.max(np.asarray(probabilidad)[indices]))})
    return pd.DataFrame(filas).set_index("decil")


def curva_ganancia(y_real, probabilidad) -> pd.DataFrame:
    """Fracción de churners capturados según la fracción de clientes contactados (por riesgo)."""
    y = np.asarray(y_real).astype(int)
    orden = orden_por_riesgo(probabilidad)
    captura = np.concatenate([[0], np.cumsum(y[orden]) / y.sum()])
    contactado = np.arange(len(y) + 1) / len(y)
    return pd.DataFrame({"fraccion_contactada": contactado, "captura": captura})


def bootstrap_escenarios(y_real, probabilidad, capacidades=config.ESCENARIOS_CAPACIDAD,
                         n_remuestras: int = 2000, semilla: int = config.SEMILLA) -> pd.DataFrame:
    """IC 95 % de captura y precisión de cada escenario por bootstrap estratificado: mide
    cuánto podrían variar esas cifras con otra muestra de clientes del mismo banco."""
    y = np.asarray(y_real).astype(int)
    p = np.asarray(probabilidad, dtype=float)
    rng = np.random.default_rng(semilla)
    positivos, negativos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    resultados = {k: {"captura": [], "precision": []} for k in capacidades}
    for _ in range(n_remuestras):
        indices = np.concatenate([rng.choice(positivos, len(positivos), replace=True),
                                  rng.choice(negativos, len(negativos), replace=True)])
        tabla = tabla_escenarios(y[indices], p[indices], capacidades)
        for k in capacidades:
            resultados[k]["captura"].append(tabla.loc[k, "captura"])
            resultados[k]["precision"].append(tabla.loc[k, "precision"])
    filas = []
    for k in capacidades:
        captura, precision = np.array(resultados[k]["captura"]), np.array(resultados[k]["precision"])
        filas.append({"capacidad": k,
                      "captura_ic95_inf": np.percentile(captura, 2.5), "captura_ic95_sup": np.percentile(captura, 97.5),
                      "precision_ic95_inf": np.percentile(precision, 2.5), "precision_ic95_sup": np.percentile(precision, 97.5)})
    return pd.DataFrame(filas).set_index("capacidad")


def curva_beneficio(y_real, probabilidad, producto_rs: float) -> pd.DataFrame:
    """Beneficio neto acumulado al contactar a los n clientes de mayor riesgo, para un valor
    de s * r (tasa de éxito por razón valor/costo). Se expresa en unidades del costo del
    beneficio por cliente de la base: (TP(n) * s * r - n) / N."""
    y = np.asarray(y_real).astype(int)
    orden = orden_por_riesgo(probabilidad)
    tp = np.concatenate([[0], np.cumsum(y[orden])])
    n = np.arange(len(y) + 1)
    return pd.DataFrame({"fraccion_contactada": n / len(y), "contactados": n, "churners_capturados": tp,
                         "beneficio_por_cliente": (tp * producto_rs - n) / len(y)})


def tabla_beneficio_optimo(y_real, probabilidad, productos_rs=config.PRODUCTOS_SR,
                           capacidades=config.ESCENARIOS_CAPACIDAD) -> pd.DataFrame:
    """Para cada valor supuesto de s * r: la fracción de contacto que maximiza el beneficio
    neto y el beneficio de cada escenario de capacidad (en unidades de costo por cliente)."""
    y = np.asarray(y_real).astype(int)
    filas = []
    for rs in productos_rs:
        curva = curva_beneficio(y, probabilidad, rs)
        mejor = curva.loc[curva["beneficio_por_cliente"].idxmax()]
        fila = {"producto_s_r": rs, "fraccion_optima": mejor["fraccion_contactada"],
                "captura_en_optimo": mejor["churners_capturados"] / y.sum(),
                "beneficio_optimo_por_cliente": mejor["beneficio_por_cliente"]}
        for k in capacidades:
            fila[f"beneficio_por_cliente_{int(round(100 * k))}"] = curva.loc[contactados(len(y), k), "beneficio_por_cliente"]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("producto_s_r")


def lectura_test(y_real, probabilidad, n_remuestras: int = 2000) -> dict[str, pd.DataFrame]:
    """Lectura de negocio sobre test: escenarios, incremental, intervalos por bootstrap y deciles."""
    escenarios = tabla_escenarios(y_real, probabilidad)
    return {"escenarios": escenarios, "incremental": tabla_incremental(escenarios),
            "intervalos": bootstrap_escenarios(y_real, probabilidad, n_remuestras=n_remuestras),
            "deciles": tabla_deciles(y_real, probabilidad)}


def beneficio_optimo_por_producto(conjuntos: dict[str, tuple], productos_rs=config.PRODUCTOS_SR) -> dict[str, pd.DataFrame]:
    """Contacto óptimo y beneficio por cliente para cada conjunto `{nombre: (y_real, probabilidad)}`."""
    return {nombre: tabla_beneficio_optimo(y, p, productos_rs) for nombre, (y, p) in conjuntos.items()}


def escenarios_por_modelo(y_real, probabilidades: dict[str, np.ndarray],
                          capacidades=config.ESCENARIOS_CAPACIDAD) -> pd.DataFrame:
    """Churners capturados y captura de cada modelo en cada escenario de contacto: compara los
    modelos con la métrica con la que decide el banco (el top k % por riesgo), no solo por AUC."""
    filas = {}
    for nombre, prob in probabilidades.items():
        escenarios = tabla_escenarios(y_real, prob, capacidades)
        fila = {}
        for k in capacidades:
            etiqueta = int(round(100 * k))
            fila[f"churners_{etiqueta}"] = int(escenarios.loc[k, "churners_capturados"])
            fila[f"captura_{etiqueta}"] = escenarios.loc[k, "captura"]
        filas[nombre] = fila
    return pd.DataFrame(filas).T
