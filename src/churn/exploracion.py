"""Estadísticas del análisis exploratorio (EDA).

Solo calcula tablas; los gráficos viven en `graficos.py`. Todas las funciones reciben
los datos de entrenamiento: el EDA se hace sobre train para que ninguna decisión de
preparación se tome mirando el conjunto de prueba.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

Z_95 = 1.959963984540054  # cuantil normal para un intervalo de confianza del 95 %

# Tramos descriptivos del monto anual de transacciones (2.4). Se definen una sola vez porque
# los reutiliza la lectura del mapa conjunto de monto y transacciones (5.6).
TRAMOS_MONTO = ([0, 3000, 6000, 11000, np.inf], ["hasta 3.000", "3.001 a 6.000", "6.001 a 11.000", "más de 11.000"])


def intervalo_wilson(exitos: np.ndarray | int, n: np.ndarray | int,
                     z: float = Z_95) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de confianza de Wilson para una proporción.

    Se prefiere al intervalo normal clásico porque no se sale de [0, 1] y se comporta
    bien con grupos pequeños o con tasas cercanas a 0 % o 100 % (por ejemplo, 42 de 42).
    """
    exitos = np.asarray(exitos, dtype=float)
    n = np.asarray(n, dtype=float)
    p = exitos / n
    denominador = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denominador
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominador
    # El recorte solo corrige errores de redondeo en los extremos (por ejemplo 42 de 42)
    inferior = np.clip(centro - margen, 0.0, p)
    superior = np.clip(centro + margen, p, 1.0)
    return inferior, superior


def resumen_numericas(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Poder de separación y forma de cada variable numérica.

    - `auc_univariada`: AUC que lograría la variable sola para ordenar churners por encima
      de clientes que se quedan (0,5 = no separa; 1 = separa perfecto). Se reporta como
      max(AUC, 1 - AUC) para que el sentido no afecte la comparación.
    - `sentido`: si valores más altos se asocian a más o a menos churn.
    - `asimetria`: coeficiente de asimetría (cola larga si |valor| > 1).
    - `pct_atipicos_iqr`: porcentaje fuera de [Q1 - 1,5 IQR, Q3 + 1,5 IQR].
    """
    filas = []
    for col in X.select_dtypes("number").columns:
        serie = X[col]
        auc = roc_auc_score(y, serie)
        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        atipicos = ((serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)).mean()
        filas.append({
            "variable": col,
            "auc_univariada": max(auc, 1 - auc),
            "sentido": "más alto, más churn" if auc >= 0.5 else "más alto, menos churn",
            "mediana_se_queda": serie[y == 0].median(),
            "mediana_se_va": serie[y == 1].median(),
            "asimetria": serie.skew(),
            "pct_atipicos_iqr": 100 * atipicos,
        })
    return (pd.DataFrame(filas).set_index("variable")
            .sort_values("auc_univariada", ascending=False))


def tasa_por_nivel(X: pd.DataFrame, y: pd.Series, columna: str,
                   orden: list | None = None) -> pd.DataFrame:
    """Tasa de churn por cada valor de una variable discreta o categórica, con IC 95 %.

    `orden` fija el orden de presentación; debe contener exactamente los niveles presentes,
    para que ningún grupo desaparezca sin aviso.
    """
    tabla = (pd.DataFrame({columna: X[columna], "churn": y})
             .groupby(columna, observed=True)["churn"].agg(clientes="count", churners="sum"))
    if orden is not None:
        if set(orden) != set(tabla.index):
            raise ValueError(f"El orden de {columna} no coincide con sus niveles: "
                             f"{sorted(set(orden) ^ set(tabla.index))}")
        tabla = tabla.loc[orden]
    tabla["tasa_churn"] = tabla["churners"] / tabla["clientes"]
    tabla["ic95_inf"], tabla["ic95_sup"] = intervalo_wilson(tabla["churners"], tabla["clientes"])
    return tabla


def asociacion_categoricas(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Prueba chi cuadrado y V de Cramér entre cada categórica y el churn.

    - `p_valor`: probabilidad de ver una asociación así si en realidad no hubiera ninguna.
    - `v_cramer`: fuerza de la asociación de 0 (ninguna) a 1 (total); por debajo de 0,1
      se considera débil, sin importar si el p-valor es pequeño.
    """
    filas = []
    for col in X.select_dtypes(exclude="number").columns:
        tabla = pd.crosstab(X[col], y)
        chi2, p_valor, _, _ = stats.chi2_contingency(tabla)
        n = tabla.values.sum()
        v = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))
        conteos = tabla.sum(axis=1)
        filas.append({
            "variable": col,
            "categorias": len(tabla),
            "p_valor": p_valor,
            "v_cramer": v,
            "categoria_menos_frecuente": conteos.idxmin(),
            "clientes_categoria_menos_frecuente": int(conteos.min()),
        })
    return pd.DataFrame(filas).set_index("variable").sort_values("v_cramer", ascending=False)


def pares_correlacionados(X: pd.DataFrame, umbral: float = 0.5,
                          metodo: str = "spearman") -> pd.DataFrame:
    """Pares de variables numéricas con |correlación| >= umbral (Spearman por defecto,
    porque capta relaciones monótonas aunque no sean lineales y resiste los atípicos)."""
    corr = X.select_dtypes("number").corr(method=metodo)
    triangulo = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pares = triangulo[triangulo.abs() >= umbral].rename("correlacion").reset_index()
    pares.columns = ["variable_1", "variable_2", "correlacion"]
    return pares.reindex(pares["correlacion"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def concentracion_moda(X: pd.DataFrame, columna: str) -> dict:
    """Valor más frecuente de una variable y qué porcentaje de clientes lo tiene."""
    conteo = X[columna].value_counts()
    return {"valor": conteo.index[0], "clientes": int(conteo.iloc[0]),
            "porcentaje": 100 * conteo.iloc[0] / len(X)}


def tramos(serie: pd.Series, cortes: list[float], etiquetas: list[str]) -> pd.Series:
    """Agrupa una variable continua en tramos fijos solo para describirla en el EDA.

    Los tramos no se usan como variables del modelo: fijar umbrales mirando los datos
    podría sobreajustar, y los modelos de árboles aprenden sus propios cortes.
    """
    resultado = pd.cut(serie, bins=cortes, labels=etiquetas, include_lowest=True)
    if resultado.isna().any():
        raise ValueError(f"Hay valores de {serie.name} fuera de los cortes {cortes}")
    return resultado


def perfil_automatico(X: pd.DataFrame, y: pd.Series, ruta_html=None) -> pd.DataFrame:
    """Perfil automático de los datos (EDA automático con ydata-profiling) y sus alertas.

    Recorre todas las variables: distribución, ceros, valores únicos, correlaciones de Spearman
    y asociación entre categóricas. Devuelve una tabla con una fila por alerta (`tipo` y
    `variable`); si se indica `ruta_html`, guarda además el reporte completo para consultarlo.
    Es un barrido de cobertura: señala qué revisar, no toma decisiones. Debe usarse solo con
    train, igual que el resto del EDA.
    """
    import os
    os.environ["YDATA_PROFILING_NO_ANALYTICS"] = "True"  # la librería no debe enviar datos de uso
    from ydata_profiling import ProfileReport

    tabla = X.assign(churn=np.asarray(y))
    perfil = ProfileReport(
        tabla, title="EDA automático de train", progress_bar=False,
        correlations={"auto": {"calculate": False}, "pearson": {"calculate": False},
                      "spearman": {"calculate": True}, "kendall": {"calculate": False},
                      "phi_k": {"calculate": False}, "cramers": {"calculate": True}},
        interactions={"continuous": False}, samples={"head": 0, "tail": 0}, duplicates={"head": 0})
    alertas = pd.DataFrame([{"tipo": alerta.alert_type.name, "variable": alerta.column_name}
                            for alerta in perfil.get_description().alerts], columns=["tipo", "variable"])
    if ruta_html is not None:
        perfil.to_file(ruta_html, silent=True)
    return alertas.sort_values(["tipo", "variable"]).reset_index(drop=True)
