"""Cifras citadas en el texto de la Fase 3 del notebook, recalculadas por otra vía.

La asimetría tras Yeo-Johnson se recalcula con scipy.stats.yeojohnson (no con sklearn), y
los conteos de columnas se reconstruyen a partir de las categorías de los datos.
"""
import numpy as np
import pandas as pd
from scipy import stats

from churn import datos, features


def test_cifras_citadas_en_la_preparacion(cifras):
    limpio = datos.limpiar_datos(datos.cargar_datos())
    p = datos.particionar(limpio)
    X = p.X_train
    chk = cifras.chk

    # Conteo de columnas reconstruido desde los datos
    numericas = X.select_dtypes("number").columns.drop("avg_open_to_buy")
    chk("numéricas originales", len(numericas), 13, 0)
    n_onehot = 0
    for col in datos.COLUMNAS_CATEGORICAS:
        conteo = X[col].value_counts()
        frecuentes = (conteo >= features.MIN_FRECUENCIA_CATEGORIA).sum()
        hay_infrecuentes = (conteo < features.MIN_FRECUENCIA_CATEGORIA).any()
        grupos = frecuentes + int(hay_infrecuentes)
        n_onehot += 1 if grupos == 2 else grupos
    chk("One-Hot", n_onehot, 21, 0)
    chk("total con candidatas", 13 + 3 + n_onehot, 37, 0)
    chk("total sin candidatas", 13 + n_onehot, 34, 0)
    raras = {c: set(X[c].value_counts()[lambda s: s < 100].index) for c in datos.COLUMNAS_CATEGORICAS}
    assert raras["card_category"] == {"Gold", "Platinum"}
    assert all(not v for k, v in raras.items() if k != "card_category")
    chk("saldo máximo", X["total_revolving_bal"].max(), 2517, 0)

    # Asimetría antes y después con scipy (Yeo-Johnson + estandarización)
    ticket = X["total_trans_amt"] / X["total_trans_ct"]
    columnas = {"total_ct_chng_q4_q1": X["total_ct_chng_q4_q1"], "total_trans_amt": X["total_trans_amt"],
                "total_amt_chng_q4_q1": X["total_amt_chng_q4_q1"], "credit_limit": X["credit_limit"],
                "ticket_promedio": ticket}
    citado = {"total_ct_chng_q4_q1": (2.25, -0.08), "total_trans_amt": (2.05, 0.00),
              "total_amt_chng_q4_q1": (1.80, -0.09), "credit_limit": (1.68, 0.10),
              "ticket_promedio": (1.90, -0.01)}
    for nombre, serie in columnas.items():
        transformada, _ = stats.yeojohnson(serie.to_numpy(dtype=float))
        chk(f"asimetría antes {nombre}", pd.Series(serie).skew(), citado[nombre][0], 0.005)
        chk(f"asimetría después {nombre}", pd.Series(transformada).skew(), citado[nombre][1], 0.01)

    # Media máxima del test transformado con los parámetros de train
    prep = features.construir_preparacion(True, True, True).fit(X)
    nombres = features.nombres_salida(prep)
    Z_test = pd.DataFrame(prep.transform(p.X_test), columns=nombres)
    media_test = Z_test[features.columnas_numericas_salida(prep)].mean().abs().max()
    chk("máx |media test|", media_test, 0.047, 0.0005)
    assert np.isfinite(Z_test.values).all()

    cifras.verificar()
