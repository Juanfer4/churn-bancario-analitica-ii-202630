"""Cifras citadas en la Fase 5 (5.9, conclusiones) y la Fase 6 (modelo viable).

Compara cada número del markdown con las tablas que guarda el notebook. Si faltan, la
prueba se omite: hay que ejecutar el notebook primero.
"""
import pandas as pd
import pytest

from churn import config
from conftest import exigir_tablas

NECESARIAS = ["test_bootstrap_auc.csv", "test_cv_vs_test.csv", "negocio_escenarios_test.csv",
              "negocio_incremental_test.csv", "explicabilidad_importancia.csv", "modelo_viable_cv.csv",
              "modelo_viable_escenarios_cv.csv", "modelo_viable_fichas.csv"]


@pytest.fixture(scope="module")
def t():
    exigir_tablas(NECESARIAS)
    indices = {"modelo_viable_escenarios_cv.csv": [0, 1]}
    return {n.replace(".csv", ""): pd.read_csv(config.RUTA_TABLAS / n, index_col=indices.get(n, 0)) for n in NECESARIAS}


def test_cifras_citadas_en_conclusiones_y_modelo_viable(t, cifras):
    chk = cifras.chk

    def pct(nombre, valor, citado):
        chk(nombre, 100 * valor, citado, 0.005)

    # Fase 5, 5.9
    b, cvt = t["test_bootstrap_auc"], t["test_cv_vs_test"]
    chk("auc test", b.loc["LightGBM", "auc_test"], 0.9933)
    chk("ic inf", b.loc["LightGBM", "auc_ic95_inf"], 0.9906); chk("ic sup", b.loc["LightGBM", "auc_ic95_sup"], 0.9957)
    chk("auc cv", cvt.loc["LightGBM", "auc_cv"], 0.9941); chk("brecha", -cvt.loc["LightGBM", "diferencia_test_menos_cv"], 0.0008)
    chk("logística", b.loc["Regresión logística", "auc_test"], 0.9445)
    assert b.loc[["Gradient Boosting", "XGBoost"], "equivalente_a_referencia"].all()
    assert not b.loc["Regresión logística", "equivalente_a_referencia"]
    e, inc = t["negocio_escenarios_test"], t["negocio_incremental_test"]
    for k, (n, tp, cap, prec) in {0.1: (203, 200, 61.54, 98.52), 0.2: (405, 315, 96.92, 77.78),
                                  0.3: (608, 324, 99.69, 53.29)}.items():
        chk(f"n {k}", e.loc[k, "contactados"], n, 0); chk(f"tp {k}", e.loc[k, "churners_capturados"], tp, 0)
        pct(f"cap {k}", e.loc[k, "captura"], cap); pct(f"prec {k}", e.loc[k, "precision"], prec)
    chk("churners test", e.loc[0.2, "churners_capturados"] + e.loc[0.2, "churners_sin_contactar"], 325, 0)
    pct("marginal 30", inc.loc[0.3, "precision_marginal"], 4.43)
    imp = t["explicabilidad_importancia"]
    actividad = ["total_trans_ct", "total_trans_amt", "total_ct_chng_q4_q1", "total_amt_chng_q4_q1"]
    chk("actividad", imp.loc[actividad, "importancia_ganancia_pct"].sum(), 62.84, 0.005)

    # Fase 6
    cv, esc, fichas = t["modelo_viable_cv"], t["modelo_viable_escenarios_cv"], t["modelo_viable_fichas"]
    chk("auc oficial cv", cv.loc["oficial", "auc_cv"], 0.99415, 0.000005)
    chk("auc candidato cv", cv.loc["candidato de producción", "auc_cv"], 0.99352, 0.000005)
    chk("ap oficial", cv.loc["oficial", "ap_cv_oof"], 0.9734, 0.00005)
    chk("ap candidato", cv.loc["candidato de producción", "ap_cv_oof"], 0.9709, 0.00005)
    chk("20 oficial", cv.loc["oficial", "churners_20"], 1269, 0)
    chk("20 candidato", cv.loc["candidato de producción", "churners_20"], 1272, 0)
    pct("cap 20 oficial", cv.loc["oficial", "captura_20"], 97.47)
    pct("cap 20 candidato", cv.loc["candidato de producción", "captura_20"], 97.70)
    chk("10 candidato", esc.loc[("candidato de producción", 0.1), "churners_capturados"], 799, 0)
    chk("30 candidato", esc.loc[("candidato de producción", 0.3), "churners_capturados"], 1295, 0)
    chk("30 oficial", esc.loc[("oficial", 0.3), "churners_capturados"], 1296, 0)
    assert (cv["origen"] == "validación cruzada en train").all()
    chk("diferencia", float(fichas.loc["metrica__diferencia_auc_vs_oficial", "candidato_produccion"]), -0.00063, 0.000005)
    chk("diferencia desv", float(fichas.loc["metrica__diferencia_desv", "candidato_produccion"]), 0.00039, 0.000005)
    chk("pliegues", float(fichas.loc["metrica__pliegues_por_debajo", "candidato_produccion"]), 4, 0)
    chk("columnas candidato", float(fichas.loc["n_columnas_entrada", "candidato_produccion"]), 17, 0)
    chk("columnas oficial", float(fichas.loc["n_columnas_entrada", "oficial"]), 19, 0)
    assert fichas.loc["evaluado_en_test", "candidato_produccion"] == "False"
    assert fichas.loc["evaluado_en_test", "oficial"] == "True"
    chk("auc test en ficha oficial", float(fichas.loc["metrica__auc_test", "oficial"]), 0.9933)
    assert pd.isna(fichas.loc["metrica__auc_test", "candidato_produccion"])

    cifras.verificar()
