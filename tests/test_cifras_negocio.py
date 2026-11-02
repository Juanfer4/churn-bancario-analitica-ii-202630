"""Cifras citadas en el texto de la Fase 5, 5.4 y 5.5 (lectura de negocio).

Compara cada número del markdown con las tablas que guarda el notebook en reports/tables.
Si las tablas no existen, la prueba se omite: hay que ejecutar el notebook primero.
"""
import pandas as pd
import pytest

from churn import config, negocio
from conftest import exigir_tablas

NECESARIAS = ["negocio_escenarios_oof.csv", "negocio_incremental_oof.csv", "negocio_escenarios_oof_logistica.csv",
              "negocio_escenarios_test.csv", "negocio_incremental_test.csv", "negocio_ic_escenarios_test.csv",
              "negocio_deciles_test.csv", "negocio_equilibrio_oof.csv", "negocio_optimo_oof.csv",
              "negocio_optimo_test.csv"]


@pytest.fixture(scope="module")
def t():
    exigir_tablas(NECESARIAS)
    salida = {}
    for n in NECESARIAS:
        indice = None if n == "negocio_equilibrio_oof.csv" else 0
        salida[n.replace(".csv", "")] = pd.read_csv(config.RUTA_TABLAS / n, index_col=indice)
    return salida


def test_cifras_citadas_en_la_lectura_de_negocio(t, cifras):
    chk = cifras.chk

    eo, io, el = t["negocio_escenarios_oof"], t["negocio_incremental_oof"], t["negocio_escenarios_oof_logistica"]
    # 5.4 (train fuera de pliegue)
    for k, (n, tp, cap, prec, lift) in {0.1: (810, 799, 61.37, 98.64, 6.14), 0.2: (1620, 1269, 97.47, 78.33, 4.87),
                                         0.3: (2430, 1296, 99.54, 53.33, 3.32)}.items():
        chk(f"n {k}", eo.loc[k, "contactados"], n, 0); chk(f"tp {k}", eo.loc[k, "churners_capturados"], tp, 0)
        chk(f"cap {k}", 100 * eo.loc[k, "captura"], cap, 0.005); chk(f"prec {k}", 100 * eo.loc[k, "precision"], prec, 0.005)
        chk(f"lift {k}", eo.loc[k, "lift"], lift, 0.005)
    chk("cap máx 10", 100 * eo.loc[0.1, "captura_maxima_posible"], 62.21, 0.005)
    chk("churners train", eo.loc[0.1, "churners_capturados"] + eo.loc[0.1, "churners_sin_contactar"], 1302, 0)
    chk("fp 20", eo.loc[0.2, "beneficios_a_no_churners"], 351, 0)
    chk("adic 30", io.loc[0.3, "churners_adicionales"], 27, 0); chk("adic 20", io.loc[0.2, "churners_adicionales"], 470, 0)
    chk("pm 30", 100 * io.loc[0.3, "precision_marginal"], 3.33, 0.005); chk("pm 20", 100 * io.loc[0.2, "precision_marginal"], 58.02, 0.005)
    chk("LR tp 20", el.loc[0.2, "churners_capturados"], 1076, 0); chk("LR cap 20", 100 * el.loc[0.2, "captura"], 82.64, 0.005)
    chk("ventaja", eo.loc[0.2, "churners_capturados"] - el.loc[0.2, "churners_capturados"], 193, 0)

    # 5.4 (test)
    et, it, ic, d = t["negocio_escenarios_test"], t["negocio_incremental_test"], t["negocio_ic_escenarios_test"], t["negocio_deciles_test"]
    chk("t n10", et.loc[0.1, "contactados"], 203, 0); chk("t tp10", et.loc[0.1, "churners_capturados"], 200, 0)
    chk("t prec10", 100 * et.loc[0.1, "precision"], 98.52, 0.005)
    chk("t tp20", et.loc[0.2, "churners_capturados"], 315, 0); chk("t n20", et.loc[0.2, "contactados"], 405, 0)
    chk("t cap20", 100 * et.loc[0.2, "captura"], 96.92, 0.005); chk("t prec20", 100 * et.loc[0.2, "precision"], 77.78, 0.005)
    chk("ic cap20 inf", 100 * ic.loc[0.2, "captura_ic95_inf"], 94.77, 0.005); chk("ic cap20 sup", 100 * ic.loc[0.2, "captura_ic95_sup"], 98.46, 0.005)
    chk("ic prec20 inf", 100 * ic.loc[0.2, "precision_ic95_inf"], 76.05, 0.005); chk("ic prec20 sup", 100 * ic.loc[0.2, "precision_ic95_sup"], 79.01, 0.005)
    chk("t tp30", et.loc[0.3, "churners_capturados"], 324, 0); chk("t n30", et.loc[0.3, "contactados"], 608, 0)
    chk("t cap30", 100 * et.loc[0.3, "captura"], 99.69, 0.005)
    chk("t pm20", 100 * it.loc[0.2, "precision_marginal"], 56.93, 0.005); chk("t pm30", 100 * it.loc[0.3, "precision_marginal"], 4.43, 0.005)
    chk("d1 tasa", 100 * d.loc[1, "tasa_churn"], 98.52, 0.005); chk("d1 lift", d.loc[1, "lift"], 6.14, 0.005)
    chk("d2 tasa", 100 * d.loc[2, "tasa_churn"], 56.65, 0.005)
    assert (d.loc[5:, "churners"] == 0).all() and d.loc[4, "churners"] > 0
    chk("umbral 20 test", et.loc[0.2, "umbral_implicito"], 0.0113, 0.00005)

    # 5.5
    eq = t["negocio_equilibrio_oof"].set_index(["capacidad", "tasa_exito"])
    for (k, s), v in {(0.1, 0.2): 5.07, (0.1, 0.5): 2.03, (0.1, 1.0): 1.01, (0.2, 0.2): 6.38, (0.2, 0.5): 2.55,
                      (0.2, 1.0): 1.28, (0.3, 0.2): 9.38, (0.3, 0.5): 3.75, (0.3, 1.0): 1.88}.items():
        chk(f"r prom {k} {s}", eq.loc[(k, s), "r_equilibrio_promedio"], v, 0.005)
    chk("r marg 10", eq.loc[(0.1, 1.0), "r_equilibrio_marginal"], 1.01, 0.005)
    chk("r marg 20", eq.loc[(0.2, 1.0), "r_equilibrio_marginal"], 1.72, 0.005)
    chk("r marg 30", eq.loc[(0.3, 1.0), "r_equilibrio_marginal"], 30.00, 0.005)
    chk("r marg 30 s02", eq.loc[(0.3, 0.2), "r_equilibrio_marginal"], 150.0, 0.005)
    oo, ot = t["negocio_optimo_oof"], t["negocio_optimo_test"]
    for rs, frac in {1.5: 14.37, 3.0: 16.81, 10.0: 20.78, 30.0: 22.26}.items():
        chk(f"óptimo {rs}", 100 * oo.loc[rs, "fraccion_optima"], frac, 0.005)
    escenarios = ["beneficio_por_cliente_10", "beneficio_por_cliente_20", "beneficio_por_cliente_30"]
    assert oo.loc[1.5, escenarios].idxmax() == "beneficio_por_cliente_10"
    for rs in (3.0, 10.0, 30.0):
        assert oo.loc[rs, escenarios].max() == oo.loc[rs, "beneficio_por_cliente_20"], rs
    assert abs(oo.loc[30.0, "beneficio_por_cliente_20"] - oo.loc[30.0, "beneficio_por_cliente_30"]) < 1e-9
    chk("óptimo test mín", 100 * ot["fraccion_optima"].min(), 14.61, 0.005)
    chk("óptimo test máx", 100 * ot["fraccion_optima"].max(), 23.40, 0.005)
    chk("contactos base", negocio.contactados(10127, 0.2), 2025, 0)

    cifras.verificar()
