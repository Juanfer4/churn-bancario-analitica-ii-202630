"""Cifras citadas en el texto de la Fase 5, 5.6 a 5.8 (explicabilidad y riesgos).

Compara cada número del markdown con las tablas que guarda el notebook en reports/tables
(y con los datos de train para las tasas crudas). Si las tablas no existen, la prueba se
omite: hay que ejecutar el notebook primero.
"""
import pandas as pd
import pytest

from churn import config, datos
from conftest import exigir_tablas

NECESARIAS = ["explicabilidad_importancia.csv", "explicabilidad_dependencia_parcial.csv",
              "explicabilidad_monto_frecuencia.csv", "explicabilidad_monto_frecuencia_tramos.csv",
              "explicabilidad_sensibilidad.csv", "explicabilidad_seis_contactos.csv",
              "explicabilidad_equidad_sexo.csv"]


@pytest.fixture(scope="module")
def t():
    exigir_tablas(NECESARIAS)
    leer = lambda n, **k: pd.read_csv(config.RUTA_TABLAS / n, **k)  # noqa: E731
    return {
        "imp": leer("explicabilidad_importancia.csv", index_col=0),
        "dp": leer("explicabilidad_dependencia_parcial.csv", index_col=[0, 1]),
        "mf": leer("explicabilidad_monto_frecuencia.csv"),
        "tramos": leer("explicabilidad_monto_frecuencia_tramos.csv", index_col=0),
        "sens": leer("explicabilidad_sensibilidad.csv", index_col=0),
        "seis": leer("explicabilidad_seis_contactos.csv", index_col=0),
        "eq": leer("explicabilidad_equidad_sexo.csv", index_col=[0, 1]),
    }


def test_cifras_citadas_en_la_explicabilidad(t, cifras):
    chk = cifras.chk

    def pct(nombre, valor, citado):
        chk(nombre, 100 * valor, citado, 0.005)

    imp, dp, mf, tramos = t["imp"], t["dp"], t["mf"], t["tramos"]
    sens, seis, eq = t["sens"], t["seis"], t["eq"]

    # 5.6 importancia
    assert list(imp.index[:2]) == ["total_trans_ct", "total_trans_amt"]
    assert imp.loc["total_trans_ct", "ranking_ganancia"] == 1 and imp.loc["total_trans_amt", "ranking_ganancia"] == 2
    for v, caida in {"total_trans_ct": 0.1555, "total_trans_amt": 0.0856, "total_amt_chng_q4_q1": 0.0080,
                     "total_relationship_count": 0.0079, "total_ct_chng_q4_q1": 0.0052,
                     "total_revolving_bal": 0.0049, "gender": 0.0001}.items():
        chk(f"caída {v}", imp.loc[v, "caida_auc_media"], caida)
    chk("gan ct", imp.loc["total_trans_ct", "importancia_ganancia_pct"], 27.21, 0.005)
    chk("gan amt", imp.loc["total_trans_amt", "importancia_ganancia_pct"], 21.63, 0.005)
    chk("gan saldo", imp.loc["total_revolving_bal", "importancia_ganancia_pct"], 13.57, 0.005)
    actividad = ["total_trans_ct", "total_trans_amt", "total_ct_chng_q4_q1", "total_amt_chng_q4_q1"]
    chk("gan actividad", imp.loc[actividad, "importancia_ganancia_pct"].sum(), 62.84, 0.005)
    chk("gan categóricas", imp.loc[datos.COLUMNAS_CATEGORICAS, "importancia_ganancia_pct"].sum(), 2.38, 0.005)
    assert imp.loc["total_revolving_bal", "ranking_ganancia"] == 3 and imp.loc["total_revolving_bal", "ranking_permutacion"] == 6
    assert (imp.loc[imp.index[2:], "caida_auc_media"] < 0.01).all()  # "ninguna más de 0,01"
    assert imp.loc["gender", "pliegues_con_caida"] == 3
    for v in ("income_category", "education_level", "card_category"):
        assert imp.loc[v, "caida_auc_media"] < 0.0001, v
    assert imp.loc["dependent_count", "pliegues_con_caida"] == 0
    assert imp.loc["months_inactive_12_mon", "ranking_permutacion"] == 7
    assert "avg_open_to_buy" not in imp.index

    # 5.6 mapa monto x frecuencia
    con_datos = mf[mf["clientes_en_celda"] >= 10]
    assert len(mf) == 256 and len(con_datos) == 74 and con_datos["clientes_en_celda"].sum() == 7856
    pct("máx > 11.000", tramos.loc["más de 11.000", "prob_max"], 2.66)
    medio = con_datos[(con_datos["total_trans_amt"] > 3000) & (con_datos["total_trans_amt"] <= 6000)]
    assert len(medio) == 18 and (medio["prob_churn_media"] < 0.10).sum() == 13
    pct("máx hasta 3.000", tramos.loc["hasta 3.000", "prob_max"], 64.87)

    def celda(monto, transacciones):
        fila = con_datos[(con_datos["total_trans_amt"].round() == monto) & (con_datos["total_trans_ct"].round() == transacciones)]
        assert len(fila) == 1, (monto, transacciones)
        return fila["prob_churn_media"].iloc[0]
    pct("8927/61", celda(8927, 61), 77.72); pct("8927/96", celda(8927, 96), 0.89)
    pct("2928/33", celda(2928, 33), 64.87); pct("2928/82", celda(2928, 82), 0.27)
    assert con_datos.loc[con_datos["prob_churn_media"].idxmax(), "total_trans_amt"].round() == 8927

    # 5.6 dependencia parcial 1D
    def curva(v):
        return dp.loc[v]
    c = curva("total_amt_chng_q4_q1")
    chk("amt_chng x0", c["valor"].iloc[0], 0.34, 0.005); chk("amt_chng x1", c["valor"].iloc[-1], 1.51, 0.005)
    pct("amt_chng p0", c["prob_churn_media"].iloc[0], 20.90); pct("amt_chng p1", c["prob_churn_media"].iloc[-1], 10.42)
    c = curva("total_ct_chng_q4_q1")
    chk("ct_chng x0", c["valor"].iloc[0], 0.23, 0.005); chk("ct_chng x1", c["valor"].iloc[-1], 1.50, 0.005)
    pct("ct_chng p0", c["prob_churn_media"].iloc[0], 18.87); pct("ct_chng p1", c["prob_churn_media"].iloc[-1], 11.56)
    c = curva("total_relationship_count").set_index("valor")["prob_churn_media"]
    pct("prod 1", c[1], 25.03); pct("prod 2", c[2], 20.61)
    pct("prod 3+ mín", c[c.index >= 3].min(), 15.05); pct("prod 3+ máx", c[c.index >= 3].max(), 15.87)
    c = curva("total_revolving_bal")
    pct("saldo 0", c["prob_churn_media"].iloc[0], 17.32); pct("saldo máx", c["prob_churn_media"].iloc[-1], 16.91)
    chk("saldo valor máx", c["valor"].iloc[-1], 2517, 0.5)
    pct("saldo pico", c["prob_churn_media"].max(), 18.24)
    bajos = c[(c["valor"] > 131) & (c["valor"] < 398)]
    assert len(bajos) == 3 and bajos["valor"].round().tolist() == [132, 265, 397]
    pct("bajos mín", bajos["prob_churn_media"].min(), 18.23); pct("bajos máx", bajos["prob_churn_media"].max(), 18.24)
    pct("saldo mín", c["prob_churn_media"].min(), 13.94)
    chk("saldo mín en", c.loc[c["prob_churn_media"].idxmin(), "valor"], 1987, 0.5)
    c = curva("months_inactive_12_mon")
    pct("inact mín", c["prob_churn_media"].min(), 14.23); pct("inact máx", c["prob_churn_media"].max(), 16.40)
    c = curva("customer_age").set_index("valor")["prob_churn_media"]
    pct("edad 27", c[27], 9.18); pct("edad 29", c[29], 14.71)
    pct("edad 31+ mín", c[c.index >= 31].min(), 15.83); pct("edad 31+ máx", c[c.index >= 31].max(), 16.56)
    assert list(dp.index.get_level_values(0).unique()) == list(imp.index[2:8])  # puestos 3 a 8

    # Tasas crudas de train citadas para comparar
    _, p, _ = datos.preparar_datos(guardar=False)  # no depende de data/processed (no se versiona)
    X, y = p.X_train, p.y_train
    crudas_prod = y.groupby(X["total_relationship_count"]).mean()
    pct("crudo prod 1", crudas_prod[1], 26.78); pct("crudo prod 2", crudas_prod[2], 28.62)
    crudas_inact = y.groupby(X["months_inactive_12_mon"]).mean()
    pct("crudo inact 1", crudas_inact[1], 4.01); pct("crudo inact 4", crudas_inact[4], 31.18)

    # 5.7 sensibilidad
    chk("auc completo", sens.loc["completo", "auc_cv"], 0.99415, 0.000005)
    chk("auc sin contactos", sens.loc["sin contactos", "auc_cv"], 0.99360, 0.000005)
    chk("dif sin contactos", sens.loc["sin contactos", "diferencia_vs_completo"], -0.00055, 0.000005)
    assert sens.loc["sin contactos", "pliegues_por_debajo"] == 4
    chk("20 completo", sens.loc["completo", "churners_20"], 1269, 0); chk("10 completo", sens.loc["completo", "churners_10"], 799, 0)
    chk("20 sin contactos", sens.loc["sin contactos", "churners_20"], 1268, 0)
    chk("10 sin contactos", sens.loc["sin contactos", "churners_10"], 798, 0)
    chk("auc sin sexo", sens.loc["sin sexo", "auc_cv"], 0.99413, 0.000005)
    chk("dif sin sexo", sens.loc["sin sexo", "diferencia_vs_completo"], -0.00002, 0.000005)
    chk("desv sin sexo", sens.loc["sin sexo", "diferencia_desv"], 0.00020, 0.000005)
    assert sens.loc["sin sexo", "pliegues_por_debajo"] == 2
    chk("20 sin sexo", sens.loc["sin sexo", "churners_20"], 1272, 0)
    chk("dif sin sexo ni ingresos", sens.loc["sin sexo ni ingresos", "diferencia_vs_completo"], -0.00024, 0.000005)
    assert sens.loc["sin sexo ni ingresos", "pliegues_por_debajo"] == 5
    chk("20 sin sexo ni ingresos", sens.loc["sin sexo ni ingresos", "churners_20"], 1271, 0)
    chk("auc prudente", sens.loc["sin sexo ni contactos", "auc_cv"], 0.99352, 0.000005)
    chk("dif prudente", sens.loc["sin sexo ni contactos", "diferencia_vs_completo"], -0.00063, 0.000005)
    chk("20 prudente", sens.loc["sin sexo ni contactos", "churners_20"], 1272, 0)
    chk("10 prudente", sens.loc["sin sexo ni contactos", "churners_10"], 799, 0)
    chk("costo aprox 0,0006", -sens.loc["sin contactos", "diferencia_vs_completo"], 0.0006, 0.00005)
    churners_train = round(sens.loc["completo", "churners_20"] / sens.loc["completo", "captura_20"])
    chk("churners train", churners_train, 1302, 0)

    chk("seis clientes", seis.loc["completo", "clientes"], 42, 0); chk("seis churners", seis.loc["completo", "churners"], 42, 0)
    chk("seis dentro completo", seis.loc["completo", "dentro_del_escenario"], 41, 0)
    chk("seis dentro sin", seis.loc["sin contactos", "dentro_del_escenario"], 40, 0)
    pct("seis prob completo", seis.loc["completo", "prob_media"], 95.63)
    pct("seis prob sin", seis.loc["sin contactos", "prob_media"], 85.68)
    pct("seis prob mínima", seis.loc["completo", "prob_minima"], 0.31)

    # 5.7 equidad
    for (v, g), (tasa, cont, cap, prec, razon) in {
            ("completo", "F"): (17.53, 20.83, 98.53, 82.92, 1.19), ("completo", "M"): (14.44, 19.07, 96.02, 72.74, 1.32),
            ("sin sexo ni contactos", "F"): (17.53, 20.79, 98.80, 83.33, 1.19),
            ("sin sexo ni contactos", "M"): (14.44, 19.12, 96.20, 72.68, 1.32)}.items():
        fila = eq.loc[(v, g)]
        pct(f"tasa {v} {g}", fila["tasa_churn_real"], tasa); pct(f"cont {v} {g}", fila["pct_contactado"], cont)
        pct(f"cap {v} {g}", fila["captura_en_grupo"], cap); pct(f"prec {v} {g}", fila["precision_en_grupo"], prec)
        chk(f"razón {v} {g}", fila["razon_contacto_riesgo"], razon, 0.005)
    assert (eq["captura_en_grupo"] > 0.96).all()
    for v in ("completo", "sin sexo ni contactos"):
        brecha = 100 * (eq.loc[(v, "F"), "captura_en_grupo"] - eq.loc[(v, "M"), "captura_en_grupo"])
        chk(f"brecha {v}", brecha, {"completo": 2.51, "sin sexo ni contactos": 2.60}[v], 0.005)

    cifras.verificar()
