"""Cifras citadas en el texto de la Fase 5, 5.1 a 5.3 (evaluación final en test).

Compara cada número del markdown con las tablas que guarda el notebook en reports/tables.
Si las tablas no existen, la prueba se omite: hay que ejecutar el notebook primero.
"""
import pandas as pd
import pytest

from churn import config
from conftest import exigir_tablas

NECESARIAS = ["test_umbrales_oof.csv", "test_metricas_umbral_05.csv", "test_metricas_umbral_f1.csv",
              "test_cv_vs_test.csv", "test_bootstrap_auc.csv", "test_calibracion.csv",
              "test_calibracion_deciles_lightgbm.csv", "test_puntos_operacion.csv"]


@pytest.fixture(scope="module")
def t():
    exigir_tablas(NECESARIAS)
    leer = lambda n: pd.read_csv(config.RUTA_TABLAS / n, index_col=0)  # noqa: E731
    return {n.replace(".csv", ""): leer(n) for n in NECESARIAS}


def test_cifras_citadas_en_la_evaluacion_final(t, cifras):
    chk = cifras.chk

    LG, GB, XG, LR = "LightGBM", "Gradient Boosting", "XGBoost", "Regresión logística"
    # 5.1 (umbrales con train)
    oof = t["test_umbrales_oof"]
    for m, v in {LG: 0.9941, GB: 0.9934, XG: 0.9936, LR: 0.9509}.items():
        chk(f"auc oof {m}", oof.loc[m, "auc_oof"], v)
    assert ((oof["auc_oof"] - oof["auc_cv_medio"]).abs() < 0.0002).all()
    chk("auc cv LR", oof.loc[LR, "auc_cv_medio"], 0.9510)
    for m, v in {LG: 0.2219, GB: 0.4263, XG: 0.4041, LR: 0.4239}.items():
        chk(f"umbral {m}", oof.loc[m, "umbral_max_f1"], v)
    assert (oof["umbral_max_f1"] < 0.5).all()

    # 5.1 (test)
    m05, mf1, cvt = t["test_metricas_umbral_05"], t["test_metricas_umbral_f1"], t["test_cv_vs_test"]
    for m, v in {LG: 0.9933, XG: 0.9934, GB: 0.9927, LR: 0.9445}.items():
        chk(f"auc test {m}", m05.loc[m, "auc"], v)
    finalistas = cvt.loc[[LG, GB, XG], "diferencia_test_menos_cv"]
    chk("caída mín", -finalistas.max(), 0.0002); chk("caída máx", -finalistas.min(), 0.0008)
    chk("caída LR", -cvt.loc[LR, "diferencia_test_menos_cv"], 0.0065)
    chk("caída LGBM", -cvt.loc[LG, "diferencia_test_menos_cv"], 0.0008)
    total = m05.loc[LG, ["tp", "fp", "fn", "tn"]].sum()
    churners = m05.loc[LG, "tp"] + m05.loc[LG, "fn"]
    chk("n test", total, 2026, 0); chk("churners test", churners, 325, 0)
    chk("tasa test", 100 * churners / total, 16.04, 0.005); chk("baseline", 100 * (1 - churners / total), 83.96, 0.005)
    assert (m05.loc[[LG, GB, XG], "accuracy"] > 0.97).all() and (mf1.loc[[LG, GB, XG], "accuracy"] > 0.97).all()
    chk("LG05 tp", m05.loc[LG, "tp"], 285, 0); chk("LG05 fp", m05.loc[LG, "fp"], 14, 0)
    chk("LG05 sens", 100 * m05.loc[LG, "sensibilidad"], 87.69, 0.005); chk("LG05 prec", 100 * m05.loc[LG, "precision"], 95.32, 0.005)
    chk("LGf1 tp", mf1.loc[LG, "tp"], 292, 0); chk("LGf1 fn", mf1.loc[LG, "fn"], 33, 0); chk("LGf1 fp", mf1.loc[LG, "fp"], 26, 0)
    chk("LGf1 sens", 100 * mf1.loc[LG, "sensibilidad"], 89.85, 0.005); chk("LGf1 prec", 100 * mf1.loc[LG, "precision"], 91.82, 0.005)
    chk("LRf1 tp", mf1.loc[LR, "tp"], 229, 0); chk("LRf1 fn", mf1.loc[LR, "fn"], 96, 0); chk("LRf1 fp", mf1.loc[LR, "fp"], 64, 0)
    chk("LRf1 sens", 100 * mf1.loc[LR, "sensibilidad"], 70.46, 0.005); chk("LRf1 prec", 100 * mf1.loc[LR, "precision"], 78.16, 0.005)
    chk("AP LG", m05.loc[LG, "ap"], 0.9686); chk("AP LR", m05.loc[LR, "ap"], 0.8039)
    for m, v in {LG: 0.1019, XG: 0.0803, GB: 0.0848}.items():
        chk(f"logloss {m}", m05.loc[m, "log_loss"], v)
    assert m05.loc[[LG, GB, XG], "log_loss"].idxmax() == LG
    for m, v in {LG: 0.0233, XG: 0.0226, GB: 0.0228}.items():
        chk(f"brier {m}", m05.loc[m, "brier"], v)

    # 5.1 (curvas)
    po = t["test_puntos_operacion"]
    chk("prec90 LG", 100 * po.loc[LG, "precision_con_captura_90"], 91.85, 0.005)
    chk("prec90 LR", 100 * po.loc[LR, "precision_con_captura_90"], 49.08, 0.005)
    for m, v in {LG: 96.62, GB: 95.38, XG: 96.00, LR: 72.62}.items():
        chk(f"sens5 {m}", 100 * po.loc[m, "sensibilidad_con_falsas_alarmas_5"], v, 0.005)
    fin = mf1.loc[[LG, GB, XG]]
    chk("tp mín", fin["tp"].min(), 284, 0); chk("tp máx", fin["tp"].max(), 292, 0)
    chk("fp mín", fin["fp"].min(), 13, 0); chk("fp máx", fin["fp"].max(), 26, 0)
    chk("no churners", mf1.loc[LG, "fp"] + mf1.loc[LG, "tn"], 1701, 0)

    # 5.2
    b = t["test_bootstrap_auc"]
    chk("ic inf LG", b.loc[LG, "auc_ic95_inf"], 0.9906); chk("ic sup LG", b.loc[LG, "auc_ic95_sup"], 0.9957)
    chk("dif XGB fina", b.loc[XG, "dif_vs_referencia"], 0.00004, 0.000005)
    for m, (d, lo, hi, prop) in {GB: (-0.0006, -0.0017, 0.0005, 14.90), XG: (0.0000, -0.0007, 0.0008, 54.70),
                                 LR: (-0.0488, -0.0602, -0.0389, 0.0)}.items():
        chk(f"dif {m}", b.loc[m, "dif_vs_referencia"], d); chk(f"lo {m}", b.loc[m, "dif_ic95_inf"], lo)
        chk(f"hi {m}", b.loc[m, "dif_ic95_sup"], hi)
        chk(f"prop {m}", 100 * b.loc[m, "prop_remuestras_mejor_que_ref"], prop, 0.005)
    assert bool(b.loc[GB, "equivalente_a_referencia"]) and bool(b.loc[XG, "equivalente_a_referencia"])
    assert not bool(b.loc[LR, "equivalente_a_referencia"])

    # 5.3
    c = t["test_calibracion"]
    for m, v in {LG: 0.0148, GB: 0.0152, XG: 0.0125, LR: 0.0099}.items():
        chk(f"ece {m}", c.loc[m, "error_calibracion"], v)
    assert (c["error_calibracion"] < 0.02).all()
    chk("ece oof LG", c.loc[LG, "error_calibracion_oof_train"], 0.0095)
    assert c["error_calibracion"].idxmin() == LR
    chk("brier LR", c.loc[LR, "brier"], 0.0617)
    chk("p media LG", 100 * c.loc[LG, "prob_media_predicha"], 14.84, 0.005)
    chk("tasa real", 100 * c.loc[LG, "tasa_real"], 16.04, 0.005)
    d = t["test_calibracion_deciles_lightgbm"]
    chk("clientes 6 deciles", d.loc[1:6, "clientes"].sum(), 1216, 0)
    chk("churners 6 deciles", (d.loc[1:6, "clientes"] * d.loc[1:6, "tasa_real"]).sum(), 0, 1e-9)
    assert (d.loc[1:6, "prob_media"] < 1e-4).all()
    chk("d9 pred", 100 * d.loc[9, "prob_media"], 48.24, 0.005); chk("d9 real", 100 * d.loc[9, "tasa_real"], 56.93, 0.005)
    chk("d10 pred", 100 * d.loc[10, "prob_media"], 99.94, 0.005); chk("d10 real", 100 * d.loc[10, "tasa_real"], 98.52, 0.005)
    top2 = (d.loc[9:10, "clientes"] * d.loc[9:10, "tasa_real"]).round().sum()
    chk("churners top 20 %", top2, 315, 0); chk("pct top 20 %", 100 * top2 / 325, 96.92, 0.005)
    chk("clientes top 20 %", 100 * d.loc[9:10, "clientes"].sum() / 2026, 20.0, 0.05)

    cifras.verificar()
