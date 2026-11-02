"""Cifras citadas en el texto de la Fase 4 del notebook.

Compara cada número del markdown con las tablas que el notebook guarda en reports/tables
(la salida real de la ejecución). Si las tablas no existen, la prueba se omite: hay que
ejecutar el notebook primero. Una segunda prueba recalcula las tablas desde la caché de
modelos para confirmar que las tablas corresponden a los modelos guardados.
"""
import ast

import joblib
import numpy as np
import pandas as pd
import pytest

from churn import comparacion, config, modelos
from conftest import exigir_tablas

TABLAS = config.RUTA_TABLAS
NECESARIAS = ["modelado_candidatas.csv", "modelado_tabla_cv.csv", "modelado_estrategias.csv",
              "modelado_rondas_1_2.csv", "modelado_rondas_2_3.csv", "modelado_bordes_ronda_1.csv",
              "modelado_bordes_final.csv", "modelado_pareada.csv", "modelado_pycaret.csv"]


def _leer(nombre, **kwargs):
    return pd.read_csv(TABLAS / nombre, **kwargs)


@pytest.fixture(scope="module")
def tablas():
    exigir_tablas(NECESARIAS)
    return {
        "cand": _leer("modelado_candidatas.csv"),
        "cv": _leer("modelado_tabla_cv.csv", index_col=0),
        "estr": _leer("modelado_estrategias.csv", index_col=0),
        "r12": _leer("modelado_rondas_1_2.csv", index_col=0),
        "r23": _leer("modelado_rondas_2_3.csv", index_col=0),
        "b1": _leer("modelado_bordes_ronda_1.csv"),
        "bf": _leer("modelado_bordes_final.csv"),
        "par": _leer("modelado_pareada.csv", index_col=0),
        "pyc": _leer("modelado_pycaret.csv", index_col=0),
    }


def test_cifras_citadas_en_el_modelado(tablas, cifras):
    chk = cifras.chk

    cand = tablas["cand"].set_index(["modelo", "combinacion"])
    # 4.1
    chk("LR +ambas", cand.loc[("Regresión logística", "+ ticket + saldo"), "mejora_media"], 0.0130)
    chk("SVM +ambas", cand.loc[("SVM (kernel RBF)", "+ ticket + saldo"), "mejora_media"], 0.0047)
    chk("RF +ambas", cand.loc[("Random Forest", "+ ticket + saldo"), "mejora_media"], 0.0022)
    for modelo in ["Regresión logística", "SVM (kernel RBF)", "Random Forest"]:
        assert cand.loc[(modelo, "+ ticket + saldo"), "pliegues_que_mejoran"] == 5, modelo
    otros = cand.loc[["Red neuronal (MLP)", "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM"]]
    otros = otros[otros.index.get_level_values("combinacion") != "base"]
    chk("mín mejora otros", otros["mejora_media"].min(), -0.0009)
    chk("máx mejora otros", otros["mejora_media"].max(), 0.0007)
    xgb_saldo = cand.loc[("XGBoost", "+ saldo")]
    chk("xgb saldo", xgb_saldo["mejora_media"], 0.0006); chk("xgb saldo desv", xgb_saldo["mejora_desv"], 0.0005)
    assert xgb_saldo["pliegues_que_mejoran"] == 4 and xgb_saldo["mejora_media"] > xgb_saldo["mejora_desv"]
    ada = cand.loc[("AdaBoost", "+ ticket")]
    chk("ada ticket", ada["mejora_media"], 0.0007)
    assert ada["pliegues_que_mejoran"] == 4 and ada["mejora_desv"] > ada["mejora_media"]

    # 4.2 (ronda 1: se reconstruye a partir de la tabla de estrategias y de las rondas)
    estr = tablas["estr"]
    r1 = tablas["r12"]["auc_cv_anterior"]
    chk("r1 LGBM", r1["LightGBM"], 0.9941); chk("r1 XGB", r1["XGBoost"], 0.9935)
    chk("r1 GB", r1["Gradient Boosting"], 0.9929); chk("r1 Ada", r1["AdaBoost"], 0.9920)
    chk("r1 RF", r1["Random Forest"], 0.9894); chk("r1 SVM", r1["SVM (kernel RBF)"], 0.9739)
    chk("r1 MLP", r1["Red neuronal (MLP)"], 0.9684)
    chk("r1 LR", tablas["cv"].loc["Regresión logística", "auc_cv"], 0.9510)
    chk("rango mín", estr["rango_entre_estrategias"].min(), 0.0001)
    assert estr["rango_entre_estrategias"].idxmin() == "XGBoost"
    chk("rango máx", estr["rango_entre_estrategias"].max(), 0.0017)
    assert estr["rango_entre_estrategias"].idxmax() == "Random Forest"
    assert (estr["rango_entre_estrategias"] < estr["desv_mejor_candidato"]).all()
    chk("desv mín", estr["desv_mejor_candidato"].min(), 0.0013)
    chk("desv máx", estr["desv_mejor_candidato"].max(), 0.0031)
    ventaja = estr[list(modelos.ESTRATEGIAS)].max(axis=1) - estr["ninguna"]
    assert ventaja.idxmax() == "Red neuronal (MLP)"
    chk("ventaja SMOTE MLP", ventaja.max(), 0.0015)
    chk("desv MLP r1", estr.loc["Red neuronal (MLP)", "desv_mejor_candidato"], 0.0031)

    # 4.3
    b1 = tablas["b1"]
    chk("bordes ronda 1", len(b1), 14, 0)
    chk("bordes boosting r1", b1["modelo"].isin(["AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM"]).sum(), 12, 0)
    assert "Regresión logística" not in set(b1["modelo"])

    # 4.4
    r12 = tablas["r12"]
    sin_mlp = r12.drop(index="Red neuronal (MLP)")
    chk("cambio mín r2", sin_mlp["cambio"].min(), 0.0001)
    assert sin_mlp["cambio"].idxmin() == "XGBoost"
    chk("cambio máx r2", sin_mlp["cambio"].max(), 0.0013)
    assert sin_mlp["cambio"].idxmax() == "SVM (kernel RBF)"
    assert (sin_mlp["cambio"] < sin_mlp["desv_nueva"]).all()
    chk("MLP r2 cambio", r12.loc["Red neuronal (MLP)", "cambio"], 0.0042)
    chk("MLP r2 desv", r12.loc["Red neuronal (MLP)", "desv_nueva"], 0.0025)
    r23 = tablas["r23"]
    chk("MLP r3 cambio", r23.loc["Red neuronal (MLP)", "cambio"], 0.0004)
    chk("MLP r3 desv", r23.loc["Red neuronal (MLP)", "desv_nueva"], 0.0038)
    assert ast.literal_eval(r23.loc["Red neuronal (MLP)", "hiperparametros_nueva"])["alpha"] == 0.01
    chk("bordes finales", len(tablas["bf"]), 13, 0)

    # 4.5
    cv = tablas["cv"]
    for modelo, (auc, desv) in {"LightGBM": (0.9941, 0.0011), "Gradient Boosting": (0.9935, 0.0020),
                                "XGBoost": (0.9936, 0.0017)}.items():
        chk(f"auc {modelo}", cv.loc[modelo, "auc_cv"], auc); chk(f"desv {modelo}", cv.loc[modelo, "auc_cv_desv"], desv)
    for modelo, auc in {"AdaBoost": 0.9927, "Random Forest": 0.9899, "SVM (kernel RBF)": 0.9753,
                        "Red neuronal (MLP)": 0.9729, "Regresión logística": 0.9510}.items():
        chk(f"auc {modelo}", cv.loc[modelo, "auc_cv"], auc)
    par = tablas["par"]
    chk("GB pareada", par.loc["Gradient Boosting", "diferencia_media"], -0.0007)
    chk("GB pareada desv", par.loc["Gradient Boosting", "diferencia_desv"], 0.0011)
    assert par.loc["Gradient Boosting", "pliegues_por_debajo_del_mejor"] == 4
    chk("XGB pareada", par.loc["XGBoost", "diferencia_media"], -0.0006)
    chk("XGB pareada desv", par.loc["XGBoost", "diferencia_desv"], 0.0006)
    assert par.loc["XGBoost", "pliegues_por_debajo_del_mejor"] == 4
    assert (par.iloc[3:]["pliegues_por_debajo_del_mejor"] == 5).all()
    chk("LR vs LGBM", par.loc["Regresión logística", "diferencia_media"], -0.0431)
    mejor = cv.iloc[0]
    assert cv.index[0] == "LightGBM"
    finalistas = cv.index[cv["auc_cv"] >= mejor["auc_cv"] - mejor["auc_cv_desv"]].tolist()
    assert finalistas == ["LightGBM", "XGBoost", "Gradient Boosting"]
    chk("t LGBM", cv.loc["LightGBM", "tiempo_ajuste_s"], 4.83, 0.005)
    chk("t XGB", cv.loc["XGBoost", "tiempo_ajuste_s"], 3.25, 0.005)
    chk("t GB", cv.loc["Gradient Boosting", "tiempo_ajuste_s"], 22.58, 0.005)
    for modelo in finalistas:
        chk(f"train {modelo}", cv.loc[modelo, "auc_train"], 1.0000)
    chk("brecha mín fin", cv.loc[finalistas, "brecha_train_cv"].min(), 0.0059)
    chk("brecha máx fin", cv.loc[finalistas, "brecha_train_cv"].max(), 0.0065)
    assert (cv["estrategia"] == "ninguna").all()

    # 4.6
    pyc = tablas["pyc"]
    assert list(pyc.index[:5]) == ["lightgbm", "xgboost", "gbc", "rf", "ada"]
    for clave, auc in {"lightgbm": 0.9931, "xgboost": 0.9922, "gbc": 0.9892, "rf": 0.9847,
                       "ada": 0.9829, "lr": 0.9375, "et": 0.9712, "dummy": 0.5000}.items():
        chk(f"pycaret {clave}", pyc.loc[clave, "AUC"], auc)
    chk("pycaret dummy acc", pyc.loc["dummy", "Accuracy"], 0.8394)
    fuera_catalogo = pyc.drop(index=["lightgbm", "xgboost", "gbc", "rf", "ada", "lr", "dummy"])
    assert fuera_catalogo["AUC"].idxmax() == "et"

    cifras.verificar()


def test_tablas_corresponden_a_los_modelos_en_cache(tablas):
    """La tabla final de CV debe salir exactamente de los modelos guardados en la caché."""
    rutas = {}
    for prefijo in ("ronda_3", "ronda_2", "ronda_1"):
        for nombre in tablas["cv"].index:
            ruta = comparacion._archivo_cache(config.RUTA_EXPERIMENTOS, prefijo, nombre)
            if nombre not in rutas and ruta.exists():
                rutas[nombre] = ruta
    if len(rutas) != len(tablas["cv"]):
        pytest.skip("No está completa la caché de modelos")
    for nombre, ruta in rutas.items():
        resumen = joblib.load(ruta)["modelo"].resumen()
        assert np.isclose(resumen["auc_cv"], tablas["cv"].loc[nombre, "auc_cv"], atol=5e-6), nombre
        assert np.isclose(resumen["auc_train"], tablas["cv"].loc[nombre, "auc_train"], atol=5e-6), nombre
