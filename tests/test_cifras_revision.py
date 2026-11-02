"""Cifras citadas en el texto que agregó la revisión del 2026-11-01: captura por modelo (5.2),
calibración por rangos (5.3), trato por sexo y por edad (5.7) y semáforos de monitoreo (6.3). Solo se fijan las de LightGBM
y la logística, que son iguales en Windows y Linux; las de XGBoost y Gradient Boosting se fijan
con la ejecución de referencia en Windows. Si las tablas no existen, la prueba se omite.
"""
import pandas as pd
import pytest

from churn import config


def _leer(nombre, **kwargs):
    ruta = config.RUTA_TABLAS / f"{nombre}.csv"
    if not ruta.exists():
        pytest.skip(f"Ejecute el notebook para generar {ruta.name}")
    return pd.read_csv(ruta, **kwargs)


def test_captura_por_modelo_5_2():
    t = _leer("test_captura_por_modelo", header=[0, 1], index_col=0)
    cv, test = t[("train fuera de pliegue", "churners_20")], t[("test", "churners_20")]
    assert (cv["LightGBM"], test["LightGBM"]) == (1269, 315)
    assert (cv["Regresión logística"], test["Regresión logística"]) == (1076, 263)


def test_calibracion_por_rangos_5_3():
    t = _leer("test_calibracion_rangos_lightgbm", index_col=0)
    assert t.loc["5% a 20%", "clientes"] == 33
    assert round(100 * t.loc["5% a 20%", "prob_media"], 2) == 10.11 and round(100 * t.loc["5% a 20%", "tasa_real"], 2) == 24.24
    assert t.loc["95% a 100%", "clientes"] == 254
    assert round(100 * t.loc["95% a 100%", "prob_media"], 2) == 99.60 and round(100 * t.loc["95% a 100%", "tasa_real"], 2) == 97.24
    intermedios = t.loc[["5% a 20%", "20% a 40%", "40% a 60%", "60% a 80%"]]
    assert intermedios["clientes"].sum() == 76 and (intermedios["diferencia"] < 0).all()


def test_brecha_captura_por_sexo_5_7():
    fila = _leer("explicabilidad_brecha_captura_sexo", index_col=0).iloc[0]
    assert (fila["churners_F"], fila["churners_M"]) == (749, 553)
    assert round(100 * fila["brecha_captura"], 2) == 2.51
    assert (round(100 * fila["ic95_inf"], 2), round(100 * fila["ic95_sup"], 2)) == (0.79, 4.37)


def test_semaforos_de_monitoreo_6_3():
    t = _leer("monitoreo_semaforos", index_col=0)
    assert list(t["estado"]) == ["verde", "rojo", "verde"]
    assert list(t["estado puntaje"]) == ["verde", "amarillo", "rojo"]
    assert list(t["estado captura"]) == ["verde", "verde", "rojo"]
    assert t.loc["montos +20 %", "variable con más PSI"] == "total_trans_amt"
    assert round(t.loc["sin cambios", "PSI puntaje"], 4) == 0.0067
    assert round(t.loc["montos +20 %", "PSI variable"], 4) == 0.4676
    assert round(t.loc["montos +20 %", "PSI puntaje"], 4) == 0.1729
    assert round(t.loc["actividad barajada", "PSI puntaje"], 4) == 0.2774
    assert round(100 * t.loc["actividad barajada", "captura 20 %"], 2) == 43.93


def test_calidad_del_orden_por_sexo_y_edad_5_7():
    sexo = _leer("explicabilidad_equidad_sexo", index_col=[0, 1]).loc["completo"]
    if "auc_en_grupo" not in sexo.columns:
        pytest.skip("La tabla de equidad es de una ejecución anterior: vuelva a ejecutar el notebook")
    assert (round(sexo.loc["F", "auc_en_grupo"], 4), round(sexo.loc["M", "auc_en_grupo"], 4)) == (0.9963, 0.9908)
    assert (round(100 * sexo.loc["F", "falsas_alarmas_en_grupo"], 2), round(100 * sexo.loc["M", "falsas_alarmas_en_grupo"], 2)) == (4.31, 6.07)
    edad = _leer("explicabilidad_equidad_edad", index_col=[0, 1]).loc["completo"]
    assert len(edad) == 4
    assert (round(edad["auc_en_grupo"].min(), 4), round(edad["auc_en_grupo"].max(), 4)) == (0.9921, 0.9943)
    assert (round(100 * edad["captura_en_grupo"].min(), 2), round(100 * edad["captura_en_grupo"].max(), 2)) == (97.30, 97.78)
    assert (round(edad["razon_contacto_riesgo"].min(), 2), round(edad["razon_contacto_riesgo"].max(), 2)) == (1.22, 1.29)
