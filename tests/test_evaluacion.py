"""Pruebas de las métricas (con matrices calculadas a mano), del umbral, del bootstrap y de la calibración."""
import numpy as np
import pandas as pd
import pytest

from churn import evaluacion


def test_metricas_con_caso_conocido():
    # 4 churners y 6 que se quedan; con umbral 0,5: tp=3, fn=1, fp=1, tn=5
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
    m = evaluacion.calcular_metricas(y, p, umbral=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["sensibilidad"] == pytest.approx(3 / 4)
    assert m["especificidad"] == pytest.approx(5 / 6)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["vpn"] == pytest.approx(5 / 6)
    assert m["accuracy"] == pytest.approx(8 / 10)
    assert m["f1"] == pytest.approx(0.75)
    # AUC a mano: de 24 pares churner-no churner, 20,5 bien ordenados (6 + 6 + 6 + 2 y un empate a 0,2)
    assert m["auc"] == pytest.approx(20.5 / 24)


def test_sensibilidad_y_especificidad_no_se_invierten():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])  # detecta a todos los churners, 1 falsa alarma
    m = evaluacion.calcular_metricas(y, p)
    assert m["sensibilidad"] == 1.0
    assert m["especificidad"] == pytest.approx(3 / 4)


def test_umbral_cambia_la_clasificacion_pero_no_el_ranking():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0, 1)
    a = evaluacion.calcular_metricas(y, p, umbral=0.3)
    b = evaluacion.calcular_metricas(y, p, umbral=0.7)
    assert a["auc"] == b["auc"] and a["ap"] == b["ap"]
    assert a["sensibilidad"] >= b["sensibilidad"]


def test_rechaza_entradas_invalidas():
    with pytest.raises(ValueError):
        evaluacion.calcular_metricas([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        evaluacion.calcular_metricas([0, 1], [0.1, 1.2])


def test_sin_predicciones_positivas_la_precision_queda_indefinida():
    m = evaluacion.calcular_metricas([0, 1, 0], [0.1, 0.2, 0.3], umbral=0.9)
    assert m["tp"] == 0 and m["fp"] == 0
    assert np.isnan(m["precision"]) and m["sensibilidad"] == 0.0


def test_bootstrap_modelo_identico_es_equivalente_y_reproducible():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p_bueno = np.clip(0.5 * y + rng.uniform(0, 0.5, 300), 0, 1)
    p_azar = rng.uniform(0, 1, 300)
    probs = {"bueno": p_bueno, "copia": p_bueno.copy(), "azar": p_azar}
    t1 = evaluacion.bootstrap_auc(y, probs, referencia="bueno", n_remuestras=300)
    t2 = evaluacion.bootstrap_auc(y, probs, referencia="bueno", n_remuestras=300)
    pd.testing.assert_frame_equal(t1, t2)
    assert t1.loc["copia", "dif_vs_referencia"] == 0 and bool(t1.loc["copia", "equivalente_a_referencia"])
    assert not bool(t1.loc["azar", "equivalente_a_referencia"]) and t1.loc["azar", "dif_ic95_sup"] < 0
    for nombre in probs:
        assert t1.loc[nombre, "auc_ic95_inf"] <= t1.loc[nombre, "auc_test"] <= t1.loc[nombre, "auc_ic95_sup"]


def test_calibracion_de_un_modelo_bien_calibrado_y_uno_sesgado():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # probabilidades verdaderas
    tabla = evaluacion.tabla_calibracion(y, p)
    assert tabla["clientes"].sum() == 20000 and len(tabla) == 10
    assert evaluacion.error_calibracion(tabla) < 0.02
    sesgada = evaluacion.tabla_calibracion(y, p ** 3)  # subestima el riesgo
    assert evaluacion.error_calibracion(sesgada) > 0.1
