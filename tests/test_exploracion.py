"""Pruebas de las estadísticas del EDA con casos de resultado conocido."""
import numpy as np
import pandas as pd
import pytest

from churn import exploracion


def test_wilson_contiene_la_proporcion_y_no_sale_de_0_1():
    inf, sup = exploracion.intervalo_wilson(np.array([0, 5, 42]), np.array([10, 10, 42]))
    assert np.all(inf >= 0) and np.all(sup <= 1)
    assert inf[1] < 0.5 < sup[1]
    assert sup[2] == pytest.approx(1.0) and inf[2] < 1.0  # 42 de 42 no da un intervalo degenerado


def test_wilson_valor_de_referencia():
    # Valor de referencia de Wilson para 5 de 10 con z = 1,96: [0,2366; 0,7634]
    inf, sup = exploracion.intervalo_wilson(5, 10)
    assert float(inf) == pytest.approx(0.2366, abs=1e-4)
    assert float(sup) == pytest.approx(0.7634, abs=1e-4)


def test_auc_univariada_y_sentido():
    y = pd.Series([0, 0, 0, 1, 1, 1])
    X = pd.DataFrame({"sube": [1, 2, 3, 4, 5, 6], "baja": [6, 5, 4, 3, 2, 1], "ruido": [1, 2, 1, 2, 1, 2]})
    r = exploracion.resumen_numericas(X, y)
    assert r.loc["sube", "auc_univariada"] == pytest.approx(1.0)
    assert r.loc["baja", "auc_univariada"] == pytest.approx(1.0)
    assert r.loc["sube", "sentido"] == "más alto, más churn"
    assert r.loc["baja", "sentido"] == "más alto, menos churn"
    assert r.loc["ruido", "auc_univariada"] < 0.7


def test_tasa_por_nivel_cuadra_con_los_conteos():
    X = pd.DataFrame({"nivel": ["a", "a", "b", "b", "b"]})
    y = pd.Series([1, 0, 1, 1, 0])
    t = exploracion.tasa_por_nivel(X, y, "nivel")
    assert t.loc["a", "clientes"] == 2 and t.loc["b", "churners"] == 2
    assert t.loc["b", "tasa_churn"] == pytest.approx(2 / 3)
    assert t["clientes"].sum() == len(X)


def test_cramer_v_asociacion_total_y_nula():
    y = pd.Series([0, 1] * 50)
    X = pd.DataFrame({"igual": y.map({0: "no", 1: "si"}), "independiente": ["x"] * 50 + ["z"] * 50})
    r = exploracion.asociacion_categoricas(X, y)
    assert r.loc["igual", "v_cramer"] > 0.95
    assert r.loc["independiente", "v_cramer"] < 0.05


def test_pares_correlacionados_detecta_identidad():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "doble": 2 * a, "otra": rng.normal(size=200)})
    pares = exploracion.pares_correlacionados(X, umbral=0.9)
    assert len(pares) == 1
    assert set(pares.loc[0, ["variable_1", "variable_2"]]) == {"a", "doble"}


def test_tasa_por_nivel_respeta_orden_y_rechaza_orden_incompleto():
    X = pd.DataFrame({"nivel": ["a", "b", "c", "a"]})
    y = pd.Series([1, 0, 0, 0])
    t = exploracion.tasa_por_nivel(X, y, "nivel", orden=["c", "a", "b"])
    assert list(t.index) == ["c", "a", "b"]
    with pytest.raises(ValueError):
        exploracion.tasa_por_nivel(X, y, "nivel", orden=["a", "b"])


def test_tramos_cubre_todo_y_rechaza_valores_fuera():
    s = pd.Series([0, 5, 10, 2517], name="saldo")
    t = exploracion.tramos(s, [-0.5, 0.5, 2516.5, 2517.5], ["cero", "medio", "tope"])
    assert list(t.astype(str)) == ["cero", "medio", "medio", "tope"]
    with pytest.raises(ValueError):
        exploracion.tramos(pd.Series([3000], name="saldo"), [-0.5, 0.5, 2517.5], ["a", "b"])
