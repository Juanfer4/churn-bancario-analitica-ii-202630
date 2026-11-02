"""Pruebas de la Fase 5, 5.6 a 5.8 (explicabilidad) con datos sintéticos pequeños y rápidos."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from churn import datos, explicabilidad, features
from imblearn.pipeline import Pipeline


def _datos_sinteticos(n=400, semilla=0):
    """Clientes con las columnas que exige el pipeline; solo total_trans_ct explica el churn."""
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame({
        "total_trans_ct": rng.integers(10, 120, n),
        "total_trans_amt": rng.uniform(500, 15000, n),
        "total_revolving_bal": rng.integers(0, 2517, n),
        "avg_open_to_buy": rng.uniform(0, 30000, n),
        "ruido": rng.normal(size=n),
    })
    for col in datos.COLUMNAS_CATEGORICAS:
        X[col] = rng.choice(["a", "b"], n)
    y = pd.Series((X["total_trans_ct"] < 50).astype(int), name="churn")
    return X, y


def _pipeline():
    return Pipeline(features.pasos_preparacion(escalar=False) + [
        ("balanceo", "passthrough"), ("modelo", DecisionTreeClassifier(max_depth=3, random_state=0))])


@pytest.fixture(scope="module")
def caso():
    X, y = _datos_sinteticos()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    return X, y, cv, _pipeline().fit(X, y)


def test_variable_original_agrupa_one_hot():
    assert explicabilidad._variable_original("gender_M") == "gender"
    assert explicabilidad._variable_original("income_category_infrequent_sklearn") == "income_category"
    assert explicabilidad._variable_original("total_trans_ct") == "total_trans_ct"


def test_importancia_ganancia_suma_100_y_agrupa(caso):
    *_, ajustado = caso
    imp = explicabilidad.importancia_ganancia(ajustado)
    assert imp.sum() == pytest.approx(100)
    assert imp.index[0] == "total_trans_ct"
    assert "gender" in imp.index and not any(i.startswith("gender_") for i in imp.index)
    assert "avg_open_to_buy" not in imp.index  # el pipeline la elimina


def test_permutacion_detecta_la_variable_informativa(caso):
    X, y, cv, _ = caso
    tabla = explicabilidad.importancia_permutacion_cv(_pipeline(), X, y, cv, n_repeticiones=2, n_jobs=1)
    assert tabla.index[0] == "total_trans_ct"
    assert tabla.loc["total_trans_ct", "caida_auc_media"] > 0.3
    assert tabla.loc["total_trans_ct", "pliegues_con_caida"] == 3
    assert "avg_open_to_buy" not in tabla.index
    assert tabla.loc["ruido", "caida_auc_media"] == pytest.approx(0, abs=0.05)


def test_neutralizar():
    X, _ = _datos_sinteticos(50)
    sin_ruido = explicabilidad.neutralizar(X, "ruido")
    assert "ruido" not in sin_ruido.columns and "ruido" in X.columns  # no modifica el original
    sin_sexo = explicabilidad.neutralizar(X, "gender")
    assert sin_sexo["gender"].nunique() == 1 and X["gender"].nunique() == 2
    with pytest.raises(ValueError):
        explicabilidad.neutralizar(X, "total_trans_ct")  # la necesita CreadorVariables
    with pytest.raises(ValueError):
        explicabilidad.neutralizar(X, "no_existe")


def test_sensibilidad_pareada(caso):
    X, y, cv, _ = caso
    tabla, oof = explicabilidad.sensibilidad_variables(_pipeline(), X, y, cv, {"sin ruido": ["ruido"], "sin sexo": ["gender"]}, n_jobs=1)
    assert list(tabla.index) == ["completo", "sin ruido", "sin sexo"]
    assert tabla.loc["completo", "diferencia_vs_completo"] == 0 and tabla.loc["completo", "pliegues_por_debajo"] == 0
    assert set(oof) == {"completo", "sin ruido", "sin sexo"} and all(len(v) == len(y) for v in oof.values())
    # la diferencia media es exactamente la resta de los AUC medios (mismos pliegues)
    for v in ("sin ruido", "sin sexo"):
        assert tabla.loc[v, "diferencia_vs_completo"] == pytest.approx(tabla.loc[v, "auc_cv"] - tabla.loc["completo", "auc_cv"])


def test_equidad_por_grupo_cuadra_con_el_escenario():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.2, 0.3, 0.05, 0.01, 0.4])
    grupo = pd.Series(list("FFFFFMMMMM"))
    tabla = explicabilidad.equidad_por_grupo(y, p, grupo, capacidad=0.3)  # contacta 3: índices 0, 1, 2
    assert tabla["clientes"].sum() == 10
    assert tabla.loc["F", "pct_contactado"] == pytest.approx(3 / 5) and tabla.loc["M", "pct_contactado"] == 0
    assert tabla.loc["F", "captura_en_grupo"] == pytest.approx(2 / 3)
    assert tabla.loc["F", "precision_en_grupo"] == pytest.approx(2 / 3)
    assert np.isnan(tabla.loc["M", "captura_en_grupo"]) and np.isnan(tabla.loc["M", "precision_en_grupo"])
    assert tabla.loc["F", "falsas_alarmas_en_grupo"] == pytest.approx(1 / 2)  # 1 de los 2 que no se iban
    assert tabla.loc["M", "falsas_alarmas_en_grupo"] == 0
    assert tabla.loc["F", "auc_en_grupo"] == pytest.approx(5 / 6)  # 5 de 6 pares bien ordenados
    assert np.isnan(tabla.loc["M", "auc_en_grupo"])  # sin churners no hay AUC


def test_seleccion_en_grupo():
    y = np.array([1, 1, 0, 0, 1])
    probs = {"a": np.array([0.9, 0.1, 0.8, 0.2, 0.7]), "b": np.array([0.1, 0.9, 0.2, 0.3, 0.4])}
    tabla = explicabilidad.seleccion_en_grupo(y, probs, [True, True, False, False, False], capacidad=0.4)  # 2 contactos
    assert tabla.loc["a", "dentro_del_escenario"] == 1 and tabla.loc["b", "dentro_del_escenario"] == 1
    assert tabla.loc["a", "churners"] == 2 and tabla.loc["a", "prob_minima"] == pytest.approx(0.1)


def test_dependencia_parcial_y_conjunta(caso):
    X, _, _, ajustado = caso
    curvas = explicabilidad.dependencia_parcial(ajustado, X, ["total_trans_ct"], puntos=10)
    curva = curvas["total_trans_ct"]
    assert len(curva) == 10 and curva["valor"].is_monotonic_increasing
    assert curva["prob_churn_media"].iloc[0] > curva["prob_churn_media"].iloc[-1]  # menos transacciones, más churn
    conjunta = explicabilidad.dependencia_parcial_2d(ajustado, X, "total_trans_amt", "total_trans_ct", puntos=5, muestra=100)
    assert len(conjunta) == 25
    assert conjunta["prob_churn_media"].between(0, 1).all()
    assert conjunta["clientes_en_celda"].sum() <= len(X)


def test_bordes_rodean_cada_valor():
    bordes = explicabilidad._bordes(np.array([1.0, 2.0, 4.0]))
    assert np.allclose(bordes, [0.5, 1.5, 3.0, 5.0])


def test_ic_brecha_captura_contiene_la_observada_y_detecta_una_brecha_clara():
    rng = np.random.default_rng(5)
    n = 2000
    grupo = pd.Series(np.where(rng.random(n) < 0.5, "F", "M"))
    y = (rng.random(n) < 0.2).astype(int)
    igual = y + rng.random(n) * 0.5                      # mismo trato en los dos grupos
    t = explicabilidad.ic_brecha_captura(y, igual, grupo, capacidad=0.2, n_remuestras=500)
    fila = t.iloc[0]
    assert fila["ic95_inf"] <= fila["brecha_captura"] <= fila["ic95_sup"] and fila["incluye_cero"]
    sesgo = y * np.where(grupo == "F", 1.0, 0.1) + rng.random(n) * 0.5   # a los hombres casi no se les ve
    t2 = explicabilidad.ic_brecha_captura(y, sesgo, grupo, capacidad=0.2, n_remuestras=500)
    assert not t2.iloc[0]["incluye_cero"] and t2.iloc[0]["brecha_captura"] > 0
    with pytest.raises(ValueError):
        explicabilidad.ic_brecha_captura(y, igual, pd.Series(["a", "b", "c"] * (n // 3) + ["a"] * (n % 3)))
