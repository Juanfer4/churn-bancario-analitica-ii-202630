"""Pruebas de la lectura de negocio con casos calculados a mano."""
import numpy as np
import pytest

from churn import negocio


@pytest.fixture
def caso():
    # 10 clientes, 3 churners. Orden por riesgo: c0 (1), c1 (0), c2 (1), c3 (0), c4 (1), resto 0
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    return y, p


def test_escenarios_a_mano(caso):
    y, p = caso
    t = negocio.tabla_escenarios(y, p, capacidades=(0.2, 0.5))
    assert t.loc[0.2, "contactados"] == 2 and t.loc[0.2, "churners_capturados"] == 1
    assert t.loc[0.2, "precision"] == pytest.approx(0.5) and t.loc[0.2, "captura"] == pytest.approx(1 / 3)
    assert t.loc[0.2, "lift"] == pytest.approx(0.5 / 0.3)
    assert t.loc[0.2, "umbral_implicito"] == pytest.approx(0.8)
    assert t.loc[0.5, "churners_capturados"] == 3 and t.loc[0.5, "beneficios_a_no_churners"] == 2
    assert t.loc[0.5, "churners_sin_contactar"] == 0 and t.loc[0.5, "captura_maxima_posible"] == 1.0


def test_orden_no_depende_del_orden_de_las_filas(caso):
    y, p = caso
    perm = np.random.default_rng(0).permutation(len(y))
    a = negocio.tabla_escenarios(y, p, (0.2, 0.5))
    b = negocio.tabla_escenarios(y[perm], p[perm], (0.2, 0.5))
    assert a.equals(b)


def test_incremental_y_precision_marginal(caso):
    y, p = caso
    inc = negocio.tabla_incremental(negocio.tabla_escenarios(y, p, (0.2, 0.5)))
    assert inc.loc[0.2, "contactos_adicionales"] == 2 and inc.loc[0.2, "churners_adicionales"] == 1
    assert inc.loc[0.5, "contactos_adicionales"] == 3 and inc.loc[0.5, "churners_adicionales"] == 2
    assert inc.loc[0.5, "precision_marginal"] == pytest.approx(2 / 3)


def test_razon_equilibrio_y_cortes_por_tramo(caso):
    y, p = caso
    assert negocio.razon_equilibrio(0.5, 0.5) == pytest.approx(4.0)
    assert negocio.razon_equilibrio(0.0, 0.5) == np.inf
    with pytest.raises(ValueError):
        negocio.razon_equilibrio(0.5, 0)
    incremental = negocio.tabla_incremental(negocio.tabla_escenarios(y, p, (0.2, 0.5)))
    cortes = negocio.tabla_cortes(incremental, tasa_exito_ejemplo=0.5)
    # El corte de cada tramo es 1 / precisión marginal; con s = 0,5 el valor mínimo de r es el doble
    assert cortes.loc[0.5, "sr_minimo"] == pytest.approx(1 / incremental.loc[0.5, "precision_marginal"])
    assert cortes["r_minimo_ejemplo"].tolist() == pytest.approx((2 * cortes["sr_minimo"]).tolist())
    # Justo en el corte, los contactos del tramo ni ganan ni pierden: TP * (s * r) - n = 0
    fila = cortes.loc[0.5]
    assert fila["churners_adicionales"] * fila["sr_minimo"] - fila["contactos_adicionales"] == pytest.approx(0.0)


def test_deciles_suman_y_captura_llega_a_1():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    p = rng.uniform(size=1000)
    d = negocio.tabla_deciles(y, p)
    assert d["clientes"].sum() == 1000 and d["churners"].sum() == y.sum()
    assert d["captura_acumulada"].iloc[-1] == pytest.approx(1.0)
    assert (d["prob_min"].values[:-1] >= d["prob_max"].values[1:]).all()  # deciles ordenados por riesgo


def test_curva_ganancia_modelo_perfecto():
    y = np.array([0] * 8 + [1] * 2)
    p = y.astype(float)
    curva = negocio.curva_ganancia(y, p)
    assert curva.loc[2, "captura"] == 1.0 and curva.iloc[-1]["fraccion_contactada"] == 1.0


def test_contactados_redondeo_y_limites():
    assert negocio.contactados(2026, 0.1) == 203 and negocio.contactados(2026, 0.2) == 405
    assert negocio.contactados(2026, 0.3) == 608 and negocio.contactados(10127, 0.2) == 2025
    with pytest.raises(ValueError):
        negocio.contactados(100, 0)


def test_bootstrap_escenarios_contiene_el_valor_observado():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.4 * y + rng.uniform(0, 0.6, 500), 0, 1)
    ic = negocio.bootstrap_escenarios(y, p, (0.2,), n_remuestras=300, semilla=1)
    t = negocio.tabla_escenarios(y, p, (0.2,))
    assert ic.loc[0.2, "precision_ic95_inf"] <= t.loc[0.2, "precision"] <= ic.loc[0.2, "precision_ic95_sup"]
    assert ic.loc[0.2, "captura_ic95_inf"] <= t.loc[0.2, "captura"] <= ic.loc[0.2, "captura_ic95_sup"]


def test_escenarios_por_modelo_compara_con_el_top_k(caso):
    y, p = caso
    tabla = negocio.escenarios_por_modelo(y, {"bueno": p, "invertido": 1 - p}, capacidades=(0.2,))
    assert tabla.loc["bueno", "churners_20"] == 1 and tabla.loc["invertido", "churners_20"] == 0
    assert tabla.loc["bueno", "captura_20"] == pytest.approx(1 / 3)
