"""Pruebas de las funciones que antes vivían en celdas del notebook (cálculos pequeños, datos sintéticos)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from churn import comparacion, datos, evaluacion, explicabilidad, features, negocio, produccion


@pytest.fixture(scope="module")
def sintetico():
    rng = np.random.default_rng(0)
    y = pd.Series((rng.random(400) < 0.2).astype(int))
    prob = np.clip(0.2 * y + 0.8 * rng.random(400) * 0.6, 0, 1)
    return y, prob


def test_resumen_preparacion_cuenta_columnas_y_saldo_maximo():
    limpio = datos.limpiar_datos(datos.cargar_datos())
    p = datos.particionar(limpio)
    prep = features.construir_preparacion(escalar=False, ticket_promedio=True, indicadores_saldo=True)
    prep.fit(p.X_train, p.y_train)
    r = features.resumen_preparacion(prep)
    assert r["columnas_por_origen"]["columnas"].sum() == len(features.nombres_salida(prep))
    assert r["saldo_maximo"] == p.X_train["total_revolving_bal"].max()


def test_oof_resumen_y_comparacion_cv_test(sintetico):
    y, _ = sintetico
    X = pd.DataFrame({"a": np.random.default_rng(1).normal(size=len(y)) + y})
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    oof = evaluacion.oof_de_modelos({"lr": LogisticRegression()}, X, y, cv, n_jobs=1)
    assert oof["lr"].shape == (len(y),)
    resumen = evaluacion.resumen_oof(oof, y, {"lr": 0.8})
    assert list(resumen.index) == ["lr"] and resumen.loc["lr", "auc_cv_medio"] == 0.8
    con_05 = evaluacion.metricas_en_test(y, oof)
    assert list(con_05.index) == ["lr"] and con_05.loc["lr", "umbral"] == 0.5
    comp = evaluacion.comparar_cv_test(resumen, con_05)
    assert np.isclose(comp.loc["lr", "diferencia_test_menos_cv"], comp.loc["lr", "auc_test"] - 0.8)


def test_lectura_test_reune_las_cuatro_tablas(sintetico):
    y, prob = sintetico
    lectura = negocio.lectura_test(y, prob, n_remuestras=50)
    assert set(lectura) == {"escenarios", "incremental", "intervalos", "deciles"}
    pd.testing.assert_frame_equal(lectura["escenarios"], negocio.tabla_escenarios(y, prob))


def test_tabla_importancia_rankings():
    perm = pd.DataFrame({"importancia": [0.3, 0.2, 0.1]}, index=["a", "b", "c"])
    gan = pd.Series([10.0, 50.0, 40.0], index=["a", "b", "c"], name="importancia_ganancia_pct")
    t = explicabilidad.tabla_importancia(perm, gan)
    assert list(t["ranking_permutacion"]) == [1, 2, 3]
    assert list(t["ranking_ganancia"]) == [3, 1, 2]


def test_resumir_tramos_descarta_celdas_pequenas():
    conj = pd.DataFrame({"total_trans_amt": [1000, 2000, 4000, 12000], "prob_churn_media": [.9, .8, .3, .1],
                         "clientes_en_celda": [20, 5, 30, 40]})
    con_datos, resumen = explicabilidad.resumir_tramos(conj)
    assert len(con_datos) == 3 and resumen.loc["hasta 3.000", "celdas"] == 1
    assert resumen["clientes"].sum() == 90


def test_agregar_escenarios_agrega_columnas(sintetico):
    y, prob = sintetico
    sens = pd.DataFrame(index=["v"])
    out = explicabilidad.agregar_escenarios(sens, y, {"v": prob})
    esc = negocio.tabla_escenarios(y, prob)
    assert out.loc["v", "captura_20"] == esc.loc[0.2, "captura"]
    assert out.loc["v", "churners_10"] == esc.loc[0.1, "churners_capturados"]


def test_parametros_comparables_quita_direcciones():
    assert produccion.parametros_comparables({"f": object()})["f"].count(" at 0x") == 0


def test_tablas_y_decision_de_candidatas():
    class M:
        def __init__(self, n): self.nombre, self.ticket_promedio, self.indicadores_saldo = n, False, False
    ms = [M("a"), M("b")]
    combs = list(comparacion.COMBINACIONES_CANDIDATAS)
    filas = [(m.nombre, c, 0.01, 3) for m in ms for c in combs]
    tabla = pd.DataFrame(filas, columns=["modelo", "combinacion", "mejora_media", "pliegues_que_mejoran"])
    mejora, pliegues = comparacion.tablas_candidatas(tabla, ms)
    assert list(mejora.columns) == combs and list(pliegues.index) == ["a", "b"]
    dec = comparacion.aplicar_decision_candidatas(ms, {"a": (True, False), "b": (False, True)})
    assert ms[0].ticket_promedio and ms[1].indicadores_saldo and dec.loc["a", "ticket_promedio"]
