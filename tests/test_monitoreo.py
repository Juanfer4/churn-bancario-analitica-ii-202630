"""Pruebas de `monitoreo`: el PSI y los semáforos detectan cambios y no dan falsas alarmas."""
import numpy as np
import pandas as pd
import pytest

from churn import config, monitoreo


@pytest.fixture
def referencia():
    return pd.Series(np.random.default_rng(0).normal(100, 15, 5000))


def test_psi_es_cero_si_nada_cambia(referencia):
    assert monitoreo.psi(referencia, referencia) == pytest.approx(0, abs=1e-9)


def test_psi_no_da_alarma_con_otra_muestra_de_la_misma_distribucion(referencia):
    otra = pd.Series(np.random.default_rng(1).normal(100, 15, 5000))
    assert monitoreo.psi(referencia, otra) < config.UMBRALES_MONITOREO["psi"][0]


def test_psi_detecta_un_desplazamiento_grande(referencia):
    desplazada = referencia + 15  # una desviación estándar
    assert monitoreo.psi(referencia, desplazada) > config.UMBRALES_MONITOREO["psi"][1]


def test_psi_con_categorias_incluidas_las_nuevas():
    ref = pd.Series(["A"] * 80 + ["B"] * 20)
    assert monitoreo.psi(ref, ref) == pytest.approx(0, abs=1e-9)
    assert monitoreo.psi(ref, pd.Series(["A"] * 20 + ["B"] * 20 + ["C"] * 60)) > 0.25


def test_estado_indicador_respeta_los_umbrales():
    assert [monitoreo.estado_indicador(v, 0.1, 0.25) for v in (0.0, 0.1, 0.2499, 0.25, 1)] == [
        "verde", "amarillo", "amarillo", "rojo", "rojo"]


def test_psi_por_variable_ordena_y_marca_la_variable_movida(referencia):
    X = pd.DataFrame({"estable": referencia, "movida": referencia})
    nuevo = pd.DataFrame({"estable": referencia, "movida": referencia + 15})
    tabla = monitoreo.psi_por_variable(X, nuevo)
    assert list(tabla.index) == ["movida", "estable"]
    assert list(tabla["estado"]) == ["rojo", "verde"]


def test_captura_vs_base_distingue_un_modelo_sano_de_uno_degradado():
    rng = np.random.default_rng(3)
    y = (rng.random(4000) < 0.16).astype(int)
    buena = y * 0.6 + rng.random(4000) * 0.4          # separa casi perfecto
    mala = rng.random(4000)                            # al azar
    sana = monitoreo.captura_vs_base(y, buena, captura_base=monitoreo.captura_vs_base(y, buena, 1.0)["captura"])
    degradada = monitoreo.captura_vs_base(y, mala, captura_base=sana["captura"])
    assert sana["estado"] == "verde" and sana["caida"] == pytest.approx(0)
    assert degradada["estado"] == "rojo" and degradada["caida"] > 0.5


def test_psi_puntaje_verde_con_la_misma_distribucion_y_rojo_si_se_desplaza():
    rng = np.random.default_rng(4)
    ref = rng.beta(0.3, 1.5, 5000)                         # muchos puntajes bajos, como el modelo
    igual = monitoreo.psi_puntaje(ref, rng.beta(0.3, 1.5, 5000))
    movido = monitoreo.psi_puntaje(ref, rng.beta(1.5, 1.5, 5000))
    assert igual["estado"] == "verde" and igual["psi"] < 0.10
    assert movido["estado"] == "rojo" and movido["psi"] > 0.25


def test_tabla_semaforos_lee_los_tres_indicadores_y_deja_pendiente_la_captura_sin_churn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    y = pd.Series((X["a"] + 0.3 * rng.normal(size=3000) > 1).astype(int))
    modelo = LogisticRegression().fit(X, y)
    estable = X.sample(1000, random_state=0)
    lotes = {"estable": estable, "corrido": estable.assign(a=estable["a"] + 2)}
    base = monitoreo.captura_vs_base(y, modelo.predict_proba(X)[:, 1], 1.0)["captura"]
    tabla = monitoreo.tabla_semaforos(modelo, ["a", "b"], X, lotes, captura_base=base,
                                      y_lotes={"estable": y.loc[estable.index]})
    assert list(tabla.index) == ["estable", "corrido"]
    assert tabla.loc["estable", "estado"] == "verde" and tabla.loc["estable", "estado puntaje"] == "verde"
    assert tabla.loc["corrido", "estado"] == "rojo" and tabla.loc["corrido", "variable con más PSI"] == "a"
    assert tabla.loc["estable", "estado captura"] in ("verde", "amarillo", "rojo")
    assert tabla.loc["corrido", "estado captura"].startswith("pendiente")
