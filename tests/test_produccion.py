"""Pruebas del modelo viable (Fase 6): construcción, validación de entrada, puntuación,
guardado y reproducibilidad. Usan los datos reales de train; el test no se toca."""
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from churn import config, datos, features, negocio, produccion

CACHE_OFICIAL = config.RUTA_EXPERIMENTOS / "ronda_2_lightgbm.joblib"


@pytest.fixture(scope="module")
def particion():
    _, p, _ = datos.preparar_datos(guardar=False)
    return p


@pytest.fixture(scope="module")
def artefacto(particion):
    return produccion.crear_artefacto(
        produccion.construir_pipeline_produccion(), particion.X_train, particion.y_train,
        nombre="prueba", rol="candidato_produccion", evaluado_en_test=False,
        excluidas=produccion.VARIABLES_EXCLUIDAS_PRODUCCION, metricas={"origen": "prueba"})


@pytest.fixture(scope="module")
def clientes(particion):
    """Clientes con el formato del archivo del banco (clientnum como columna y el objetivo
    incluido, que debe ignorarse). Son 300 clientes de train."""
    crudo = datos.cargar_datos()
    return crudo[crudo[datos.COLUMNA_ID].isin(particion.X_train.index[:300])].reset_index(drop=True)


def _sin_direcciones(params):
    """repr de los parámetros sin las direcciones de memoria (Windows las escribe en mayúsculas)."""
    return {k: re.sub(r" at 0x[0-9a-fA-F]+", "", repr(v)) for k, v in params.items()}


# --- Construcción -----------------------------------------------------------------------
def test_hiperparametros_fijos_coinciden_con_la_busqueda():
    if not CACHE_OFICIAL.exists():
        pytest.skip("Sin la caché de la Fase 4 no hay contra qué comparar")
    buscado = joblib.load(CACHE_OFICIAL)["modelo"].busqueda_.best_estimator_
    assert _sin_direcciones(produccion.construir_pipeline_oficial().get_params()) == _sin_direcciones(buscado.get_params())


def test_columnas_de_entrada():
    oficial, prod = produccion.columnas_entrada(), produccion.columnas_entrada(produccion.VARIABLES_EXCLUIDAS_PRODUCCION)
    assert len(oficial) == 19 and len(prod) == 17
    assert set(oficial) - set(prod) == {"gender", "contacts_count_12_mon"}
    assert datos.COLUMNA_ID not in oficial and config.OBJETIVO not in oficial
    assert prod == [c for c in oficial if c in prod]  # mismo orden


def test_el_candidato_no_usa_las_variables_excluidas(artefacto):
    nombres = features.nombres_salida(artefacto["pipeline"])
    assert not any(n.startswith("gender") or n.startswith("contacts") for n in nombres)
    # mismos hiperparámetros que el oficial
    for clave, valor in produccion.HIPERPARAMETROS_OFICIALES.items():
        assert artefacto["hiperparametros"][clave] == valor


# --- Validación de entrada ----------------------------------------------------------------
def test_acepta_el_formato_del_banco_e_ignora_columnas_de_mas(clientes, artefacto):
    X, reporte = produccion.preparar_clientes(clientes, artefacto["columnas_entrada"], artefacto["rangos_train"])
    assert list(X.columns) == artefacto["columnas_entrada"] and X.index.name == datos.COLUMNA_ID
    assert set(reporte["columnas_ignoradas"]) == {config.OBJETIVO, "gender", "contacts_count_12_mon"}


@pytest.mark.parametrize("dano, mensaje", [
    (lambda df: df.drop(columns="total_trans_ct"), "Faltan columnas"),
    (lambda df: df.assign(total_trans_amt=df["total_trans_amt"].where(df.index != 3)), "vacíos"),
    (lambda df: df.assign(clientnum=df["clientnum"].where(df.index != 1, df["clientnum"].iloc[0])), "repetidos"),
    (lambda df: df.assign(credit_limit=df["credit_limit"].astype(str)), "numérica"),
    (lambda df: df.assign(avg_utilization_ratio=df["avg_utilization_ratio"].where(df.index != 2, 1.5)), "rangos de negocio"),
    (lambda df: df.assign(card_category=df["card_category"].where(df.index != 0, "Diamante")), "no vistas"),
    (lambda df: df.iloc[0:0], "No hay clientes"),
])
def test_detiene_datos_invalidos(clientes, artefacto, dano, mensaje):
    with pytest.raises(datos.ErrorDeDatos, match=mensaje):
        produccion.puntuar_clientes(dano(clientes), artefacto)


def test_no_es_dataframe(artefacto):
    with pytest.raises(TypeError):
        produccion.puntuar_clientes([[1, 2]], artefacto)


def test_categoria_desconocida_no_estricta_solo_avisa(clientes, artefacto):
    sucio = clientes.assign(card_category=clientes["card_category"].where(clientes.index != 0, "Diamante"))
    with pytest.warns(UserWarning, match="no vistas"):
        lista = produccion.puntuar_clientes(sucio, artefacto, estricto=False)
    assert len(lista) == len(clientes)


def test_avisa_valores_fuera_del_rango_de_train(clientes, artefacto):
    extremo = clientes.assign(total_trans_amt=clientes["total_trans_amt"].where(clientes.index != 0, 10 ** 6))
    with pytest.warns(UserWarning, match="fuera del rango de train"):
        lista = produccion.puntuar_clientes(extremo, artefacto)
    assert lista.attrs["fuera_de_rango_train"] == {"total_trans_amt": 1}


def test_textos_con_espacios_se_limpian_como_en_el_entrenamiento(clientes, artefacto):
    con_espacios = clientes.assign(card_category=" " + clientes["card_category"] + " ")
    pd.testing.assert_frame_equal(produccion.puntuar_clientes(con_espacios, artefacto),
                                  produccion.puntuar_clientes(clientes, artefacto))


# --- Puntuación ---------------------------------------------------------------------------
def test_lista_priorizada_y_consistente(clientes, artefacto):
    lista = produccion.puntuar_clientes(clientes, artefacto, capacidad=0.2)
    assert lista["prob_churn"].is_monotonic_decreasing
    assert list(lista["prioridad"]) == list(range(1, len(clientes) + 1))
    assert lista["contactar"].sum() == negocio.contactados(len(clientes), 0.2)
    assert lista["contactar"].iloc[:lista["contactar"].sum()].all()  # los contactados son los primeros
    assert lista.index.is_unique and set(lista.index) == set(clientes[datos.COLUMNA_ID])
    # mismas probabilidades que el pipeline aplicado directamente a las columnas del modelo
    X = clientes.set_index(datos.COLUMNA_ID)[artefacto["columnas_entrada"]]
    directa = pd.Series(artefacto["pipeline"].predict_proba(X)[:, 1], index=X.index)
    assert np.allclose(lista["prob_churn"], directa.loc[lista.index])
    # el orden de entrada no cambia la probabilidad de cada cliente ni la lista (sin empates)
    barajados = produccion.puntuar_clientes(clientes.sample(frac=1, random_state=7), artefacto, capacidad=0.2)
    assert lista["prob_churn"].is_unique
    pd.testing.assert_frame_equal(lista, barajados)


def test_empates_conservan_el_orden_de_entrada(clientes, artefacto):
    duplicado = pd.concat([clientes.iloc[[5]], clientes.iloc[[5]]]).reset_index(drop=True)
    duplicado[datos.COLUMNA_ID] = [111, 222]
    lista = produccion.puntuar_clientes(duplicado, artefacto)
    assert list(lista.index) == [111, 222]


def test_probabilidad_es_de_la_clase_attrited(particion, artefacto):
    pipeline = artefacto["pipeline"]
    assert produccion.indice_clase_positiva(pipeline) == 1
    crudo = datos.cargar_datos().set_index(datos.COLUMNA_ID).loc[particion.X_train.index]
    es_attrited = (crudo[config.OBJETIVO] == "Attrited Customer").astype(int)
    prob = pipeline.predict_proba(particion.X_train[artefacto["columnas_entrada"]])[:, 1]
    assert roc_auc_score(es_attrited, prob) > 0.95  # en train; con la clase invertida sería < 0,05


def test_indice_clase_positiva_no_supone_el_orden():
    class Falso:
        classes_ = np.array([1, 0])
    assert produccion.indice_clase_positiva(Falso()) == 0
    Falso.classes_ = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        produccion.indice_clase_positiva(Falso())


def test_capacidad_invalida(clientes, artefacto):
    with pytest.raises(ValueError):
        produccion.puntuar_clientes(clientes, artefacto, capacidad=1.5)


# --- Guardado, carga y reproducibilidad --------------------------------------------------
def test_guardar_y_cargar_da_las_mismas_predicciones(tmp_path, clientes, artefacto):
    ruta = produccion.guardar_modelo(artefacto, tmp_path / "modelo.joblib")
    cargado = produccion.cargar_modelo(ruta)
    pd.testing.assert_frame_equal(produccion.puntuar_clientes(clientes, cargado),
                                  produccion.puntuar_clientes(clientes, artefacto))


def test_artefactos_invalidos(tmp_path, artefacto):
    with pytest.raises(ValueError, match="incompleto"):
        produccion.guardar_modelo({k: v for k, v in artefacto.items() if k != "columnas_entrada"}, tmp_path / "a.joblib")
    with pytest.raises(ValueError, match="clase positiva"):
        produccion.guardar_modelo({**artefacto, "codigo_clase_positiva": 0}, tmp_path / "b.joblib")
    with pytest.raises(FileNotFoundError):
        produccion.cargar_modelo(tmp_path / "no_existe.joblib")
    otra_version = {**artefacto, "versiones": {**artefacto["versiones"], "lightgbm": "0.0.1"}}
    joblib.dump(otra_version, tmp_path / "c.joblib")
    with pytest.warns(UserWarning, match="Versiones distintas"):
        produccion.cargar_modelo(tmp_path / "c.joblib")


def test_reentrenar_desde_cero_reproduce_el_modelo(particion, clientes, artefacto):
    otro = produccion.crear_artefacto(
        produccion.construir_pipeline_produccion(), particion.X_train, particion.y_train,
        nombre="prueba", rol="candidato_produccion", evaluado_en_test=False,
        excluidas=produccion.VARIABLES_EXCLUIDAS_PRODUCCION)
    assert otro["huella_datos_train"] == artefacto["huella_datos_train"]
    pd.testing.assert_frame_equal(produccion.puntuar_clientes(clientes, otro), produccion.puntuar_clientes(clientes, artefacto))


def test_oficial_reconstruido_igual_al_evaluado_en_test(particion):
    if not CACHE_OFICIAL.exists():
        pytest.skip("Sin la caché de la Fase 4 no hay contra qué comparar")
    buscado = joblib.load(CACHE_OFICIAL)["modelo"].busqueda_.best_estimator_
    oficial = produccion.crear_artefacto(produccion.construir_pipeline_oficial(), particion.X_train, particion.y_train,
                                         nombre="oficial", rol="oficial", evaluado_en_test=True)
    muestra = particion.X_train.iloc[:500]
    assert np.array_equal(oficial["pipeline"].predict_proba(muestra), buscado.predict_proba(muestra))


def test_generar_modelos_distingue_los_dos_papeles(tmp_path, particion):
    artefactos = produccion.generar_modelos(particion.X_train, particion.y_train, carpeta=tmp_path)
    oficial = produccion.cargar_modelo(tmp_path / "modelo_oficial.joblib")
    prod = produccion.cargar_modelo(tmp_path / "modelo_produccion.joblib")
    assert oficial["rol"] == "oficial" and oficial["evaluado_en_test"] is True
    assert prod["rol"] == "candidato_produccion" and prod["evaluado_en_test"] is False
    assert "no evaluado en test" in prod["metricas"]["origen"]
    assert not any("test" in k for k in prod["metricas"])  # ninguna métrica del candidato viene del test
    assert prod["metricas"]["churners_capturados_20"] == artefactos["produccion"]["metricas"]["churners_capturados_20"]
    assert oficial["huella_datos_train"] == prod["huella_datos_train"]  # mismos clientes de train
    assert 0.98 < prod["metricas"]["auc_cv"] < 1
    assert produccion.ficha(prod)["evaluado_en_test"] is False


def test_linea_de_comandos_puntua_un_archivo(tmp_path, clientes, artefacto):
    ruta_modelo = produccion.guardar_modelo(artefacto, tmp_path / "modelo.joblib")
    entrada, salida = tmp_path / "clientes.csv", tmp_path / "lista.csv"
    clientes.to_csv(entrada, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # el archivo limpio no debe generar advertencias
        produccion.main(["puntuar", str(entrada), str(salida), "--capacidad", "0.2", "--modelo", str(ruta_modelo)])
    lista = pd.read_csv(salida, index_col=0)
    assert len(lista) == len(clientes) and lista["contactar"].sum() == negocio.contactados(len(clientes), 0.2)
    assert lista["prob_churn"].is_monotonic_decreasing
