"""Pruebas del módulo de datos.

Verifican que el archivo cumple lo que el resto del proyecto da por hecho y que
la validación sí detecta datos rotos (una prueba que nunca falla no protege nada).
"""
import pandas as pd
import pytest

from churn import config, datos


@pytest.fixture(scope="module")
def crudo():
    return datos.cargar_datos()


@pytest.fixture(scope="module")
def limpio(crudo):
    return datos.limpiar_datos(crudo)


# --- Carga y validación con los datos reales -------------------------------------------
def test_dimensiones_del_caso(crudo):
    # El PDF oficial describe 10.127 clientes y 21 variables
    assert crudo.shape == (10127, 21)
    assert list(crudo.columns) == datos.COLUMNAS_ESPERADAS


def test_validacion_pasa_con_datos_reales(crudo):
    reporte = datos.validar_datos(crudo, estricto=True)
    assert reporte["cumple"].all()


# --- La validación debe detectar errores --------------------------------------------
def test_detecta_identidad_de_cupo_rota(crudo):
    roto = crudo.copy()
    roto.loc[roto.index[0], "avg_open_to_buy"] += 100
    with pytest.raises(datos.ErrorDeDatos):
        datos.validar_datos(roto)


def test_detecta_columna_faltante(crudo):
    with pytest.raises(datos.ErrorDeDatos):
        datos.validar_datos(crudo.drop(columns="total_trans_ct"))


def test_detecta_categoria_nueva(crudo):
    roto = crudo.copy()
    roto.loc[roto.index[0], "card_category"] = "Black"
    with pytest.raises(datos.ErrorDeDatos):
        datos.validar_datos(roto)


def test_detecta_faltantes_y_duplicados(crudo):
    roto = pd.concat([crudo, crudo.iloc[[0]]])
    roto.iloc[1, 2] = None
    reporte = datos.validar_datos(roto, estricto=False)
    fallas = set(reporte.loc[~reporte["cumple"], "chequeo"])
    assert {"Sin valores faltantes", "Sin filas duplicadas", "Identificador único"} <= fallas


# --- Limpieza --------------------------------------------------------------------
def test_objetivo_binario_y_sin_texto_original(limpio):
    assert config.OBJETIVO not in limpio.columns
    assert set(limpio[datos.COLUMNA_CHURN].unique()) == {0, 1}
    assert limpio[datos.COLUMNA_CHURN].sum() == 1627  # Attrited Customer en el archivo


def test_indice_es_el_cliente(limpio):
    assert limpio.index.name == datos.COLUMNA_ID
    assert limpio.index.is_unique


def test_limpieza_no_altera_variables_explicativas(crudo, limpio):
    explicativas = [c for c in datos.COLUMNAS_ESPERADAS if c not in (datos.COLUMNA_ID, config.OBJETIVO)]
    assert list(limpio.columns) == explicativas + [datos.COLUMNA_CHURN]
    original = crudo.set_index(datos.COLUMNA_ID)[explicativas]
    pd.testing.assert_frame_equal(limpio[explicativas], original)


# --- Partición ---------------------------------------------------------------------
def test_particion_tamanos_y_estratificacion(limpio):
    p = datos.particionar(limpio)
    assert len(p.X_train) == 8101 and len(p.X_test) == 2026
    tasa_total = limpio[datos.COLUMNA_CHURN].mean()
    assert abs(p.y_train.mean() - tasa_total) < 0.002
    assert abs(p.y_test.mean() - tasa_total) < 0.002


def test_particion_sin_solapamiento_y_reproducible(limpio):
    p1 = datos.particionar(limpio)
    p2 = datos.particionar(limpio)
    assert p1.X_train.index.intersection(p1.X_test.index).empty
    assert p1.X_test.index.equals(p2.X_test.index)
    assert datos.COLUMNA_CHURN not in p1.X_train.columns


def test_guardar_y_recargar_reconstruye_lo_mismo(limpio, tmp_path):
    p = datos.particionar(limpio)
    datos.guardar_procesado(limpio, p, tmp_path)
    limpio_2, p_2 = datos.cargar_procesado(tmp_path)
    pd.testing.assert_frame_equal(limpio_2, limpio, check_dtype=False)
    # Mismos clientes y en el mismo orden: el orden define los pliegues de la validación cruzada
    assert p_2.X_train.index.equals(p.X_train.index) and p_2.X_test.index.equals(p.X_test.index)
    assert p_2.y_train.sum() == p.y_train.sum()


# --- Tablas de vista general -------------------------------------------------------
def test_balance_objetivo_suma_uno(limpio):
    tabla = datos.balance_objetivo(limpio)
    assert tabla["clientes"].sum() == 10127
    assert abs(tabla["proporcion"].sum() - 1) < 1e-3
    assert tabla.loc[1, "clientes"] == 1627


def test_resumen_variables_cubre_todas_las_columnas(crudo):
    resumen = datos.resumen_variables(crudo)
    assert list(resumen.index) == list(crudo.columns)
    assert resumen["faltantes"].sum() == 0


def test_orden_de_categorias_coincide_con_las_esperadas():
    for col, orden in datos.ORDEN_CATEGORIAS.items():
        assert set(orden) == datos.CATEGORIAS_ESPERADAS[col]
        assert len(orden) == len(set(orden))
