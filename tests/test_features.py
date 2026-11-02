"""Pruebas de la preparación de variables: resultados correctos y ausencia de fuga."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from churn import datos, features


@pytest.fixture(scope="module")
def particion():
    limpio = datos.limpiar_datos(datos.cargar_datos())
    return datos.particionar(limpio)


# --- CreadorVariables --------------------------------------------------------------
def test_elimina_avg_open_to_buy_y_conserva_el_resto(particion):
    X = particion.X_train
    salida = features.CreadorVariables().fit_transform(X)
    assert "avg_open_to_buy" not in salida.columns
    assert list(salida.columns) == [c for c in X.columns if c != "avg_open_to_buy"]
    pd.testing.assert_frame_equal(salida, X.drop(columns="avg_open_to_buy"))


def test_ticket_promedio_es_monto_entre_transacciones(particion):
    X = particion.X_train
    salida = features.CreadorVariables(ticket_promedio=True).fit_transform(X)
    esperado = X["total_trans_amt"] / X["total_trans_ct"]
    np.testing.assert_allclose(salida["ticket_promedio"], esperado)


def test_ticket_promedio_sin_transacciones_no_rompe():
    X = pd.DataFrame({"total_revolving_bal": [0, 10], "total_trans_amt": [100, 50],
                      "total_trans_ct": [0, 5], "avg_open_to_buy": [1, 1]})
    salida = features.CreadorVariables(ticket_promedio=True).fit_transform(X)
    assert list(salida["ticket_promedio"]) == [0.0, 10.0]


def test_indicadores_de_saldo(particion):
    X = particion.X_train
    creador = features.CreadorVariables(indicadores_saldo=True).fit(X)
    salida = creador.transform(X)
    assert creador.saldo_maximo_ == 2517
    assert salida["saldo_cero"].sum() == (X["total_revolving_bal"] == 0).sum() == 1968
    assert salida["saldo_en_maximo"].sum() == (X["total_revolving_bal"] == 2517).sum() == 400


def test_saldo_maximo_se_aprende_solo_de_los_datos_de_ajuste(particion):
    """Si el test tuviera un saldo mayor, no debe cambiar lo aprendido con train."""
    creador = features.CreadorVariables(indicadores_saldo=True).fit(particion.X_train)
    test_modificado = particion.X_test.copy()
    test_modificado.iloc[0, test_modificado.columns.get_loc("total_revolving_bal")] = 99999
    salida = creador.transform(test_modificado)
    assert creador.saldo_maximo_ == 2517
    assert salida["saldo_en_maximo"].iloc[0] == 1  # por encima del tope de train también cuenta


def test_rechaza_entradas_sin_nombres_o_sin_columnas(particion):
    with pytest.raises(TypeError):
        features.CreadorVariables().fit(particion.X_train.values)
    with pytest.raises(ValueError):
        features.CreadorVariables().fit(particion.X_train.drop(columns="total_trans_ct"))


# --- Codificación y escalado --------------------------------------------------------
@pytest.mark.parametrize("escalar", [True, False])
@pytest.mark.parametrize("ticket,saldo", [(False, False), (True, True)])
def test_salida_completa_sin_faltantes_ni_infinitos(particion, escalar, ticket, saldo):
    prep = features.construir_preparacion(escalar, ticket, saldo).fit(particion.X_train)
    for X in (particion.X_train, particion.X_test):
        Z = prep.transform(X)
        assert Z.shape[0] == len(X)
        assert np.isfinite(Z).all()
    nombres = features.nombres_salida(prep)
    assert len(nombres) == Z.shape[1] == len(set(nombres))
    # 13 numéricas base (+1 ticket, +2 saldo) y 21 One-Hot: gender 1, education 7, marital 4, income 6, card 3
    assert Z.shape[1] == 13 + (1 if ticket else 0) + (2 if saldo else 0) + 21


def test_columnas_one_hot_esperadas(particion):
    prep = features.construir_preparacion(escalar=False).fit(particion.X_train)
    nombres = features.nombres_salida(prep)
    # Gold y Platinum (menos de 100 clientes) quedan agrupadas; gender conserva una columna
    assert "card_category_infrequent_sklearn" in nombres
    assert not any(n in nombres for n in ("card_category_Gold", "card_category_Platinum"))
    assert {"card_category_Blue", "card_category_Silver"} <= set(nombres)
    assert [n for n in nombres if n.startswith("gender_")] == ["gender_M"]
    assert {"education_level_Unknown", "marital_status_Unknown", "income_category_Unknown"} <= set(nombres)
    codificador = prep.named_steps["codificador"].named_transformers_["categoricas"]
    infrecuentes = dict(zip(datos.COLUMNAS_CATEGORICAS, codificador.infrequent_categories_))
    assert set(infrecuentes["card_category"]) == {"Gold", "Platinum"}
    assert all(infrecuentes[c] is None for c in datos.COLUMNAS_CATEGORICAS if c != "card_category")


def test_one_hot_cada_fila_marca_una_categoria_por_variable(particion):
    prep = features.construir_preparacion(escalar=False).fit(particion.X_train)
    Z = pd.DataFrame(prep.transform(particion.X_test), columns=features.nombres_salida(prep))
    for col in datos.COLUMNAS_CATEGORICAS:
        bloque = Z[[n for n in Z.columns if n.startswith(col + "_")]]
        esperado = 1 if col != "gender" else None
        if esperado:
            assert (bloque.sum(axis=1) == 1).all(), col
        else:
            assert bloque.isin([0, 1]).all().all()


def test_estandarizacion_solo_con_train(particion):
    prep = features.construir_preparacion(escalar=True).fit(particion.X_train)
    nombres = features.nombres_salida(prep)
    numericas = [i for i, n in enumerate(nombres) if not any(n.startswith(c + "_") for c in datos.COLUMNAS_CATEGORICAS)]
    Z_train = prep.transform(particion.X_train)[:, numericas]
    Z_test = prep.transform(particion.X_test)[:, numericas]
    np.testing.assert_allclose(Z_train.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(Z_train.std(axis=0), 1, atol=1e-6)
    # En test la media no es exactamente 0: prueba de que no se reajustó con el test
    assert np.abs(Z_test.mean(axis=0)).max() > 1e-3


def test_sin_escalar_las_numericas_pasan_intactas(particion):
    prep = features.construir_preparacion(escalar=False).fit(particion.X_train)
    nombres = features.nombres_salida(prep)
    Z = pd.DataFrame(prep.transform(particion.X_test), columns=nombres, index=particion.X_test.index)
    for col in ["total_trans_ct", "credit_limit", "customer_age"]:
        np.testing.assert_allclose(Z[col], particion.X_test[col])


def test_transformar_test_no_cambia_lo_aprendido(particion):
    prep = features.construir_preparacion(escalar=True, ticket_promedio=True, indicadores_saldo=True)
    prep.fit(particion.X_train)
    lambdas_antes = prep.named_steps["codificador"].named_transformers_["numericas"].lambdas_.copy()
    prep.transform(particion.X_test)
    lambdas_despues = prep.named_steps["codificador"].named_transformers_["numericas"].lambdas_
    np.testing.assert_array_equal(lambdas_antes, lambdas_despues)


def test_categoria_nueva_no_rompe_la_transformacion(particion):
    prep = features.construir_preparacion(escalar=False).fit(particion.X_train)
    nuevo = particion.X_test.iloc[[0]].copy()
    nuevo["card_category"] = "Black"
    with pytest.warns(UserWarning, match="unknown categories"):  # sklearn avisa, pero la envía a infrecuentes
        Z = pd.DataFrame(prep.transform(nuevo), columns=features.nombres_salida(prep))
    assert Z["card_category_infrequent_sklearn"].iloc[0] == 1


# --- Compatibilidad con GridSearchCV -------------------------------------------------
def test_parametros_expuestos_y_clonables():
    prep = features.construir_preparacion(escalar=True)
    parametros = prep.get_params()
    assert "variables__ticket_promedio" in parametros and "variables__indicadores_saldo" in parametros
    copia = clone(prep).set_params(variables__ticket_promedio=True)
    assert copia.named_steps["variables"].ticket_promedio is True
    assert prep.named_steps["variables"].ticket_promedio is False


def test_preparacion_funciona_dentro_de_validacion_cruzada(particion):
    """Integración: la preparación se reajusta en cada pliegue sin errores y el pipeline aprende."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    modelo = Pipeline([("preparacion", features.construir_preparacion(escalar=True)),
                       ("modelo", LogisticRegression(max_iter=2000))])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    auc = cross_val_score(modelo, particion.X_train, particion.y_train, cv=cv, scoring="roc_auc")
    assert np.isfinite(auc).all() and auc.mean() > 0.85


@pytest.mark.parametrize("escalar,n_chequeos", [(True, 7), (False, 4)])
def test_verificar_preparacion_pasa(particion, escalar, n_chequeos):
    prep = features.construir_preparacion(escalar, True, True).fit(particion.X_train)
    tabla = features.verificar_preparacion(prep, particion.X_train, particion.X_test)
    assert len(tabla) == n_chequeos and tabla["cumple"].all()
    assert len(features.columnas_numericas_salida(prep)) == 16
