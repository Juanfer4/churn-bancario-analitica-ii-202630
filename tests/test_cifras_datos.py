"""Cifras citadas en la interpretación de la Fase 2, 2.1 (carga, validación y partición)."""
from churn import datos


def test_cifras_citadas_en_la_fase_2():
    crudo = datos.cargar_datos()
    assert crudo.shape == (10127, 21)
    assert crudo.isna().sum().sum() == 0 and not crudo.duplicated().any() and crudo[datos.COLUMNA_ID].is_unique
    reporte = datos.validar_datos(crudo, estricto=False)
    assert reporte["cumple"].all() and len(reporte) == 26

    limpio = datos.limpiar_datos(crudo)
    balance = datos.balance_objetivo(limpio)
    assert balance.loc[1, "clientes"] == 1627 and balance.loc[0, "clientes"] == 8500
    assert round(100 * 1627 / 10127, 2) == 16.07 and round(100 * 8500 / 10127, 2) == 83.93

    p = datos.particionar(limpio)
    assert len(p.y_train) == 8101 and len(p.y_test) == 2026
    assert round(100 * p.y_train.mean(), 2) == 16.07 and round(100 * p.y_test.mean(), 2) == 16.04
