"""Pruebas de la estructura del experimento (sin entrenar los diez modelos completos)."""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from churn import comparacion, config, datos, modelos


@pytest.fixture(scope="module")
def particion():
    return datos.particionar(datos.limpiar_datos(datos.cargar_datos()))


def _modelo_pequeno(**kwargs):
    base = dict(nombre="LR", familia="lineal",
                estimador=LogisticRegression(solver="liblinear", max_iter=2000),
                hiperparametros={"modelo__C": [0.1, 1, 10]}, escalar=True,
                pesos=("class_weight", "balanced"))
    base.update(kwargs)
    return modelos.Modelo(**base)


def test_catalogo_tiene_diez_modelos_con_nombres_unicos():
    catalogo = modelos.catalogo_modelos()
    assert len(catalogo) == 10
    assert len({m.nombre for m in catalogo}) == 10
    # Modelos sensibles a la escala con escalado; árboles y ensambles sin él
    escalados = {m.nombre for m in catalogo if m.escalar}
    assert escalados == {"Regresión logística", "SVM", "Red neuronal (MLP)"}


def test_semilla_fija_en_todos_los_estimadores():
    for m in modelos.catalogo_modelos():
        assert m.estimador.get_params().get("random_state") == config.SEMILLA, m.nombre


def test_grilla_combina_hiperparametros_con_estrategias():
    m = _modelo_pequeno()
    grilla = m.grilla(pd.Series([0, 0, 1]))
    assert len(grilla) == 3  # ninguna, pesos, SMOTE
    assert m.n_candidatos() == 9
    ninguna, pesos, smote = grilla
    assert ninguna["balanceo"] == ["passthrough"] and ninguna["modelo__class_weight"] == [None]
    assert pesos["balanceo"] == ["passthrough"] and pesos["modelo__class_weight"] == ["balanced"]
    assert isinstance(smote["balanceo"][0], SMOTE) and smote["modelo__class_weight"] == [None]


def test_pesos_por_proporcion_usan_la_relacion_de_clases_de_train(particion):
    m = _modelo_pequeno(pesos=("scale_pos_weight", modelos.PESOS_PROPORCION))
    grilla = m.grilla(particion.y_train)
    esperado = (particion.y_train == 0).sum() / (particion.y_train == 1).sum()
    assert grilla[1]["modelo__scale_pos_weight"] == [pytest.approx(esperado)]
    assert grilla[0]["modelo__scale_pos_weight"] == [1.0]
    assert esperado == pytest.approx(6799 / 1302)


def test_modelo_sin_pesos_ni_smote_solo_prueba_ninguna():
    m = _modelo_pequeno(pesos=None, smote=False)
    assert m.estrategias() == ["ninguna"]
    assert m.n_candidatos() == 3


def test_estrategia_de_traduce_los_parametros():
    m = _modelo_pequeno()
    assert m.estrategia_de({"balanceo": "passthrough", "modelo__class_weight": None}) == "ninguna"
    assert m.estrategia_de({"balanceo": "passthrough", "modelo__class_weight": "balanced"}) == "pesos"
    assert m.estrategia_de({"balanceo": SMOTE(), "modelo__class_weight": None}) == "SMOTE"


def test_ajuste_pequeno_resultados_y_borde(particion):
    X, y = particion.X_train.iloc[:1500], particion.y_train.iloc[:1500]
    m = _modelo_pequeno().ajustar(X, y, n_jobs=1)
    res = m.resultados()
    assert len(res) == 9 and set(res["estrategia"]) == {"ninguna", "pesos", "SMOTE"}
    assert res["rango"].iloc[0] == 1
    resumen = m.resumen()
    assert resumen["auc_cv"] == pytest.approx(res["auc_cv"].max())
    assert len(m.auc_por_pliegue()) == config.N_PLIEGUES
    assert np.mean(m.auc_por_pliegue()) == pytest.approx(resumen["auc_cv"])
    elegido = m.busqueda_.best_params_["modelo__C"]
    borde = m.hiperparametros_en_borde()
    assert bool(borde) == (elegido in (0.1, 10))


def test_comparador_ordena_y_compara_en_pares(particion):
    X, y = particion.X_train.iloc[:1500], particion.y_train.iloc[:1500]
    a = _modelo_pequeno(nombre="A", hiperparametros={"modelo__C": [1]}, pesos=None, smote=False)
    b = _modelo_pequeno(nombre="B", hiperparametros={"modelo__C": [0.001]}, pesos=None, smote=False)
    comp = comparacion.Comparador([a, b])
    comp.entrenar_todos(X, y, n_jobs=1)
    tabla = comp.tabla_cv()
    assert list(tabla.index) == sorted(["A", "B"], key=lambda n: -tabla.loc[n, "auc_cv"])
    pareada = comp.comparacion_pareada()
    assert pareada.loc[tabla.index[0], "diferencia_media"] == 0
    assert tabla.index[0] in comp.finalistas()
    with pytest.raises(ValueError):
        comparacion.Comparador([a, a])


def test_decidir_candidatas_aplica_la_regla():
    filas = [
        # modelo X: "+ saldo" mejora de forma consistente; "+ ticket" mejora poco y con ruido
        ("X", "base", 0.90, 0.0, 0.0, 0), ("X", "+ ticket", 0.901, 0.001, 0.002, 3),
        ("X", "+ saldo", 0.91, 0.010, 0.002, 5), ("X", "+ ticket + saldo", 0.905, 0.005, 0.006, 4),
        # modelo Y: nada mejora de forma consistente
        ("Y", "base", 0.95, 0.0, 0.0, 0), ("Y", "+ ticket", 0.951, 0.001, 0.003, 3),
        ("Y", "+ saldo", 0.949, -0.001, 0.001, 1), ("Y", "+ ticket + saldo", 0.950, 0.0, 0.001, 2),
    ]
    tabla = pd.DataFrame(filas, columns=["modelo", "combinacion", "auc_cv", "mejora_media",
                                         "mejora_desv", "pliegues_que_mejoran"])
    decision = comparacion.decidir_candidatas(tabla)
    assert decision["X"] == (False, True)
    assert decision["Y"] == (False, False)


def test_seleccion_parsimoniosa_prefiere_la_estrategia_mas_simple_si_hay_empate():
    m = _modelo_pequeno()
    cv_results = {
        "mean_test_auc": np.array([0.950, 0.952, 0.940]),
        "std_test_auc": np.array([0.003, 0.003, 0.003]),
        "params": [{"balanceo": "passthrough", "modelo__class_weight": None},       # ninguna
                   {"balanceo": SMOTE(), "modelo__class_weight": None},              # SMOTE (mejor)
                   {"balanceo": "passthrough", "modelo__class_weight": "balanced"}],  # pesos, fuera del empate
    }
    assert m.seleccionar_candidato(cv_results) == 0
    # Si la diferencia supera la desviación, gana el de mayor AUC aunque sea más complejo
    cv_results["mean_test_auc"] = np.array([0.940, 0.952, 0.930])
    assert m.seleccionar_candidato(cv_results) == 1


def test_ampliar_grillas_conserva_candidatas_y_usan_solo_ninguna():
    catalogo = modelos.catalogo_modelos()
    catalogo[0].ticket_promedio = True
    comp = comparacion.Comparador(catalogo)
    grillas = {catalogo[0].nombre: {"modelo__C": [0.5, 1, 2]}}
    nuevos = comparacion.ampliar_grillas(comp, grillas)
    assert len(nuevos) == 1 and nuevos[0].estrategias() == ["ninguna"]
    assert nuevos[0].ticket_promedio is True and nuevos[0].hiperparametros == {"modelo__C": [0.5, 1, 2]}
    assert catalogo[0].hiperparametros != nuevos[0].hiperparametros  # el original no se modifica


def test_grillas_de_ampliacion_cubren_modelos_del_catalogo():
    nombres = {m.nombre for m in modelos.catalogo_modelos()}
    assert set(comparacion.GRILLAS_RONDA_2) <= nombres and set(comparacion.GRILLAS_RONDA_3) <= set(comparacion.GRILLAS_RONDA_2)


def test_huella_cambia_con_la_grilla_y_no_sin_cambios(particion):
    X, y = particion.X_train.iloc[:500], particion.y_train.iloc[:500]
    cv = modelos.validacion_cruzada()
    h1 = comparacion.huella_modelo(X, y, _modelo_pequeno(), cv)
    h2 = comparacion.huella_modelo(X, y, _modelo_pequeno(), cv)
    h3 = comparacion.huella_modelo(X, y, _modelo_pequeno(hiperparametros={"modelo__C": [1]}), cv)
    h4 = comparacion.huella_modelo(X.iloc[:499], y.iloc[:499], _modelo_pequeno(), cv)
    h5 = comparacion.huella_modelo(X, y, _modelo_pequeno(ticket_promedio=True), cv)
    assert h1 == h2 and len({h1, h3, h4, h5}) == 4


def test_huella_cambia_si_cambia_la_version_de_una_libreria(particion, monkeypatch):
    X, y = particion.X_train.iloc[:500], particion.y_train.iloc[:500]
    cv = modelos.validacion_cruzada()
    antes = comparacion.huella_modelo(X, y, _modelo_pequeno(), cv)
    nombres = [n for n, _ in comparacion.versiones_librerias()]
    assert nombres == ["scikit-learn", "imbalanced-learn", "lightgbm", "xgboost"]
    monkeypatch.setattr(comparacion, "versiones_librerias", lambda: (("lightgbm", "0.0.0"),))
    assert comparacion.huella_modelo(X, y, _modelo_pequeno(), cv) != antes


def test_cache_por_modelo_guarda_recarga_y_detecta_cambios(particion, tmp_path, capsys):
    X, y = particion.X_train.iloc[:800], particion.y_train.iloc[:800]
    def lote(c_b):
        return [_modelo_pequeno(nombre="A", pesos=None, smote=False),
                _modelo_pequeno(nombre="B", pesos=None, smote=False, hiperparametros={"modelo__C": [c_b]})]
    a = comparacion.entrenar_o_cargar(lote(1), X, y, tmp_path, "prueba", n_jobs=1)
    capsys.readouterr()
    b = comparacion.entrenar_o_cargar(lote(1), X, y, tmp_path, "prueba", n_jobs=1)
    assert capsys.readouterr().out.count("cargado de la caché") == 2
    assert a.tabla_cv()["auc_cv"].equals(b.tabla_cv()["auc_cv"])
    comparacion.entrenar_o_cargar(lote(2), X, y, tmp_path, "prueba", n_jobs=1)
    salida = capsys.readouterr().out
    assert salida.count("cargado de la caché") == 1 and "B: 1 candidatos" in salida


def test_tablas_de_apoyo_y_combinacion_de_rondas(particion):
    X, y = particion.X_train.iloc[:800], particion.y_train.iloc[:800]
    r1 = comparacion.Comparador([_modelo_pequeno(nombre="A"), _modelo_pequeno(nombre="B", pesos=None, smote=False)])
    r1.entrenar_todos(X, y, n_jobs=1)
    catalogo = comparacion.tabla_catalogo(list(r1.modelos.values()))
    assert catalogo.loc["A", "candidatos"] == 9 and catalogo.loc["A", "ajustes_cv"] == 45
    por_estrategia = comparacion.auc_por_estrategia(r1)
    assert por_estrategia.loc["A", ["ninguna", "pesos", "SMOTE"]].notna().all()
    assert np.isnan(por_estrategia.loc["B", "pesos"])
    r2 = comparacion.Comparador(comparacion.ampliar_grillas(r1, {"A": {"modelo__C": [0.5, 1, 2]}}))
    r2.entrenar_todos(X, y, n_jobs=1)
    tabla_rondas = comparacion.comparar_rondas(r1, r2)
    assert list(tabla_rondas.index) == ["A"]
    final = comparacion.combinar(r1, r2)
    assert final.modelos["A"] is r2.modelos["A"] and final.modelos["B"] is r1.modelos["B"]


def test_regla_para_seguir_ampliando(particion):
    X, y = particion.X_train.iloc[:800], particion.y_train.iloc[:800]
    r1 = comparacion.Comparador([_modelo_pequeno(nombre="A", hiperparametros={"modelo__C": [0.0001, 0.001]},
                                             pesos=None, smote=False)])
    r1.entrenar_todos(X, y, n_jobs=1)
    r2 = comparacion.Comparador(comparacion.ampliar_grillas(r1, {"A": {"modelo__C": [0.001, 1, 10]}}))
    r2.entrenar_todos(X, y, n_jobs=1)
    mejora = r2.modelos["A"].resumen()["auc_cv"] - r1.modelos["A"].resumen()["auc_cv"]
    en_borde = not r2.hiperparametros_en_borde().empty
    esperado = ["A"] if (en_borde and mejora > r2.modelos["A"].resumen()["auc_cv_desv"]) else []
    assert comparacion.modelos_a_ampliar(r1, r2) == esperado
    final = comparacion.combinar(r1, r2)
    assert final.modelos["A"] is r2.modelos["A"]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_borde_en_arquitecturas_de_red_por_numero_de_neuronas(particion):
    from sklearn.neural_network import MLPClassifier
    X, y = particion.X_train.iloc[:600], particion.y_train.iloc[:600]
    m = modelos.Modelo("MLP", "red neuronal", MLPClassifier(max_iter=300, random_state=0),
                       {"modelo__hidden_layer_sizes": [(4,), (8, 4)]}, escalar=True, pesos=None, smote=False)
    m.ajustar(X, y, n_jobs=1)
    borde = m.hiperparametros_en_borde()
    assert len(borde) == 1 and borde[0]["probados"] == [(4,), (8, 4)]


def test_cache_corrupta_se_reentrena(particion, tmp_path, capsys):
    X, y = particion.X_train.iloc[:600], particion.y_train.iloc[:600]
    lote = [_modelo_pequeno(nombre="A", pesos=None, smote=False)]
    comparacion._archivo_cache(tmp_path, "p", "A").write_bytes(b"no es un joblib")
    comp = comparacion.entrenar_o_cargar(lote, X, y, tmp_path, "p", n_jobs=1)
    assert "no se pudo leer la caché" in capsys.readouterr().out
    assert comp.modelos["A"].busqueda_ is not None


def test_auc_por_valor_y_combinacion_de_finalistas(particion):
    X, y = particion.X_train.iloc[:1200], particion.y_train.iloc[:1200]
    m = _modelo_pequeno(pesos=None, smote=False).ajustar(X, y, n_jobs=1)
    por_valor = comparacion.auc_por_valor(m, "C")
    assert len(por_valor) == 3 and por_valor["auc_cv"].iloc[0] == pytest.approx(m.resumen()["auc_cv"])
    assert por_valor["auc_cv"].is_monotonic_decreasing
    # Combinar un modelo consigo mismo no puede cambiar la votación: mismo AUC que la referencia
    pipe = m.busqueda_.best_estimator_
    tabla = comparacion.combinar_finalistas({"A": pipe, "B": pipe}, X, y, referencia="A", n_jobs=1)
    assert list(tabla.index) == ["A", "Votación de finalistas", "Stacking de finalistas"]
    assert tabla.loc["Votación de finalistas", "diferencia_vs_referencia"] == pytest.approx(0, abs=1e-12)
    assert 0.5 < tabla.loc["Stacking de finalistas", "auc_cv"] <= 1
