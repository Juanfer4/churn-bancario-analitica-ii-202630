"""Pruebas de `reportes`: guardar tablas igual que el código que reemplazan."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from churn import reportes


def test_guardar_tabla_redondea_y_conserva_el_indice(tmp_path):
    tabla = pd.DataFrame({"a": [0.123456789, 2.0], "b": ["x", "y"]}, index=pd.Index(["u", "v"], name="idx"))
    ruta = reportes.guardar_tabla(tabla, "ejemplo", decimales=4, carpeta=tmp_path)
    assert ruta == tmp_path / "ejemplo.csv"
    leida = pd.read_csv(ruta, index_col="idx")
    assert leida.loc["u", "a"] == 0.1235 and list(leida.index) == ["u", "v"]
    # idéntico al código que reemplaza
    tabla.round(4).to_csv(tmp_path / "referencia.csv")
    assert ruta.read_bytes() == (tmp_path / "referencia.csv").read_bytes()


def test_guardar_tabla_sin_redondeo_ni_indice(tmp_path):
    tabla = pd.DataFrame({"a": [0.123456789]})
    ruta = reportes.guardar_tabla(tabla, "sin_redondeo", decimales=None, indice=False, carpeta=tmp_path)
    assert ruta.read_text().splitlines() == ["a", "0.123456789"]


def test_guardar_tabla_acepta_series_y_crea_la_carpeta(tmp_path):
    ruta = reportes.guardar_tabla(pd.Series([1.23456789], name="valor"), "serie", carpeta=tmp_path / "nueva")
    assert ruta.exists() and "1.234568" in ruta.read_text()


def test_mostrar_figura_guarda_el_png(tmp_path, monkeypatch):
    from churn import graficos
    monkeypatch.setattr(graficos.config, "RUTA_FIGURAS", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    monkeypatch.setattr(graficos, "guardar_figura", lambda f, n: f.savefig(tmp_path / f"{n}.png") or tmp_path / f"{n}.png")
    ruta = reportes.mostrar_figura(fig, "figura_prueba")
    assert ruta.exists()
    plt.close("all")
