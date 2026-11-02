"""Pruebas de la revisión del entorno de Windows con DLL falsas (corren en cualquier sistema)."""
from pathlib import Path

import pytest

from churn import entorno


def _dll_falsa(ruta: Path, version=None, enlazador=None) -> Path:
    """Archivo con lo mínimo que leen las funciones: encabezado PE y estructura de versión."""
    contenido = bytearray(512)
    contenido[0x3C:0x40] = (64).to_bytes(4, "little")
    contenido[64:68] = b"PE\x00\x00"
    if enlazador:
        contenido[64 + 24 + 2], contenido[64 + 24 + 3] = enlazador
    if version:
        mayor, menor, compilacion, revision = version
        contenido[300:304] = b"\xbd\x04\xef\xfe"
        contenido[308:312] = ((mayor << 16) | menor).to_bytes(4, "little")
        contenido[312:316] = ((compilacion << 16) | revision).to_bytes(4, "little")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(bytes(contenido))
    return ruta


def _escenario(tmp_path, v_sklearn, v_sistema, enlazador=(14, 44)):
    sklearn_dir, lightgbm_dir, system32 = tmp_path / "sklearn", tmp_path / "lightgbm", tmp_path / "System32"
    _dll_falsa(lightgbm_dir / "bin" / "lib_lightgbm.dll", enlazador=enlazador)
    for dll in entorno.DLLS_REEMPLAZABLES:
        _dll_falsa(sklearn_dir / ".libs" / dll, version=v_sklearn)
        _dll_falsa(system32 / dll, version=v_sistema)
    return entorno.rutas(sklearn_dir, lightgbm_dir, system32)


def test_lee_versiones():
    assert entorno.version_dll(Path("no_existe.dll")) is None


def test_lee_versiones_de_archivo(tmp_path):
    ruta = _dll_falsa(tmp_path / "a.dll", version=(14, 32, 31326, 0), enlazador=(14, 31))
    assert entorno.version_dll(ruta) == (14, 32, 31326, 0)
    assert entorno.version_enlazador(ruta) == (14, 31)


def test_detecta_el_choque_y_lo_repara_con_respaldo(tmp_path):
    rutas = _escenario(tmp_path, (14, 32, 31326, 0), (14, 44, 35211, 0))
    informe = entorno.diagnosticar(rutas)
    assert informe["hay_choque"] and informe["sistema_actualizado"]
    acciones = entorno.reparar(rutas)
    assert len(acciones) == 2
    assert entorno.version_dll(rutas["sklearn_msvcp140.dll"]) == (14, 44, 35211, 0)
    assert entorno.version_dll(Path(str(rutas["sklearn_msvcp140.dll"]) + ".original")) == (14, 32, 31326, 0)
    assert not entorno.diagnosticar(rutas)["hay_choque"]
    assert entorno.reparar(rutas) == ["No hay choque: no se cambió nada."]


def test_no_repara_si_windows_tambien_es_viejo(tmp_path):
    rutas = _escenario(tmp_path, (14, 32, 0, 0), (14, 36, 0, 0))
    with pytest.raises(RuntimeError, match="Redistributable"):
        entorno.reparar(rutas)
    assert entorno.version_dll(rutas["sklearn_msvcp140.dll"]) == (14, 32, 0, 0)  # no tocó nada


def test_sin_choque_si_lightgbm_es_compilado_con_version_vieja(tmp_path):
    rutas = _escenario(tmp_path, (14, 32, 0, 0), (14, 44, 0, 0), enlazador=(14, 30))
    assert not entorno.diagnosticar(rutas)["hay_choque"]


def test_lightgbm_funciona_en_este_sistema():
    assert entorno.probar_lightgbm()


def test_linea_de_comandos_en_windows_simulado(tmp_path, monkeypatch, capsys):
    rutas = _escenario(tmp_path, (14, 32, 0, 0), (14, 44, 0, 0))
    monkeypatch.setattr(entorno.sys, "platform", "win32")
    monkeypatch.setattr(entorno, "rutas", lambda *a, **k: rutas)
    monkeypatch.setattr(entorno, "probar_lightgbm", lambda: True)
    assert entorno.main([]) == 0 and "Choque detectado: sí" in capsys.readouterr().out
    assert entorno.main(["--reparar"]) == 0
    assert "msvcp140.dll: 14.32.0.0 -> 14.44.0.0" in capsys.readouterr().out
    assert not entorno.diagnosticar(rutas)["hay_choque"]
