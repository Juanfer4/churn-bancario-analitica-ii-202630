"""El código y la documentación de uso citan las fases CRISP-DM del notebook (por ejemplo "Fase 5"
o "5.7"), no la numeración anterior por secciones ("sección 5", "§ 6")."""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PATRON = re.compile(r"secci[oó]n(es)?\s+\d|§\s*\d", re.IGNORECASE)
DOCUMENTOS_DE_USO = ["README.md", "docs/GUIA_ARCHIVOS_Y_EJECUCION.md"]  # los historiales se dejan como están


def _referencias_viejas(archivos) -> list[str]:
    viejos = []
    for archivo in archivos:
        for n, linea in enumerate(archivo.read_text(encoding="utf-8").splitlines(), 1):
            if PATRON.search(linea):
                viejos.append(f"{archivo.relative_to(RAIZ)}:{n}")
    return viejos


def test_el_codigo_no_cita_la_numeracion_antigua():
    archivos = [a for carpeta in ("src", "tests") for a in (RAIZ / carpeta).rglob("*.py")
                if a.name != Path(__file__).name]
    viejos = _referencias_viejas(archivos)
    assert not viejos, f"Referencias a la numeración antigua: {viejos}"


def test_la_documentacion_de_uso_no_cita_la_numeracion_antigua():
    archivos = [RAIZ / ruta for ruta in DOCUMENTOS_DE_USO if (RAIZ / ruta).exists()]
    viejos = _referencias_viejas(archivos)
    assert not viejos, f"Referencias a la numeración antigua: {viejos}"
