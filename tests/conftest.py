"""Herramientas compartidas por las pruebas de cifras citadas (`test_cifras_*.py`).

Cada prueba de cifras compara los números escritos en el texto del notebook con los que
calcula el código o guarda el notebook en `reports/tables`. Antes cada archivo repetía la
misma función de comparación y la misma omisión cuando faltan tablas; aquí viven una sola vez.
"""
import pytest

from churn import config


class Cifras:
    """Acumula todas las diferencias entre lo calculado y lo citado, para reportarlas juntas."""

    def __init__(self):
        self.fallas = []

    def chk(self, nombre: str, calculado, citado: float, tol: float = 0.00005) -> None:
        """Registra una falla si la cifra calculada se aleja de la citada más que `tol`
        (la precisión con que se escribió en el texto)."""
        if abs(float(calculado) - citado) > tol + 1e-9:
            self.fallas.append(f"{nombre}: calculado {calculado} vs citado {citado}")

    def verificar(self) -> None:
        assert not self.fallas, self.fallas


@pytest.fixture
def cifras() -> Cifras:
    return Cifras()


def exigir_tablas(nombres: list[str]) -> None:
    """Omite la prueba si faltan tablas del notebook: hay que ejecutarlo primero."""
    faltan = [n for n in nombres if not (config.RUTA_TABLAS / n).exists()]
    if faltan:
        pytest.skip(f"Ejecute el notebook para generar: {faltan}")
