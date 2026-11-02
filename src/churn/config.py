"""Parametros globales del proyecto: rutas relativas a la raiz del repositorio y constantes.

Todas las rutas se construyen desde la ubicacion de este archivo, asi el proyecto corre igual
en cualquier computador sin rutas absolutas del tipo C:\\Users\\...
"""
from pathlib import Path

# Raiz del repositorio: src/churn/config.py -> parents[2]
RAIZ = Path(__file__).resolve().parents[2]

# Datos: los crudos nunca se modifican; los procesados se regeneran con codigo
RUTA_DATOS_CRUDOS = RAIZ / "data" / "raw" / "bank_churn.xlsx"
RUTA_DATOS_PROCESADOS = RAIZ / "data" / "processed"

# Salidas
RUTA_MODELOS = RAIZ / "models"
RUTA_EXPERIMENTOS = RUTA_MODELOS / "experimentos"  # caché de las búsquedas de hiperparámetros
RUTA_FIGURAS = RAIZ / "reports" / "figures"
RUTA_TABLAS = RAIZ / "reports" / "tables"

# Variable objetivo y clase positiva (churn = cliente que se fue a otro banco)
OBJETIVO = "attrition_flag"
CLASE_POSITIVA = "Attrited Customer"

# Reproducibilidad y particion
SEMILLA = 42
TEST_SIZE = 0.20
N_PLIEGUES = 5

# Escenarios de capacidad de contacto: SUPUESTOS DE ANALISIS, no datos oficiales del banco
ESCENARIOS_CAPACIDAD = (0.10, 0.20, 0.30)

# Monitoreo y control: umbrales (amarillo, rojo) de cada indicador. Los del PSI son la
# convención de la industria; los de captura son SUPUESTOS del equipo y se pueden ajustar.
UMBRALES_MONITOREO = {
    "psi": (0.10, 0.25),             # índice de estabilidad poblacional por variable
    "caida_captura": (0.02, 0.05),   # caída de la captura del 20 % contactado, en puntos (0,02 = 2 pp)
}

# Valores del producto s · r (tasa de éxito del beneficio por valor relativo del cliente)
# que se prueban en la lectura de negocio: SUPUESTOS de análisis, no datos del banco.
PRODUCTOS_SR = (1.5, 3.0, 10.0, 30.0)
