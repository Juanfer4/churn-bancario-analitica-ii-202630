"""Librería del proyecto final de Analítica II: predicción de churn bancario.

El notebook y las pruebas importan estos módulos para mantener las celdas cortas y sin código
duplicado: datos, exploracion, features, modelos, comparacion, evaluacion, negocio,
explicabilidad, produccion, monitoreo, reportes y los de gráficos (graficos, graficos_modelado
y graficos_negocio). config guarda rutas y parámetros; entorno revisa el entorno en Windows.

En Windows se revisa al importar si scikit-learn trae una msvcp140.dll demasiado vieja para
LightGBM (ver `churn.entorno`); si es así, se muestra un aviso con el comando que lo repara.
"""
from churn import entorno as _entorno

_entorno.advertir_si_hay_choque()
