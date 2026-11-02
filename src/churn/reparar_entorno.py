"""Punto de entrada de la línea de comandos para revisar y reparar el entorno en Windows.

    python -m churn.reparar_entorno              # diagnóstico
    python -m churn.reparar_entorno --reparar    # reparación

La lógica vive en `churn.entorno`; este archivo solo existe para poder ejecutarla con -m.
"""
import sys

from churn.entorno import main

sys.exit(main())
