"""Preparación de variables sin fuga de información (Fase 3 del notebook).

Todo lo que aprende algo de los datos (el saldo máximo, las categorías frecuentes, los
parámetros de Yeo-Johnson, medias y desviaciones) se ajusta dentro de un Pipeline con
los datos de entrenamiento de cada pliegue, nunca con el test. Así la validación cruzada
y la evaluación final miden el desempeño en datos que el modelo de verdad no ha visto.

Decisiones que implementa (notebook 2.9):
- Elimina `avg_open_to_buy` (identidad exacta: credit_limit - total_revolving_bal).
- Variables candidatas opcionales: ticket promedio, saldo en cero y saldo en el máximo.
  Se activan o desactivan con parámetros para que la validación cruzada decida.
- One-Hot para las categóricas, `Unknown` como categoría propia y agrupación de las
  categorías con menos de 100 clientes. Así se preparan para probarlas; la validación
  cruzada de 4.3 decide, modelo por modelo, si se usan (`Modelo.usar_categoricas`).
- Yeo-Johnson + estandarización solo para los modelos sensibles a la escala.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.utils.validation import check_is_fitted

from churn import datos

COLUMNAS_A_ELIMINAR = ["avg_open_to_buy"]
MIN_FRECUENCIA_CATEGORIA = 100  # categorías con menos clientes se agrupan como "infrecuentes"

COLUMNA_TICKET = "ticket_promedio"
COLUMNA_SALDO_CERO = "saldo_cero"
COLUMNA_SALDO_MAXIMO = "saldo_en_maximo"


class CreadorVariables(BaseEstimator, TransformerMixin):
    """Elimina las columnas redundantes y crea las variables candidatas.

    Parámetros
    ----------
    ticket_promedio : bool
        Agrega `ticket_promedio` = total_trans_amt / total_trans_ct (monto por transacción).
    indicadores_saldo : bool
        Agrega `saldo_cero` (1 si el saldo rotativo es 0) y `saldo_en_maximo` (1 si el
        saldo está en el máximo observado en train, que el EDA identificó como un tope).

    Lo único que aprende en `fit` es el saldo máximo de los datos que recibe.
    """

    def __init__(self, ticket_promedio: bool = False, indicadores_saldo: bool = False):
        self.ticket_promedio = ticket_promedio
        self.indicadores_saldo = indicadores_saldo

    def fit(self, X: pd.DataFrame, y=None):
        self._validar_entrada(X)
        self.saldo_maximo_ = float(X["total_revolving_bal"].max())
        self.columnas_entrada_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "saldo_maximo_")
        self._validar_entrada(X)
        salida = X.drop(columns=COLUMNAS_A_ELIMINAR).copy()
        if self.ticket_promedio:
            transacciones = salida["total_trans_ct"]
            # Protección para datos nuevos: sin transacciones no hay ticket; se deja en 0
            salida[COLUMNA_TICKET] = np.where(
                transacciones > 0, salida["total_trans_amt"] / transacciones.where(transacciones > 0, 1), 0.0)
        if self.indicadores_saldo:
            saldo = salida["total_revolving_bal"]
            salida[COLUMNA_SALDO_CERO] = (saldo == 0).astype(int)
            salida[COLUMNA_SALDO_MAXIMO] = (saldo >= self.saldo_maximo_).astype(int)
        return salida

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "saldo_maximo_")
        nombres = [c for c in self.columnas_entrada_ if c not in COLUMNAS_A_ELIMINAR]
        if self.ticket_promedio:
            nombres.append(COLUMNA_TICKET)
        if self.indicadores_saldo:
            nombres += [COLUMNA_SALDO_CERO, COLUMNA_SALDO_MAXIMO]
        return np.array(nombres, dtype=object)

    @staticmethod
    def _validar_entrada(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreadorVariables espera un DataFrame con los nombres de columna originales")
        requeridas = {"total_revolving_bal", "total_trans_amt", "total_trans_ct", *COLUMNAS_A_ELIMINAR}
        faltantes = requeridas - set(X.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas para crear variables: {sorted(faltantes)}")


def construir_codificador(escalar: bool, categoricas: list[str] | None = None) -> ColumnTransformer:
    """Codificación de categóricas y, si se pide, transformación de numéricas.

    - Numéricas: Yeo-Johnson con estandarización (media 0 y desviación 1) cuando `escalar`
      es True; sin cambios cuando es False (árboles y ensambles no lo necesitan).
    - Categóricas: One-Hot; las categorías con menos de MIN_FRECUENCIA_CATEGORIA clientes
      se agrupan en una columna de "infrecuentes"; una categoría nunca vista se envía a ese
      grupo si existe (o queda en ceros si no existe); en variables de dos valores se
      conserva una sola columna.

    `categoricas` indica cuáles se codifican; por defecto, las cinco. Con una lista vacía
    solo pasan las numéricas y las categóricas se descartan (así trabajan los modelos).
    """
    numericas = PowerTransformer(method="yeo-johnson", standardize=True) if escalar else "passthrough"
    categoricas = list(datos.COLUMNAS_CATEGORICAS if categoricas is None else categoricas)
    transformadores = [("numericas", numericas, make_column_selector(dtype_include="number"))]
    if categoricas:
        codificador_categoricas = OneHotEncoder(min_frequency=MIN_FRECUENCIA_CATEGORIA,
                                                handle_unknown="infrequent_if_exist",
                                                drop="if_binary", sparse_output=False)
        transformadores.append(("categoricas", codificador_categoricas, categoricas))
    return ColumnTransformer(transformers=transformadores, remainder="drop", verbose_feature_names_out=False)


def pasos_preparacion(escalar: bool, ticket_promedio: bool = False,
                      indicadores_saldo: bool = False, categoricas: list[str] | None = None) -> list[tuple]:
    """Pasos de la preparación como lista, para insertarlos directamente en el pipeline
    de cada modelo. imblearn no permite un Pipeline anidado antes de SMOTE, por eso los
    pasos van "planos": `variables`, `codificador`, `balanceo`, `modelo`.

    En GridSearchCV las candidatas se controlan con `variables__ticket_promedio` y
    `variables__indicadores_saldo`. `categoricas=[]` deja pasar solo las numéricas.
    """
    return [
        ("variables", CreadorVariables(ticket_promedio=ticket_promedio,
                                       indicadores_saldo=indicadores_saldo)),
        ("codificador", construir_codificador(escalar=escalar, categoricas=categoricas)),
    ]


def construir_preparacion(escalar: bool, ticket_promedio: bool = False,
                          indicadores_saldo: bool = False) -> Pipeline:
    """Preparación sola (variables + codificación de las cinco categóricas), útil para
    inspeccionarla y probarla. Los modelos usan los mismos pasos a través de `pasos_preparacion`."""
    return Pipeline(pasos_preparacion(escalar, ticket_promedio, indicadores_saldo))


def nombres_salida(preparacion: Pipeline) -> list[str]:
    """Nombres de las columnas que recibe el modelo (para la importancia de variables)."""
    return list(preparacion.named_steps["codificador"].get_feature_names_out())


def columnas_numericas_salida(preparacion: Pipeline) -> list[str]:
    """Nombres de salida que vienen de variables numéricas (no de la codificación One-Hot)."""
    nombres = nombres_salida(preparacion)
    return [n for n in nombres if not any(n.startswith(c + "_") for c in datos.COLUMNAS_CATEGORICAS)]


def verificar_preparacion(preparacion: Pipeline, X_train: pd.DataFrame,
                          X_test: pd.DataFrame) -> pd.DataFrame:
    """Chequeos visibles de una preparación ya ajustada con train (para el notebook).

    Revisa que no haya faltantes ni infinitos, que la columna redundante no llegue al
    modelo, que los nombres sean únicos y, si hay escalado, que train quede con media 0
    y desviación 1 mientras test no (señal de que el test no se usó para ajustar).
    """
    nombres = nombres_salida(preparacion)
    Z_train = pd.DataFrame(preparacion.transform(X_train), columns=nombres)
    Z_test = pd.DataFrame(preparacion.transform(X_test), columns=nombres)
    numericas = columnas_numericas_salida(preparacion)
    escala = preparacion.named_steps["codificador"].named_transformers_["numericas"]
    chequeos = [
        ("Sin faltantes ni infinitos en train", bool(np.isfinite(Z_train.values).all()), f"{Z_train.shape[0]:,} x {Z_train.shape[1]}"),
        ("Sin faltantes ni infinitos en test", bool(np.isfinite(Z_test.values).all()), f"{Z_test.shape[0]:,} x {Z_test.shape[1]}"),
        ("avg_open_to_buy no llega al modelo", "avg_open_to_buy" not in nombres, f"{len(nombres)} columnas de salida"),
        ("Nombres de columna únicos", len(nombres) == len(set(nombres)), ""),
    ]
    # Con "passthrough", sklearn guarda un transformador identidad, no el texto: se compara el tipo
    if isinstance(escala, PowerTransformer):
        media_train = Z_train[numericas].mean().abs().max()
        desv_train = (Z_train[numericas].std(ddof=0) - 1).abs().max()
        media_test = Z_test[numericas].mean().abs().max()
        chequeos += [
            ("Train estandarizado (media 0)", bool(media_train < 1e-6), f"máx |media| = {media_train:.1e}"),
            ("Train estandarizado (desviación 1)", bool(desv_train < 1e-6), f"máx |desv - 1| = {desv_train:.1e}"),
            ("Test transformado con parámetros de train", bool(media_test > 1e-3), f"máx |media test| = {media_test:.3f} (distinto de 0)"),
        ]
    return pd.DataFrame(chequeos, columns=["chequeo", "cumple", "detalle"])


def resumen_preparacion(preparacion: Pipeline) -> dict:
    """Qué produjo la preparación ya ajustada con train: columnas de salida por origen,
    categorías agrupadas como infrecuentes y el saldo máximo aprendido de train."""
    nombres = nombres_salida(preparacion)
    numericas = columnas_numericas_salida(preparacion)
    candidatas = [COLUMNA_TICKET, COLUMNA_SALDO_CERO, COLUMNA_SALDO_MAXIMO]
    origen = pd.Series(["One-Hot" if n not in numericas else ("candidata" if n in candidatas else "numérica original")
                        for n in nombres], index=nombres, name="origen")
    codificador = preparacion.named_steps["codificador"].named_transformers_["categoricas"]
    infrecuentes = {col: list(cats) for col, cats in zip(datos.COLUMNAS_CATEGORICAS, codificador.infrequent_categories_)
                    if cats is not None}
    return {"columnas_por_origen": origen.value_counts().rename("columnas").to_frame(),
            "infrecuentes": infrecuentes,
            "saldo_maximo": preparacion.named_steps["variables"].saldo_maximo_}
