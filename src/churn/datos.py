"""Carga, validación, limpieza y partición de los datos del caso de churn.

Responsabilidad de este módulo: entregar datos confiables y una partición fija y reproducible.
Nada de lo que hay aquí aprende de los datos (no imputa, no escala, no balancea clases):
esas transformaciones van dentro del Pipeline de modelado y se ajustan solo con train,
para evitar fuga de información (que el modelo "vea" el test antes de ser evaluado).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

# ---------------------------------------------------------------------------
# Esquema esperado del archivo del caso (Tabla 1 del PDF oficial)
# ---------------------------------------------------------------------------
COLUMNA_ID = "clientnum"
COLUMNA_CHURN = "churn"  # objetivo binario que creamos: 1 = se fue, 0 = se quedó

COLUMNAS_ESPERADAS = [
    "clientnum", "attrition_flag", "customer_age", "gender", "dependent_count",
    "education_level", "marital_status", "income_category", "card_category",
    "months_on_book", "total_relationship_count", "months_inactive_12_mon",
    "contacts_count_12_mon", "credit_limit", "total_revolving_bal", "avg_open_to_buy",
    "total_amt_chng_q4_q1", "total_trans_amt", "total_trans_ct", "total_ct_chng_q4_q1",
    "avg_utilization_ratio",
]

CATEGORIAS_ESPERADAS = {
    "attrition_flag": {"Existing Customer", "Attrited Customer"},
    "gender": {"F", "M"},
    "education_level": {"Uneducated", "High School", "College", "Graduate",
                        "Post-Graduate", "Doctorate", "Unknown"},
    "marital_status": {"Single", "Married", "Divorced", "Unknown"},
    "income_category": {"Less than $40K", "$40K - $60K", "$60K - $80K",
                        "$80K - $120K", "$120K +", "Unknown"},
    "card_category": {"Blue", "Silver", "Gold", "Platinum"},
}
COLUMNAS_CATEGORICAS = [c for c in CATEGORIAS_ESPERADAS if c != config.OBJETIVO]

# Orden lógico de las categorías, solo para mostrarlas en tablas y gráficos.
# `Unknown` va al final porque no ocupa un lugar en la escala.
ORDEN_CATEGORIAS = {
    "gender": ["F", "M"],
    "education_level": ["Uneducated", "High School", "College", "Graduate",
                        "Post-Graduate", "Doctorate", "Unknown"],
    "marital_status": ["Single", "Married", "Divorced", "Unknown"],
    "income_category": ["Less than $40K", "$40K - $60K", "$60K - $80K",
                        "$80K - $120K", "$120K +", "Unknown"],
    "card_category": ["Blue", "Silver", "Gold", "Platinum"],
}

# Rangos de negocio plausibles (mínimo, máximo); None significa sin límite
RANGOS_ESPERADOS = {
    "customer_age": (18, 100),
    "dependent_count": (0, None),
    "months_on_book": (0, None),
    "total_relationship_count": (1, None),
    "months_inactive_12_mon": (0, 12),
    "contacts_count_12_mon": (0, 12),
    "credit_limit": (0, None),
    "total_revolving_bal": (0, None),
    "avg_open_to_buy": (0, None),
    "total_amt_chng_q4_q1": (0, None),
    "total_trans_amt": (0, None),
    "total_trans_ct": (0, None),
    "total_ct_chng_q4_q1": (0, None),
    "avg_utilization_ratio": (0, 1),
}

# Tolerancias para las identidades contables (el archivo redondea la utilización a 3 decimales)
TOLERANCIA_CUPO = 1e-6
TOLERANCIA_UTILIZACION = 1e-3

NOMBRE_PROCESADO = "bank_churn_limpio.csv"
NOMBRE_PARTICION = "particion.csv"


class ErrorDeDatos(ValueError):
    """Se lanza cuando el archivo no cumple el esquema o las reglas de negocio esperadas."""


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
def cargar_datos(ruta: Path = config.RUTA_DATOS_CRUDOS) -> pd.DataFrame:
    """Lee el Excel crudo del caso tal como viene, sin modificarlo."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta.name}")
    return pd.read_excel(ruta)


# ---------------------------------------------------------------------------
# 2. Validación
# ---------------------------------------------------------------------------
def validar_datos(df: pd.DataFrame, estricto: bool = True) -> pd.DataFrame:
    """Revisa esquema, calidad y coherencia contable del archivo.

    Devuelve una tabla con un chequeo por fila (para mostrarla en el notebook).
    Si `estricto` es True y algún chequeo falla, detiene el proceso con ErrorDeDatos,
    porque modelar sobre datos rotos daría conclusiones falsas sin avisar.
    """
    chequeos = []

    def registrar(nombre: str, cumple: bool, detalle: str) -> None:
        chequeos.append({"chequeo": nombre, "cumple": bool(cumple), "detalle": detalle})

    faltantes_col = sorted(set(COLUMNAS_ESPERADAS) - set(df.columns))
    sobrantes_col = sorted(set(df.columns) - set(COLUMNAS_ESPERADAS))
    registrar("Columnas esperadas", not faltantes_col and not sobrantes_col,
              f"faltan: {faltantes_col or 'ninguna'}; sobran: {sobrantes_col or 'ninguna'}")
    if faltantes_col:
        # Sin las columnas no tiene sentido seguir revisando: se reporta y se corta aquí
        return _cerrar_validacion(chequeos, estricto)

    n_faltantes = int(df.isna().sum().sum())
    registrar("Sin valores faltantes", n_faltantes == 0, f"{n_faltantes} celdas vacías")

    n_dup = int(df.duplicated().sum())
    registrar("Sin filas duplicadas", n_dup == 0, f"{n_dup} filas repetidas")

    n_id_dup = int(df[COLUMNA_ID].duplicated().sum())
    registrar("Identificador único", n_id_dup == 0, f"{n_id_dup} clientnum repetidos")

    for col, permitidas in CATEGORIAS_ESPERADAS.items():
        nuevas = sorted(set(df[col].dropna().astype(str).str.strip()) - permitidas)
        registrar(f"Categorías válidas en {col}", not nuevas,
                  f"no esperadas: {nuevas or 'ninguna'}")

    for col, (minimo, maximo) in RANGOS_ESPERADOS.items():
        serie = df[col]
        fuera = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera |= serie < minimo
        if maximo is not None:
            fuera |= serie > maximo
        registrar(f"Rango de {col}", not fuera.any(),
                  f"{int(fuera.sum())} valores fuera de [{minimo}, {maximo}]")

    # Identidad contable: cupo total = saldo rotativo + cupo disponible
    dif_cupo = (df["credit_limit"] - df["total_revolving_bal"] - df["avg_open_to_buy"]).abs()
    registrar("credit_limit = total_revolving_bal + avg_open_to_buy",
              (dif_cupo <= TOLERANCIA_CUPO).all(),
              f"diferencia máxima {dif_cupo.max():.2e}")

    # Identidad de uso: utilización = saldo rotativo / cupo total
    utilizacion_calculada = df["total_revolving_bal"] / df["credit_limit"]
    dif_util = (utilizacion_calculada - df["avg_utilization_ratio"]).abs()
    registrar("avg_utilization_ratio = total_revolving_bal / credit_limit",
              (dif_util <= TOLERANCIA_UTILIZACION + 1e-12).all(),
              f"diferencia máxima {dif_util.max():.4f}")

    return _cerrar_validacion(chequeos, estricto)


def _cerrar_validacion(chequeos: list[dict], estricto: bool) -> pd.DataFrame:
    """Arma la tabla de chequeos y, si se pide, detiene el proceso ante cualquier falla."""
    reporte = pd.DataFrame(chequeos)
    fallas = reporte.loc[~reporte["cumple"], "chequeo"].tolist()
    if estricto and fallas:
        raise ErrorDeDatos(f"Los datos no pasaron la validación: {fallas}")
    return reporte


def resumen_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Vista general por variable: tipo, valores distintos, faltantes y un ejemplo."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "valores_distintos": df.nunique(),
        "faltantes": df.isna().sum(),
        "ejemplo": df.iloc[0],
    })


def balance_objetivo(df_limpio: pd.DataFrame) -> pd.DataFrame:
    """Conteo y proporción de cada clase del objetivo, con su nombre de negocio."""
    conteo = df_limpio[COLUMNA_CHURN].value_counts().sort_index()
    tabla = pd.DataFrame({
        "clase": ["Se queda (Existing Customer)", "Se va (Attrited Customer)"],
        "clientes": conteo.values,
        "proporcion": (conteo / conteo.sum()).round(4).values,
    }, index=conteo.index)
    tabla.index.name = COLUMNA_CHURN
    return tabla


# ---------------------------------------------------------------------------
# 3. Limpieza (solo operaciones que no aprenden de los datos)
# ---------------------------------------------------------------------------
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Deja los datos en un formato estable para el análisis.

    - Quita espacios sobrantes en los textos.
    - Crea el objetivo binario `churn` (1 = Attrited Customer) y elimina el texto original,
      para que nunca quede por error como variable explicativa.
    - Usa `clientnum` como índice: no es una variable del modelo, pero hace falta para
      entregar la lista priorizada de clientes.
    No elimina ni transforma variables explicativas: esas decisiones salen del EDA y
    viven en el módulo de features.
    """
    limpio = df.copy()
    for col in CATEGORIAS_ESPERADAS:
        limpio[col] = limpio[col].astype(str).str.strip()

    limpio[COLUMNA_CHURN] = (limpio[config.OBJETIVO] == config.CLASE_POSITIVA).astype(int)
    limpio = limpio.drop(columns=config.OBJETIVO).set_index(COLUMNA_ID)
    return limpio


# ---------------------------------------------------------------------------
# 4. Partición
# ---------------------------------------------------------------------------
@dataclass
class Particion:
    """Guarda la división train/test. El test se reserva y se usa una sola vez al final."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def resumen(self) -> pd.DataFrame:
        """Tabla de tamaños y tasa de churn por conjunto (para comprobar la estratificación)."""
        filas = []
        for nombre, y in [("train", self.y_train), ("test", self.y_test)]:
            filas.append({"conjunto": nombre, "clientes": len(y),
                          "churners": int(y.sum()), "tasa_churn": round(float(y.mean()), 4)})
        return pd.DataFrame(filas).set_index("conjunto")


def particionar(df_limpio: pd.DataFrame, test_size: float = config.TEST_SIZE,
                semilla: int = config.SEMILLA) -> Particion:
    """División estratificada: train y test conservan la misma proporción de churners (16 %)."""
    X = df_limpio.drop(columns=COLUMNA_CHURN)
    y = df_limpio[COLUMNA_CHURN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=semilla)
    return Particion(X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# 5. Guardado y recarga de los datos procesados
# ---------------------------------------------------------------------------
def guardar_procesado(df_limpio: pd.DataFrame, particion: Particion,
                      carpeta: Path = config.RUTA_DATOS_PROCESADOS) -> dict[str, Path]:
    """Guarda el dataset limpio y la asignación de cada cliente a train o test.

    Guardar la partición permite que el notebook, PyCaret y cualquier script usen
    exactamente los mismos clientes de prueba.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_datos = carpeta / NOMBRE_PROCESADO
    ruta_particion = carpeta / NOMBRE_PARTICION

    df_limpio.to_csv(ruta_datos)
    asignacion = pd.concat([
        pd.DataFrame({COLUMNA_ID: particion.X_train.index, "conjunto": "train"}),
        pd.DataFrame({COLUMNA_ID: particion.X_test.index, "conjunto": "test"}),
    ])  # sin reordenar: el orden de las filas define los pliegues de la validación cruzada
    asignacion.to_csv(ruta_particion, index=False)
    return {"datos": ruta_datos, "particion": ruta_particion}


def cargar_procesado(carpeta: Path = config.RUTA_DATOS_PROCESADOS) -> tuple[pd.DataFrame, Particion]:
    """Recarga el dataset limpio y reconstruye la misma partición guardada, con los mismos
    clientes y en el mismo orden (el orden importa: con otro orden cambian los pliegues)."""
    carpeta = Path(carpeta)
    df_limpio = pd.read_csv(carpeta / NOMBRE_PROCESADO, index_col=COLUMNA_ID)
    asignacion = pd.read_csv(carpeta / NOMBRE_PARTICION, index_col=COLUMNA_ID)["conjunto"]

    X = df_limpio.drop(columns=COLUMNA_CHURN)
    y = df_limpio[COLUMNA_CHURN]
    ids_train = asignacion.index[asignacion == "train"]
    ids_test = asignacion.index[asignacion == "test"]
    return df_limpio, Particion(X.loc[ids_train], X.loc[ids_test], y.loc[ids_train], y.loc[ids_test])


# ---------------------------------------------------------------------------
# Orquestador del módulo
# ---------------------------------------------------------------------------
def preparar_datos(ruta: Path = config.RUTA_DATOS_CRUDOS, guardar: bool = True,
                   carpeta_salida: Path = config.RUTA_DATOS_PROCESADOS):
    """Ejecuta carga, validación, limpieza y partición en orden, y guarda el resultado.

    Devuelve (df_limpio, particion, reporte_validacion).
    """
    crudo = cargar_datos(ruta)
    reporte = validar_datos(crudo, estricto=True)
    limpio = limpiar_datos(crudo)
    particion = particionar(limpio)
    if guardar:
        guardar_procesado(limpio, particion, carpeta_salida)
    return limpio, particion, reporte
