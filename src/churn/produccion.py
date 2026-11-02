"""Modelo viable: guardar el modelo, cargarlo fuera del notebook y puntuar clientes nuevos.

Se guardan dos modelos con papeles distintos, y el archivo de cada uno lo dice:

- **Modelo oficial** (`modelo_oficial.joblib`): el LightGBM completo, elegido en la Fase 4
  y evaluado una sola vez en test (Fase 5, 5.1). Sus métricas de test son los resultados
  oficiales del proyecto. Se conserva como referencia.
- **Candidato de producción** (`modelo_produccion.joblib`): el mismo LightGBM (mismos
  hiperparámetros, misma preparación) sin `gender` ni `contacts_count_12_mon`, por los
  riesgos de sesgo y de fuga revisados en la Fase 5 (5.7). Su evidencia es la validación
  cruzada en train; **no se evaluó en test** y sus métricas no deben presentarse como tales.

Ambos se entrenan con los mismos 8.101 clientes de train. El candidato no se reentrena con
train + test: así su procedimiento de entrenamiento es exactamente el que respalda la
validación cruzada y el test queda intacto.

Uso fuera del notebook (desde la raíz del repositorio, con el entorno activado):

    python -m churn.produccion entrenar
    python -m churn.produccion puntuar clientes.xlsx lista.csv --capacidad 0.2

o desde Python:

    from churn import produccion
    modelo = produccion.cargar_modelo()            # candidato de producción por defecto
    lista = produccion.puntuar_clientes(clientes, modelo, capacidad=0.2)
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import re
import warnings
from datetime import date
from pathlib import Path

import imblearn
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.preprocessing import PowerTransformer

from churn import config, datos, evaluacion, features, modelos, negocio

NOMBRE_MODELO_OFICIAL = "LightGBM"
VARIABLES_EXCLUIDAS_PRODUCCION = ["gender", "contacts_count_12_mon"]
RUTA_MODELO_OFICIAL = config.RUTA_MODELOS / "modelo_oficial.joblib"
RUTA_MODELO_PRODUCCION = config.RUTA_MODELOS / "modelo_produccion.joblib"

# Hiperparámetros que eligió la búsqueda de la Fase 4 para el LightGBM oficial (ronda 2).
# Se fijan aquí para poder reconstruir el modelo sin repetir los 50 minutos de búsqueda;
# una prueba automática comprueba que coinciden con la caché cuando existe.
HIPERPARAMETROS_OFICIALES = {
    "variables__ticket_promedio": False,
    "variables__indicadores_saldo": False,
    "balanceo": "passthrough",
    "modelo__learning_rate": 0.03,
    "modelo__n_estimators": 1000,
    "modelo__num_leaves": 31,
}

COLUMNAS_OBJETIVO = [config.OBJETIVO, datos.COLUMNA_CHURN]  # se ignoran si vienen en la entrada
CLAVES_ARTEFACTO = {"pipeline", "nombre", "rol", "evaluado_en_test", "columnas_entrada",
                    "variables_excluidas", "clase_positiva", "codigo_clase_positiva",
                    "hiperparametros", "rangos_train", "metricas", "huella_datos_train",
                    "versiones", "semilla", "fecha_entrenamiento"}


# ---------------------------------------------------------------------------
# 1. Construcción de los pipelines (sin ajustar)
# ---------------------------------------------------------------------------
def _modelo_catalogo(nombre: str = NOMBRE_MODELO_OFICIAL) -> modelos.Modelo:
    return next(m for m in modelos.catalogo_modelos() if m.nombre == nombre)


def construir_pipeline_oficial():
    """Pipeline del modelo oficial sin ajustar: el del catálogo con los hiperparámetros elegidos."""
    return _modelo_catalogo().pipeline().set_params(**HIPERPARAMETROS_OFICIALES)


def construir_pipeline_produccion(pipeline_referencia=None, excluidas=VARIABLES_EXCLUIDAS_PRODUCCION):
    """Mismo pipeline que el oficial, sin las variables excluidas.

    Se clona el pipeline de referencia (mismos pasos e hiperparámetros) y solo se cambia el
    codificador para que no espere las categóricas excluidas. Las numéricas excluidas no
    necesitan cambios: el codificador toma las numéricas que encuentra, y la función
    `columnas_entrada` ya no las pide."""
    pipeline = clone(pipeline_referencia if pipeline_referencia is not None else construir_pipeline_oficial())
    numericas = pipeline.named_steps["codificador"].transformers[0][1]
    escalar = isinstance(numericas, PowerTransformer)
    categoricas = [c for c in datos.COLUMNAS_CATEGORICAS if c not in excluidas]
    return pipeline.set_params(codificador=features.construir_codificador(escalar=escalar, categoricas=categoricas))


def columnas_entrada(excluidas=()) -> list[str]:
    """Columnas que recibe el modelo, en el orden del archivo original (sin id ni objetivo)."""
    return [c for c in datos.COLUMNAS_ESPERADAS
            if c not in (datos.COLUMNA_ID, config.OBJETIVO) and c not in excluidas]


# ---------------------------------------------------------------------------
# 2. Validación de clientes nuevos
# ---------------------------------------------------------------------------
def preparar_clientes(clientes: pd.DataFrame, columnas: list[str], rangos_train: dict | None = None,
                      estricto: bool = True) -> tuple[pd.DataFrame, dict]:
    """Valida y ordena las columnas de clientes nuevos antes de puntuarlos.

    Acepta el formato del archivo del banco: `clientnum` como columna o como índice, y
    columnas de más (el objetivo o las variables excluidas), que se ignoran.
    Detiene el proceso (ErrorDeDatos) si faltan columnas, hay vacíos, identificadores
    repetidos, textos en columnas numéricas, valores imposibles según las reglas de negocio
    o categorías desconocidas. Con `estricto=False` estas últimas solo generan una advertencia:
    el codificador las envía al grupo de infrecuentes si la variable lo tiene (`card_category`)
    y, si no, las deja en ceros (como si el cliente no tuviera ninguna categoría conocida). Los valores fuera del rango visto en train
    son posibles, pero el modelo extrapola: se avisan con una advertencia.

    Devuelve las columnas del modelo en el orden de entrenamiento y un reporte.
    """
    if not isinstance(clientes, pd.DataFrame):
        raise TypeError("Los clientes deben venir en un DataFrame de pandas")
    if clientes.empty:
        raise datos.ErrorDeDatos("No hay clientes para puntuar")
    tabla = clientes.copy()
    if datos.COLUMNA_ID in tabla.columns:
        tabla = tabla.set_index(datos.COLUMNA_ID)
    if tabla.index.duplicated().any():
        repetidos = tabla.index[tabla.index.duplicated()].unique().tolist()[:5]
        raise datos.ErrorDeDatos(f"Hay identificadores de cliente repetidos, por ejemplo: {repetidos}")

    faltantes = [c for c in columnas if c not in tabla.columns]
    if faltantes:
        raise datos.ErrorDeDatos(f"Faltan columnas que el modelo necesita: {faltantes}")
    ignoradas = [c for c in tabla.columns if c not in columnas]
    X = tabla[columnas].copy()

    vacios = X.columns[X.isna().any()].tolist()
    if vacios:
        raise datos.ErrorDeDatos(f"Hay valores vacíos en: {vacios}")

    reporte = {"clientes": len(X), "columnas_ignoradas": ignoradas,
               "categorias_desconocidas": {}, "fuera_de_rango_train": {}}
    for col in columnas:
        if col in datos.CATEGORIAS_ESPERADAS:
            X[col] = X[col].astype(str).str.strip()  # igual que datos.limpiar_datos
            nuevas = sorted(set(X[col]) - datos.CATEGORIAS_ESPERADAS[col])
            if nuevas:
                reporte["categorias_desconocidas"][col] = nuevas
        elif not pd.api.types.is_numeric_dtype(X[col]):
            raise datos.ErrorDeDatos(f"La columna {col} debe ser numérica")

    imposibles = {}
    for col, (minimo, maximo) in datos.RANGOS_ESPERADOS.items():
        if col not in X.columns:
            continue
        fuera = ((X[col] < minimo) if minimo is not None else False) | ((X[col] > maximo) if maximo is not None else False)
        if np.any(fuera):
            imposibles[col] = int(np.sum(fuera))
    if imposibles:
        raise datos.ErrorDeDatos(f"Valores fuera de los rangos de negocio permitidos: {imposibles}")

    if reporte["categorias_desconocidas"]:
        mensaje = f"Categorías no vistas en el entrenamiento: {reporte['categorias_desconocidas']}"
        if estricto:
            raise datos.ErrorDeDatos(mensaje)
        warnings.warn(mensaje + ". Van al grupo de infrecuentes si existe; si no, quedan en ceros.", stacklevel=2)

    for col, (minimo, maximo) in (rangos_train or {}).items():
        n = int(((X[col] < minimo) | (X[col] > maximo)).sum())
        if n:
            reporte["fuera_de_rango_train"][col] = n
    if reporte["fuera_de_rango_train"]:
        warnings.warn(f"Clientes con valores fuera del rango de train (el modelo extrapola): "
                      f"{reporte['fuera_de_rango_train']}", stacklevel=2)
    return X, reporte


# ---------------------------------------------------------------------------
# 3. Entrenamiento y guardado
# ---------------------------------------------------------------------------
def huella_datos(X: pd.DataFrame, y: pd.Series) -> str:
    """Huella SHA-256 de los datos de entrenamiento (para saber con qué datos se entrenó).
    Se calcula sobre todas las columnas explicativas, así el oficial y el candidato, que usan
    los mismos clientes, tienen la misma huella aunque el candidato use menos columnas."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return h.hexdigest()


def versiones() -> dict:
    """Versiones con las que se entrenó: un pickle de scikit-learn o LightGBM no está
    garantizado entre versiones distintas."""
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "imbalanced-learn": imblearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


def crear_artefacto(pipeline_sin_ajustar, X_train: pd.DataFrame, y_train: pd.Series, *, nombre: str, rol: str,
                    evaluado_en_test: bool, excluidas=(), metricas: dict | None = None) -> dict:
    """Ajusta el pipeline con train y lo empaqueta con todo lo necesario para usarlo y auditarlo."""
    if rol not in ("oficial", "candidato_produccion"):
        raise ValueError("rol debe ser 'oficial' o 'candidato_produccion'")
    columnas = columnas_entrada(excluidas)
    X = X_train[columnas]
    pipeline = clone(pipeline_sin_ajustar).fit(X, y_train)
    numericas = [c for c in columnas if c not in datos.CATEGORIAS_ESPERADAS]
    return {
        "pipeline": pipeline, "nombre": nombre, "rol": rol, "evaluado_en_test": evaluado_en_test,
        "columnas_entrada": columnas, "variables_excluidas": list(excluidas),
        "clase_positiva": config.CLASE_POSITIVA, "codigo_clase_positiva": 1,
        "hiperparametros": {k: v for k, v in pipeline.get_params().items()
                            if k.startswith("modelo__") or k in HIPERPARAMETROS_OFICIALES},
        "rangos_train": {c: (float(X[c].min()), float(X[c].max())) for c in numericas},
        "metricas": metricas or {}, "huella_datos_train": huella_datos(X_train[columnas_entrada()], y_train),
        "versiones": versiones(), "semilla": config.SEMILLA,
        "fecha_entrenamiento": date.today().isoformat(),
    }


def guardar_modelo(artefacto: dict, ruta: Path) -> Path:
    """Guarda el artefacto con joblib (crea la carpeta si no existe)."""
    _validar_artefacto(artefacto)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, ruta)
    return ruta


def cargar_modelo(ruta: Path = RUTA_MODELO_PRODUCCION) -> dict:
    """Carga un artefacto guardado y revisa que esté completo. Avisa si las versiones de las
    librerías no son las del entrenamiento."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta.name}. Genérelo con: python -m churn.produccion entrenar")
    artefacto = joblib.load(ruta)
    _validar_artefacto(artefacto)
    distintas = {k: (v, versiones()[k]) for k, v in artefacto["versiones"].items()
                 if k in ("scikit-learn", "lightgbm", "imbalanced-learn") and versiones()[k] != v}
    if distintas:
        warnings.warn(f"Versiones distintas a las del entrenamiento (guardada, actual): {distintas}", stacklevel=2)
    return artefacto


def _validar_artefacto(artefacto: dict) -> None:
    faltan = CLAVES_ARTEFACTO - set(artefacto)
    if faltan:
        raise ValueError(f"El artefacto del modelo está incompleto; faltan: {sorted(faltan)}")
    if artefacto["clase_positiva"] != config.CLASE_POSITIVA or artefacto["codigo_clase_positiva"] != 1:
        raise ValueError("El artefacto no usa la clase positiva del proyecto (Attrited Customer = 1)")


# ---------------------------------------------------------------------------
# 4. Puntuación
# ---------------------------------------------------------------------------
def indice_clase_positiva(pipeline) -> int:
    """Posición de la clase churn (1 = Attrited Customer) en `predict_proba`. No se supone que
    sea la segunda columna: se busca en `classes_`."""
    clases = list(pipeline.classes_)
    if sorted(clases) != [0, 1]:
        raise ValueError(f"El modelo debe ser binario con clases 0 y 1; tiene {clases}")
    return clases.index(1)


def puntuar_clientes(clientes: pd.DataFrame, modelo: dict | None = None, capacidad: float | None = None,
                     estricto: bool = True) -> pd.DataFrame:
    """Lista de clientes priorizada por probabilidad de churn, de mayor a menor riesgo.

    - `modelo`: artefacto de `cargar_modelo` (por defecto, el candidato de producción).
    - `capacidad`: fracción de la base que el área comercial puede contactar (por ejemplo
      0,2); marca a los clientes que entran en ese escenario, con el mismo redondeo de la
      Fase 5 (5.4 y 5.5). Si es None no marca a nadie.

    Aplica exactamente la preparación del entrenamiento, porque el pipeline guardado la
    contiene. Los empates de probabilidad conservan el orden de entrada (orden estable), así
    que la misma entrada siempre produce la misma lista.
    """
    modelo = modelo if modelo is not None else cargar_modelo()
    _validar_artefacto(modelo)
    X, reporte = preparar_clientes(clientes, modelo["columnas_entrada"], modelo["rangos_train"], estricto)
    pipeline = modelo["pipeline"]
    probabilidad = pipeline.predict_proba(X)[:, indice_clase_positiva(pipeline)]
    if not np.all(np.isfinite(probabilidad)) or probabilidad.min() < 0 or probabilidad.max() > 1:
        raise ValueError("El modelo produjo probabilidades fuera de [0, 1]")

    orden = negocio.orden_por_riesgo(probabilidad)
    lista = pd.DataFrame({"prob_churn": probabilidad[orden]}, index=X.index[orden])
    lista.index.name = datos.COLUMNA_ID
    lista.insert(0, "prioridad", np.arange(1, len(lista) + 1))
    lista["percentil_riesgo"] = lista["prioridad"] / len(lista)  # 0,05 = dentro del 5 % de mayor riesgo
    if capacidad is not None:
        if not 0 < capacidad <= 1:
            raise ValueError("capacidad debe estar entre 0 y 1")
        lista["contactar"] = lista["prioridad"] <= negocio.contactados(len(lista), capacidad)
    lista.attrs.update({"modelo": modelo["nombre"], "rol": modelo["rol"], **reporte})
    return lista


# ---------------------------------------------------------------------------
# 5. Generación de los dos modelos (la usan el notebook y la línea de comandos)
# ---------------------------------------------------------------------------
def evidencia_cv_produccion(pipeline_oficial, X_train: pd.DataFrame, y_train: pd.Series, cv=None,
                            n_jobs: int = -1) -> tuple[dict, dict]:
    """Validación cruzada del candidato de producción en train, pareada con el oficial (mismos
    pliegues). Es la única evidencia de desempeño del candidato: no se evalúa en test."""
    cv = cv or modelos.validacion_cruzada()
    pliegues = list(cv.split(X_train, y_train))
    produccion = construir_pipeline_produccion(pipeline_oficial)
    oof = {
        "oficial": evaluacion.predicciones_oof(clone(pipeline_oficial), X_train[columnas_entrada()], y_train, pliegues, n_jobs),
        "produccion": evaluacion.predicciones_oof(produccion, X_train[columnas_entrada(VARIABLES_EXCLUIDAS_PRODUCCION)],
                                                  y_train, pliegues, n_jobs),
    }
    auc = {k: evaluacion.auc_por_pliegue(y_train, v, pliegues) for k, v in oof.items()}
    diferencia = auc["produccion"] - auc["oficial"]
    escenarios = negocio.tabla_escenarios(y_train, oof["produccion"])
    metricas = {
        "origen": "validación cruzada de 5 pliegues en train (predicciones fuera de pliegue); no evaluado en test",
        "auc_cv": float(auc["produccion"].mean()), "auc_cv_desv": float(auc["produccion"].std()),
        "auc_cv_oficial": float(auc["oficial"].mean()),
        "diferencia_auc_vs_oficial": float(diferencia.mean()), "diferencia_desv": float(diferencia.std()),
        "pliegues_por_debajo": int((diferencia < 0).sum()),
        "ap_cv_oof": float(evaluacion.calcular_metricas(y_train, oof["produccion"])["ap"]),
        **{f"churners_capturados_{round(100 * k)}": int(escenarios.loc[k, "churners_capturados"]) for k in escenarios.index},
        **{f"captura_{round(100 * k)}": float(escenarios.loc[k, "captura"]) for k in escenarios.index},
    }
    return metricas, oof


def metricas_test_oficial(ruta: Path = config.RUTA_TABLAS / "test_metricas_umbral_05.csv") -> dict:
    """Métricas oficiales del test (Fase 5, 5.1) leídas de la tabla que guardó el notebook.
    No se vuelve a predecir el test."""
    if not Path(ruta).exists():
        return {"origen": "no disponible: ejecute el notebook para generar las tablas de la Fase 5"}
    fila = pd.read_csv(ruta, index_col=0).loc[NOMBRE_MODELO_OFICIAL]
    return {"origen": "evaluación única en test, Fase 5 (5.1) del notebook (2.026 clientes)",
            "auc_test": float(fila["auc"]), "ap_test": float(fila["ap"]), "brier_test": float(fila["brier"])}


def generar_modelos(X_train: pd.DataFrame, y_train: pd.Series, pipeline_oficial=None,
                    metricas_oficial: dict | None = None, metricas_produccion: dict | None = None,
                    carpeta: Path = config.RUTA_MODELOS, n_jobs: int = -1) -> dict[str, dict]:
    """Entrena, documenta y guarda el modelo oficial y el candidato de producción.

    Si no se entregan las métricas, se leen las del test de la Fase 5 (5.1) (oficial) y se
    calcula la validación cruzada del candidato (unos segundos)."""
    pipeline_oficial = clone(pipeline_oficial) if pipeline_oficial is not None else construir_pipeline_oficial()
    if metricas_produccion is None:
        metricas_produccion, _ = evidencia_cv_produccion(pipeline_oficial, X_train, y_train, n_jobs=n_jobs)
    artefactos = {
        "oficial": crear_artefacto(pipeline_oficial, X_train, y_train, nombre="LightGBM completo (modelo oficial)",
                                   rol="oficial", evaluado_en_test=True,
                                   metricas=metricas_oficial or metricas_test_oficial()),
        "produccion": crear_artefacto(construir_pipeline_produccion(pipeline_oficial), X_train, y_train,
                                      nombre="LightGBM sin gender ni contacts_count_12_mon (candidato de producción)",
                                      rol="candidato_produccion", evaluado_en_test=False,
                                      excluidas=VARIABLES_EXCLUIDAS_PRODUCCION, metricas=metricas_produccion),
    }
    carpeta = Path(carpeta)
    guardar_modelo(artefactos["oficial"], carpeta / RUTA_MODELO_OFICIAL.name)
    guardar_modelo(artefactos["produccion"], carpeta / RUTA_MODELO_PRODUCCION.name)
    return artefactos


def ficha(artefacto: dict) -> pd.Series:
    """Resumen legible de un artefacto (sin el pipeline) para mostrarlo o documentarlo."""
    datos_ficha = {k: artefacto[k] for k in ("nombre", "rol", "evaluado_en_test", "variables_excluidas",
                                              "clase_positiva", "semilla", "fecha_entrenamiento")}
    datos_ficha["n_columnas_entrada"] = len(artefacto["columnas_entrada"])
    datos_ficha["huella_datos_train"] = artefacto["huella_datos_train"][:12]
    datos_ficha.update({f"metrica__{k}": v for k, v in artefacto["metricas"].items()})
    datos_ficha.update({f"version__{k}": v for k, v in artefacto["versiones"].items()})
    return pd.Series(datos_ficha, name=artefacto["rol"])


# ---------------------------------------------------------------------------
# 6. Línea de comandos
# ---------------------------------------------------------------------------
def _leer_tabla(ruta: Path) -> pd.DataFrame:
    ruta = Path(ruta)
    return pd.read_excel(ruta) if ruta.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(ruta)


def main(argumentos: list[str] | None = None) -> None:
    lector = argparse.ArgumentParser(prog="python -m churn.produccion", description=__doc__.splitlines()[0])
    acciones = lector.add_subparsers(dest="accion", required=True)
    acciones.add_parser("entrenar", help="Entrena y guarda el modelo oficial y el candidato de producción")
    puntuar = acciones.add_parser("puntuar", help="Puntúa un archivo de clientes y guarda la lista priorizada")
    puntuar.add_argument("entrada", help="Archivo .xlsx o .csv con los clientes")
    puntuar.add_argument("salida", help="Archivo .csv donde se guarda la lista")
    puntuar.add_argument("--capacidad", type=float, default=None, help="Fracción a contactar, por ejemplo 0.2")
    puntuar.add_argument("--modelo", default=str(RUTA_MODELO_PRODUCCION), help="Ruta del artefacto del modelo")
    args = lector.parse_args(argumentos)

    if args.accion == "entrenar":
        _, particion, _ = datos.preparar_datos(guardar=False)
        artefactos = generar_modelos(particion.X_train, particion.y_train)
        for rol, artefacto in artefactos.items():
            print(f"{rol}: {artefacto['nombre']} | métricas: {artefacto['metricas'].get('origen')}")
    else:
        lista = puntuar_clientes(_leer_tabla(args.entrada), cargar_modelo(Path(args.modelo)), capacidad=args.capacidad)
        lista.to_csv(args.salida)
        print(f"{len(lista)} clientes puntuados con {lista.attrs['modelo']}; lista guardada en {Path(args.salida).name}")



def parametros_comparables(parametros: dict) -> dict:
    """Parámetros de un pipeline como texto sin direcciones de memoria, para compararlos entre objetos."""
    return {k: re.sub(r" at 0x[0-9a-fA-F]+", "", repr(v)) for k, v in parametros.items()}


def tabla_evidencia_cv(metricas: dict, ap_oficial: float, escenarios_cv: pd.DataFrame) -> pd.DataFrame:
    """Evidencia del candidato de producción frente al oficial, ambos por validación cruzada en train."""
    def valor(modelo, columna):
        return escenarios_cv.loc[(modelo, 0.2), columna]  # una celda, para conservar su tipo (entero)
    return pd.DataFrame({
        "auc_cv": [metricas["auc_cv_oficial"], metricas["auc_cv"]],
        "ap_cv_oof": [ap_oficial, metricas["ap_cv_oof"]],
        "churners_20": [valor("oficial", "churners_capturados"), valor("candidato de producción", "churners_capturados")],
        "captura_20": [valor("oficial", "captura"), valor("candidato de producción", "captura")],
        "origen": "validación cruzada en train"}, index=["oficial", "candidato de producción"])


def controles_modelo_viable(oficial_cargado: dict, produccion_cargado: dict, pipeline_final, X: pd.DataFrame,
                            y: pd.Series) -> pd.DataFrame:
    """Los ocho controles que demuestran que lo guardado es lo evaluado: tabla de `control` y `cumple`."""
    cols_prod = produccion_cargado["columnas_entrada"]
    oficial_desde_cero = clone(construir_pipeline_oficial()).fit(X, y)
    produccion_desde_cero = clone(construir_pipeline_produccion()).fit(X[cols_prod], y)
    prob_prod = produccion_cargado["pipeline"].predict_proba(X[cols_prod])[:, 1]
    return pd.DataFrame([
        ("Hiperparámetros fijos = los elegidos por la búsqueda",
         parametros_comparables(construir_pipeline_oficial().get_params()) == parametros_comparables(clone(pipeline_final).get_params())),
        ("Oficial guardado = modelo evaluado en test (predicciones en train)",
         np.array_equal(oficial_cargado["pipeline"].predict_proba(X), pipeline_final.predict_proba(X))),
        ("Oficial reconstruido sin caché = modelo evaluado en test",
         np.array_equal(oficial_desde_cero.predict_proba(X), pipeline_final.predict_proba(X))),
        ("Candidato reconstruido sin caché = candidato guardado",
         np.array_equal(produccion_desde_cero.predict_proba(X[cols_prod])[:, 1], prob_prod)),
        ("El candidato no recibe gender ni contacts_count_12_mon",
         not set(VARIABLES_EXCLUIDAS_PRODUCCION) & set(cols_prod)),
        ("Clase positiva = Attrited Customer (código 1, columna de classes_)",
         produccion_cargado["clase_positiva"] == config.CLASE_POSITIVA
         and indice_clase_positiva(produccion_cargado["pipeline"]) == 1),
        ("Huella de los datos de train igual en ambos modelos",
         oficial_cargado["huella_datos_train"] == produccion_cargado["huella_datos_train"]),
        ("Candidato marcado como no evaluado en test y sin métricas de test",
         produccion_cargado["evaluado_en_test"] is False and not any("test" in k for k in produccion_cargado["metricas"])),
    ], columns=["control", "cumple"])


if __name__ == "__main__":
    main()
