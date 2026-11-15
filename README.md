# Predicción de churn bancario. Proyecto final Analítica II

Proyecto final del curso Métodos y Aplicaciones de Analítica II (Maestría en Analítica para la Inteligencia de Negocios, Pontificia Universidad Javeriana). Equipo: Juan Fernando Rengifo y Juan Diego Díaz.

**Problema:** el banco perdió el 15 % de sus clientes activos en el último trimestre y quiere priorizar un programa de beneficios hacia los clientes con mayor probabilidad de irse a otra entidad. **Entregable analítico:** un modelo que estima la probabilidad de churn (abandono) de cada cliente y una lista priorizada para el área comercial.

## Estructura

```
├── data/
│   ├── raw/              Datos originales del caso (no se modifican)
│   └── processed/        Datos limpios generados por código (no se versionan)
├── docs/
│   ├── oficial/          Enunciado oficial del caso
│   └── evidencia/        Captura de pytest (153 pruebas) y lista de la revisión externa con ChatGPT
├── src/churn/            Librería del proyecto: config, datos, exploracion, features, modelos, comparacion,
│                         evaluacion, negocio, explicabilidad, produccion, monitoreo, reportes, gráficos
│                         (graficos, graficos_modelado, graficos_negocio) y entorno (solo Windows)
├── notebooks/            Notebook principal que soporta los resultados, organizado en las 6 fases de CRISP-DM
├── models/               Modelos serializados y caché de búsquedas (no se versionan; se regeneran)
├── reports/figures, reports/tables   Gráficos y tablas para el informe
├── tests/                Pruebas automáticas con pytest
└── informe/              Informe final y presentación (se entregan aparte; no se versionan)
```

## Instalación (Windows, VS Code)

Requisito: Python 3.11 (PyCaret 3.3.2 no soporta 3.12 o superior).

```powershell
py -3.11 --version                 # confirmar que Python 3.11 está instalado
py -3.11 -m venv .venv             # crear el entorno virtual en la raíz del proyecto
.venv\Scripts\activate             # activarlo
python -m pip install --upgrade pip
pip install -r requirements.txt    # dependencias con versiones fijas
pip install -e .                   # instalar la librería src/churn en modo editable
python -m churn.reparar_entorno --reparar   # solo Windows: corrige el choque de DLL de LightGBM (ver abajo)
pytest                             # correr las 153 pruebas (unos 2,5 minutos)
```

**Windows y LightGBM.** scikit-learn 1.4.2 trae una `msvcp140.dll` (14.32) más vieja que la que necesita LightGBM 4.7.0 (14.40 o superior); sin reparar, LightGBM falla al entrenar con `OSError: exception: access violation reading 0x0000000000000000`. El comando `python -m churn.reparar_entorno --reparar` reemplaza esa copia por la del sistema (con respaldo `.original`) sin cambiar versiones ni resultados. Requiere el Visual C++ Redistributable 2015-2022 x64 actualizado (https://aka.ms/vs/17/release/vc_redist.x64.exe) y se repite en cada `.venv` nuevo.

En VS Code: abrir la carpeta del proyecto, seleccionar el intérprete `.venv` y usarlo como kernel del notebook.

**Uso de inteligencia artificial.** El proyecto lo desarrollamos Juan Fernando Rengifo y Juan Diego Díaz como equipo; la IA fue una herramienta de apoyo y las decisiones fueron nuestras. Usamos Claude (Anthropic) como asistente de trabajo, con el que discutimos el planteamiento y escribimos y revisamos código y documentación, y ChatGPT (OpenAI) como segunda opinión externa, que revisó versiones del notebook sin el contexto de ese trabajo. Nosotros definimos los requisitos a partir del caso y de las indicaciones del profesor, elegimos entre las propuestas, revisamos cada pieza frente al temario del curso, y ejecutamos y verificamos los resultados en nuestros computadores. Ninguna observación de las herramientas se aplicó sin contrastarla antes con el proyecto. La evidencia de esa validación está en `docs/evidencia/`.

## Uso

Abrir `notebooks/churn_banco.ipynb` y ejecutar de arriba a abajo. El notebook sigue las 6 fases de CRISP-DM: comprensión del negocio (Fase 1), comprensión de los datos (Fase 2), preparación (Fase 3), modelado (Fase 4), evaluación (Fase 5) y despliegue, monitoreo y control (Fase 6). Todas las rutas son relativas a la raíz del repositorio (ver `src/churn/config.py`). En la Fase 2, el EDA automático (`ydata-profiling`, reporte en `reports/perfil_train.html`) hace el barrido completo y el análisis dirigido solo profundiza donde hay relación con el churn o un riesgo para la decisión.

Las búsquedas de hiperparámetros de la Fase 4 (diez modelos, dos rondas) tardan cerca de una hora con 2 núcleos la primera vez. Cada modelo queda guardado en `models/experimentos/` con una huella de los datos, la grilla y las versiones de scikit-learn, imbalanced-learn, LightGBM y XGBoost; en las siguientes ejecuciones se carga en segundos. Si algo cambia, solo ese modelo se reentrena. `REENTRENAR = True` en 4.2 obliga a repetir todo. La caché no se sube a GitHub.

## Modelo viable: cargar el modelo y puntuar clientes

Hay un solo modelo: el que se evalúa en test es el mismo que puntúa clientes.

| Archivo | Qué es | Evidencia de desempeño |
|---|---|---|
| `models/modelo_oficial.joblib` | LightGBM elegido en la Fase 4, con 13 variables numéricas (las categóricas no aportan en validación cruzada, 4.3) | Test (5.1 del notebook) |

La versión sin `contacts_count_12_mon` no se guarda: 5.7 mide que sin esa variable el resultado casi no cambia, y `produccion.construir_pipeline_alternativo` la deja disponible por si el banco confirma que los contactos incluyen el trámite de retiro.

Se genera al ejecutar el notebook o, sin el notebook y en segundos (no repite la búsqueda de hiperparámetros):

```powershell
python -m churn.produccion entrenar
```

Puntuar un archivo de clientes (Excel o CSV con el formato del archivo del banco) y guardar la lista priorizada:

```powershell
python -m churn.produccion puntuar clientes.xlsx lista_priorizada.csv --capacidad 0.2
```

Desde Python:

```python
from churn import produccion

modelo = produccion.cargar_modelo()                 # modelo oficial
lista = produccion.puntuar_clientes(clientes, modelo, capacidad=0.2)
lista.head()                                        # de mayor a menor riesgo
```

**Entrada.** Un DataFrame con `clientnum` (como columna o índice) y las 14 columnas numéricas del archivo original ( `avg_open_to_buy` se pide aunque el modelo la descarta). Las columnas de más se ignoran y quedan listadas en `lista.attrs["columnas_ignoradas"]`. La función se detiene con `ErrorDeDatos` si faltan columnas, hay vacíos, identificadores repetidos, textos en columnas numéricas o valores imposibles (por ejemplo, utilización mayor que 1). Las columnas categóricas no se usan: el modelo las ignora. Los valores fuera del rango de train generan una advertencia porque el modelo extrapola.

**Salida.** Una tabla indexada por `clientnum` y ordenada de mayor a menor riesgo, con `prioridad` (1 = mayor riesgo), `prob_churn` (probabilidad de la clase `Attrited Customer`), `percentil_riesgo` y, si se indica la capacidad, `contactar` (True para el top k % del archivo, con el mismo redondeo del notebook). Para seguir la recomendación del 20 %, el archivo debe ser la base activa completa del ciclo. Los empates conservan el orden de entrada, así que la misma entrada produce siempre la misma lista. Para decidir, use el ranking y no la probabilidad como umbral (5.3 y 5.4).

Al cargar, `cargar_modelo` revisa que el archivo esté completo y avisa si las versiones de scikit-learn, LightGBM o imbalanced-learn no son las del entrenamiento. `produccion.ficha(modelo)` muestra su papel, métricas con su origen, huella de los datos y versiones.

## Monitoreo y control

`monitoreo.tabla_semaforos` calcula los tres indicadores de 6.3 para uno o varios lotes de clientes, con semáforo verde, amarillo o rojo: el PSI (índice de estabilidad poblacional) de cada variable y el del puntaje del modelo, que se revisan con cada lote, y la captura del 20 % de mayor riesgo en un grupo de control, que se revisa una vez al año, cuando se conoce el churn real. Los umbrales están en `config.UMBRALES_MONITOREO`. En el notebook se demuestra con lotes simulados a partir de clientes de train; no es un resultado del modelo. Con clientes nuevos, la referencia del PSI del puntaje deben ser las predicciones fuera de pliegue (`puntaje_referencia`), porque el modelo es más extremo con los clientes con que se entrenó.

## Reproducibilidad

- Semilla global `SEMILLA = 42` en `src/churn/config.py`.
- Copias de datos verificadas con SHA-256 contra los originales del caso.
- El conjunto de prueba no participa en ninguna decisión: la selección de variables, de hiperparámetros y de modelo se hace con validación cruzada en train. Se evaluó dos veces: con el procedimiento inicial y, tras definir la selección de variables de 4.3, con el procedimiento final (declarado en 5.8).
- Los hiperparámetros del modelo oficial quedan fijos en `churn.produccion.HIPERPARAMETROS_OFICIALES`; una prueba comprueba que coinciden con la búsqueda.
- `tests/test_cifras_texto.py` lee el notebook y exige que cada cifra del texto aparezca en una salida del código o en una tabla de `reports/tables/`; además comprueba que las decisiones escritas se cumplen en las tablas; las demás pruebas `test_cifras_*.py` recalculan por otra vía las cifras del EDA y de la preparación. Se omiten si el notebook aún no se ha ejecutado en ese computador.
- **Caché y diferencias entre sistemas operativos.** XGBoost y Gradient Boosting pueden cambiar en la cuarta decimal entre Linux y Windows, e incluso entre dos ejecuciones sin caché; LightGBM, el modelo final, y todo lo que depende de él dan exactamente lo mismo. La caché `models/experimentos/` del 13-nov (generada en la nube con las mismas versiones de librerías) reproduce exactamente las cifras del texto en cualquier computador; es la que se comparte entre los dos integrantes. Sin caché (`REENTRENAR = True`), `test_cifras_texto.py` señala cualquier cifra que haya cambiado.
