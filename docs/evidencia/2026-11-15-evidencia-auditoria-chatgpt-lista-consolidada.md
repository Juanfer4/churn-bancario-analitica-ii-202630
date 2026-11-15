# Auditoría final del proyecto de churn bancario

**Curso:** Analítica II  
**Notebook de referencia:** `churn_banco(2).ipynb`  
**Alcance:** revisión metodológica, técnica y documental del proyecto, desde la preparación de datos hasta la propuesta de producción y monitoreo.  
**Estado:** lista final consolidada para revisión y cierre.

---

## 1. Instrucción principal

El proyecto está en una etapa cercana a su cierre. El objetivo de esta revisión **no es maximizar la cantidad de mejoras**, sino asegurar que la solución sea válida, coherente, defendible y proporcional al alcance de Analítica II.

Se solicita revisar los 22 cuestionamientos de este documento y priorizar las acciones indispensables. No asumir que cada pregunta exige modificar el código. Algunas pueden cerrarse mediante una explicación sustentada; otras pueden clasificarse como opcionales o descartarse si no tienen impacto material.

**Criterio rector:** ante dos alternativas técnicamente aceptables, preferir la más sencilla, transparente y fácil de defender, siempre que preserve la validez del análisis y el cumplimiento del proyecto.

### Qué evitar

- No añadir modelos, pruebas estadísticas, métricas, gráficos, librerías o arquitecturas solo porque técnicamente existan.
- No reestructurar etapas completas si una aclaración o ajuste puntual resuelve el problema.
- No convertir cada parámetro, umbral o decisión visual en una tarea independiente.
- No repetir análisis que ya aportan la misma evidencia.
- No confundir una mejora posible con un defecto que deba corregirse antes de entregar.
- No afirmar que una validación se realizó si no existe evidencia en el notebook, los módulos o sus resultados almacenados.

### Cómo responder cada cuestionamiento

Para cada punto, indicar:

1. **Diagnóstico:** problema real, justificación pendiente, mejora opcional o descartable.
2. **Evidencia:** celda, módulo, variable, resultado o requisito concreto que sustenta la conclusión.
3. **Materialidad:** impacto en validez, resultado, interpretación, decisión de negocio o cumplimiento académico.
4. **Acción requerida:** corregir antes de entregar, aclarar en el notebook/informe, dejar como anexo o no intervenir.
5. **Verificación posterior:** si se modifica algo, volver al código real y revisar también las etapas posteriores afectadas.

No proponer modificaciones extensas antes de demostrar por qué son necesarias.

---

## 2. Prioridades de la auditoría

### Prioridad crítica — puede comprometer la validez

- Disponibilidad temporal de las variables al momento de generar la predicción, especialmente `contacts_count_12_mon`.
- Definición del objetivo, integridad de los datos y ausencia de fuga convencional o temporal.
- Separación entre entrenamiento y test, y correcta interpretación de las métricas.
- Coherencia entre el modelo que se evalúa y el artefacto que se propone utilizar.
- Correspondencia entre la priorización de clientes y la capacidad operativa que se supone disponible.

### Prioridad alta — puede afectar la defensa o la decisión de negocio

- Cobertura del EDA y justificación de las decisiones de preparación.
- Cobertura de los métodos de modelado vistos en clase y comparabilidad de las búsquedas.
- Interpretación de los escenarios económicos y distinción entre riesgo de churn y efecto de una campaña.
- Uso de la edad y lectura de diferencias entre grupos.
- Trazabilidad de artefactos, validación de entradas y alcance real del monitoreo.

### Prioridad secundaria — no debe bloquear la entrega sin impacto demostrado

- Variaciones de parámetros de gráficos o discretizaciones.
- Análisis adicionales que repitan conclusiones.
- Mejoras de ingeniería o monitoreo que no sean necesarias para demostrar el alcance académico.
- Técnicas futuras que requieren datos que el proyecto todavía no tiene.

---

## 3. Cuestionamientos consolidados

Las 22 preguntas siguientes sustituyen la lista extensa de observaciones individuales. Cada una integra cuestionamientos relacionados. Solo deben volver a separarse si la revisión del código descubre problemas distintos que requieran decisiones independientes.

### A. Datos, EDA y preparación

**Q01 — Integridad y significado de los datos**

¿Las validaciones de esquema, tipos, categorías, rangos, valores faltantes, definición del objetivo e identidades contables reflejan la semántica del dataset y protegen razonablemente la entrada de datos? ¿Las decisiones de limpieza y codificación están explicadas y son coherentes con el caso?

**Q02 — Suficiencia del análisis exploratorio**

¿El EDA univariado y multivariado cubre lo exigido por el curso y las conclusiones que se utilizan posteriormente están respaldadas por evidencia suficiente? Revisar especialmente tamaños de grupo, incertidumbre, categorías poco frecuentes, asociaciones débiles y la diferencia entre asociación y causalidad. Evitar exigir una justificación separada para cada gráfico si no cambia la conclusión.

**Q03 — EDA automatizado: mantener, complementar o sustituir**

¿Un flujo o informe automatizado de EDA aportaría cobertura univariada y multivariada que actualmente falte, o duplicaría los análisis ya realizados? Comparar su utilidad frente al requisito académico, la calidad de las conclusiones, las decisiones de limpieza y preparación ya tomadas, la compatibilidad del entorno y el impacto sobre las celdas posteriores.

La respuesta debe dar una recomendación explícita entre mantener el EDA actual, complementarlo de forma acotada o sustituir una parte. No asumir que una herramienta automática reemplaza la interpretación estadística ni abrir una reestructuración tardía si el beneficio no es claro.

**Q04 — Preparación, transformaciones y fuga**

¿Las variables derivadas, transformaciones, codificaciones y tratamientos de categorías poco frecuentes están justificados por el EDA y el objetivo del modelo? ¿Los parámetros de preparación se ajustan exclusivamente con los datos correspondientes a cada fold de entrenamiento y se evita utilizar información del test? ¿La complejidad de las transformaciones es proporcional al aporte que ofrecen?

### B. Modelado y evaluación

**Q05 — Cobertura del temario**

¿El catálogo manual incluye los métodos predictivos relevantes que se estudiaron en clase, sin incorporar técnicas ajenas al temario? ¿La referencia de PyCaret se presenta de forma separada y coherente con la solicitud del profesor?

**Q06 — Comparabilidad de modelos y búsqueda**

¿Las estrategias de balanceo, pipelines, particiones, métricas y presupuestos de búsqueda permiten una comparación suficientemente justa y trazable? ¿Las grillas y sus rangos tienen una justificación razonable sin necesidad de convertir cada hiperparámetro en una tarea independiente?

**Q07 — Selección de modelos y optimismo**

¿La regla de selección basada en una desviación estándar está descrita como una heurística y no como una prueba formal de equivalencia? ¿Se reconoce el posible optimismo derivado de seleccionar hiperparámetros y calcular predicciones OOF sobre el mismo conjunto de entrenamiento? ¿La limitación está explicada de manera proporcional al diseño utilizado?

**Q08 — Separación del test y métricas**

¿El conjunto de test se mantuvo independiente de las decisiones de selección y ajuste? ¿Las métricas y visualizaciones elegidas aportan evidencia complementaria para evaluar ranking, clasificación y priorización, sin acumular indicadores redundantes? ¿Se interpreta el AUC alto con rigor, sin asumir que por sí solo demuestra fuga ni que garantiza desempeño futuro?

**Q09 — Bootstrap y calibración**

¿El bootstrap y la calibración se interpretan de acuerdo con lo que realmente estiman y con sus limitaciones? ¿Aportan información relevante para comparar modelos o tomar decisiones, o alguno de estos análisis puede quedar como diagnóstico complementario sin bloquear la entrega?

### C. Negocio y explicabilidad

**Q10 — Escenarios de capacidad y priorización**

¿Los escenarios de contacto y sus métricas de captura, precisión y rendimiento marginal están calculados e interpretados correctamente? ¿Se presentan como escenarios de capacidad y no como una restricción real del banco cuando esta no está documentada? ¿Se distingue claramente la evidencia OOF de la evidencia de test?

**Q11 — Supuestos económicos**

¿Los parámetros de efectividad, valor o margen de un cliente retenido y costo del beneficio —incluidos `s`, `r`, `V` y `c`, según la formulación utilizada— tienen definiciones, unidades y horizontes temporales coherentes? ¿El análisis se presenta como sensibilidad bajo supuestos y no como rentabilidad real estimada del banco?

**Q12 — Riesgo predictivo frente a efecto causal**

¿El trabajo distingue claramente la probabilidad de churn de la probabilidad de que una intervención evite el abandono? ¿Las conclusiones evitan atribuir al modelo capacidad para identificar quién responderá al beneficio cuando no se cuenta con datos experimentales o de campañas que permitan estimar ese efecto?

**Q13 — Aporte de la explicabilidad**

¿La importancia de variables y los análisis de dependencia parcial aportan interpretaciones distintas y útiles para el objetivo? ¿El PDP bidimensional añade una conclusión material que no se obtenga con el EDA y la importancia global, o su complejidad hace más conveniente dejarlo como complemento/anexo? No exigir análisis adicionales si no cambian una interpretación o decisión.

**Q14 — Equidad y uso de la edad**

¿El análisis por sexo y edad está definido como diagnóstico descriptivo, considera tamaños de grupo e incertidumbre y evita afirmar una certificación general de equidad? ¿La utilidad predictiva de la edad se distingue de las implicaciones de su uso operativo y de cumplimiento? ¿Se evita interpretar diferencias observadas como causalidad o declarar ausencia de sesgo únicamente porque una diferencia no sea estadísticamente significativa?

### D. Riesgo temporal, producción y monitoreo

**Q15 — Disponibilidad temporal y `contacts_count_12_mon`**

¿Está documentado el momento exacto en que se pretende generar la predicción y la disponibilidad de `contacts_count_12_mon` y del resto de variables en ese momento? ¿La decisión de utilizar el candidato sin contactos está sustentada en el riesgo temporal y en una comparación de desempeño correctamente delimitada, y no únicamente en la pequeña diferencia de métricas?

Este punto requiere especial atención. El patrón de 42 clientes con seis contactos y el desempeño alto del modelo justifican investigar la variable, pero no demuestran por sí solos que exista fuga temporal. La conclusión depende de la definición y fecha de registro de los contactos.

**Q16 — Alcance del análisis de los 42 clientes**

¿El análisis de los 42 clientes con seis contactos se presenta como evidencia puntual de un patrón y no como demostración de ausencia de fuga o como validación independiente del modelo? ¿Aporta información adicional frente a la comparación global del modelo con y sin la variable?

**Q17 — Evidencia del modelo oficial y candidato**

¿La evidencia de validación cruzada del candidato y la evidencia de test del modelo oficial están claramente diferenciadas? ¿Se evita atribuir al candidato resultados de test que no se han calculado? ¿La decisión de producción reconoce tanto la pérdida predictiva observada como el riesgo temporal que se busca evitar?

**Q18 — Trazabilidad e integridad de artefactos**

¿Los artefactos guardados contienen información suficiente para identificar su configuración, datos de entrenamiento, versiones y procedencia? ¿Los controles de reconstrucción y consistencia se presentan como evidencia de integridad técnica y no como una certificación adicional del desempeño predictivo? Evitar multiplicar controles equivalentes si no aportan garantías distintas.

**Q19 — Puntuación y priorización operativa**

¿La función de puntuación valida las entradas relevantes y produce una priorización consistente sobre el lote operativo completo? ¿Quedan claras la definición de la probabilidad, la interpretación del percentil de riesgo, la capacidad aplicada y las limitaciones de la demostración con una muestra pequeña de clientes de entrenamiento?

**Q20 — Validez del monitoreo**

¿El PSI utiliza una referencia adecuada para el modelo desplegado y se interpreta teniendo en cuenta valores faltantes, discretización, cambios de puntaje y disponibilidad de etiquetas reales? ¿La línea base de captura, los umbrales y la periodicidad están justificados de forma suficiente? ¿Los escenarios sintéticos se presentan como demostración de la mecánica y no como validación de monitoreo en producción?

**Q21 — Piloto y estimación del efecto del beneficio**

¿El piloto propuesto permitiría comparar clientes que reciben el beneficio con un grupo de control comparable y estimar el efecto incremental de la campaña? ¿La asignación y medición permitirían estimar los parámetros económicos sin confundir la captura del modelo con la efectividad causal de la intervención?

**Q22 — Alcance de las oportunidades futuras**

¿La calibración adicional, los modelos de *uplift* y el análisis de supervivencia responden a limitaciones reales del trabajo actual y cuentan con los datos necesarios? ¿Quedan claramente presentados como oportunidades futuras y no como requisitos que deban implementarse para cerrar el proyecto?

---

## 4. Revisión especial de complejidad y cierre

La otra IA debe prestar atención particular a los bloques finales. La pregunta no es si cada técnica puede justificarse en abstracto, sino si su aporte es necesario para la entrega.

### Elementos que requieren una decisión explícita

| Elemento | Qué debe determinarse |
|---|---|
| Importancia por ganancia y permutación | Si ambas medidas aportan evidencia complementaria y si sus diferencias se explican correctamente. |
| PDP unidimensional y bidimensional | Si el 2D cambia alguna conclusión o puede quedar como análisis complementario. |
| Análisis de equidad | Si las métricas seleccionadas responden una pregunta concreta y si las conclusiones respetan sus límites. |
| Controles de artefactos | Si los controles son distintos y necesarios o si varios comprueban esencialmente lo mismo. |
| Demostración de puntuación | Si prueba la funcionalidad necesaria sin presentarse como validación de generalización. |
| Monitoreo PSI | Si la mecánica está explicada y sus límites están claros, sin pretender que los escenarios sintéticos validan producción. |
| Calibración, *uplift* y supervivencia | Si deben permanecer como trabajo futuro, sin ampliar el alcance actual. |

**Regla para simplificar:** si dos análisis llevan a la misma conclusión y uno no añade evidencia relevante, no deben presentarse como dos validaciones independientes. El análisis más complejo puede conservarse en el notebook o en un anexo si aporta trazabilidad, pero no debe dominar la presentación ni convertirse en una tarea obligatoria de corrección.

---

## 5. Protocolo de doble validación

La revisión debe ejecutarse en dos pasadas.

### Primera pasada — diagnóstico

- Revisar los 22 cuestionamientos en el orden propuesto.
- Dar prioridad a datos, EDA, preparación y disponibilidad temporal de variables, porque sus decisiones pueden propagarse al resto del flujo.
- Revisar con especial rigor evaluación, explicabilidad, equidad, producción, puntuación y monitoreo.
- Clasificar cada punto como:
  - **Crítico antes de entregar**
  - **Aclaración documental**
  - **Opcional / anexo**
  - **Descartable con justificación**
- No modificar código durante esta clasificación inicial.

### Segunda pasada — verificación

Para cada punto que se considere resuelto o que haya generado un cambio:

1. Volver al código real del notebook y del módulo involucrado.
2. Comprobar que el cambio responde al problema identificado.
3. Revisar los resultados almacenados pertinentes.
4. Comprobar si el cambio afecta etapas posteriores, métricas, selección, interpretación o artefactos.
5. Cerrar el punto solo si la evidencia respalda la conclusión.

No afirmar que se ejecutaron pruebas nuevas si no se realizaron. La revisión puede ser estática cuando los resultados almacenados sean suficientes; si una conclusión depende de una nueva ejecución, debe identificarse explícitamente.

---

## 6. Criterio de cierre y entregable esperado

El proyecto se considera listo para cierre cuando:

- No quedan problemas críticos de validez, fuga, separación de datos o interpretación.
- Las decisiones metodológicas principales están justificadas con evidencia del notebook.
- Los resultados de test, OOF y escenarios de negocio están correctamente diferenciados.
- La propuesta de producción no atribuye al modelo capacidades que no se han validado.
- Las conclusiones sobre edad, equidad, economía y monitoreo respetan el alcance de los análisis.
- Las mejoras opcionales están separadas de las correcciones necesarias.
- La complejidad final es proporcional al objetivo académico y puede explicarse con claridad.

**No se exige que las 22 preguntas produzcan cambios.** El resultado esperado es una lista de decisiones cerradas con evidencia, un conjunto pequeño de correcciones indispensables —si las hay— y una explicación clara de los aspectos que se conservan como limitaciones o trabajo futuro.

---

## 7. Límites conocidos de esta auditoría

Esta lista consolida la auditoría documental realizada sobre el notebook, sus resultados almacenados y los módulos revisados durante el proceso. No certifica modificaciones futuras ni sustituye la validación de cualquier cambio que se implemente después.

En particular:

- El riesgo temporal de `contacts_count_12_mon` requiere confirmar la fecha de disponibilidad de la variable.
- Un AUC alto, por sí solo, no demuestra fuga ni garantiza desempeño futuro.
- Los escenarios económicos dependen de supuestos y no equivalen a rentabilidad real observada.
- El análisis por grupos no certifica por sí solo equidad ni cumplimiento legal.
- Los escenarios sintéticos de PSI no prueban que el monitoreo esté validado en producción.
- La conveniencia de incorporar EDA automatizado debe decidirse por su aporte marginal y no por la disponibilidad de herramientas.

**Recomendación final:** utilizar este documento como la única lista activa de revisión para cerrar el proyecto. Conservar las versiones históricas anteriores únicamente como respaldo de trazabilidad y no como tareas adicionales.
