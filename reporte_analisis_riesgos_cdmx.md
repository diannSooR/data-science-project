# Reporte de análisis de riesgos en CDMX

## 1. Objetivo del análisis

El objetivo de este trabajo fue entender si en la Ciudad de México existe una relación entre tres tipos de riesgo:

- zonas con mayor exposición a sismos,
- zonas donde hay fracturas geológicas,
- zonas con mayor tendencia a inundaciones.

La idea no es demostrar causalidad. Es decir, no estamos diciendo que un fenómeno cause el otro. Lo que se buscó fue ver si, en términos espaciales, estas condiciones aparecen juntas con frecuencia y si esa coincidencia puede ser útil para interpretar la vulnerabilidad territorial de la ciudad.

Esto tiene sentido en CDMX porque parte de la ciudad está construida sobre antiguos suelos lacustres, con condiciones geológicas e hidráulicas complejas. Por eso, analizar cómo se superponen los riesgos puede ayudar a entender mejor qué zonas concentran más vulnerabilidad.

## 2. Qué datos se usaron

Se trabajó con varias fuentes de información geoespacial y tabular:

- Atlas de riesgo sísmico.
- Atlas de riesgo por inundaciones.
- Atlas de susceptibilidad por laderas.
- Registro único de situaciones de emergencia.
- Índice de marginación urbana (IMU 2020).
- Capa geográfica de fracturas.
- Polígonos de zonas AGEB ya limpiadas.

Las bases principales para este análisis final fueron las de sismo, inundación y fracturas. Las otras se limpiaron y conservaron como parte del flujo general de preparación de datos.

## 3. Qué se hizo antes de analizar

Antes de interpretar cualquier resultado, se hizo una preparación bastante agresiva de los datos para reducir errores y dejar todo comparable.

### 3.1 Auditoría de calidad

Se revisó para cada dataset:

- número de filas y columnas,
- tipos de datos,
- porcentaje de valores faltantes,
- duplicados,
- primeras observaciones para ver estructura y contenido.

### 3.2 Limpieza

Se aplicaron varias acciones:

- eliminación de duplicados,
- tratamiento de valores nulos,
- conversión de columnas a formato numérico cuando correspondía,
- eliminación de valores extremos o outliers en columnas numéricas,
- limpieza ligera de la capa de fracturas para asegurar geometrías válidas y CRS consistente.

### 3.3 Reetiquetado

Se estandarizaron variables de riesgo para que pudieran compararse mejor.

En particular:

- en sismo e inundación se tomó la variable de intensidad y el puntaje numérico `int2`,
- se generaron categorías de riesgo homogéneas:
  - BAJO,
  - MEDIO,
  - ALTO,
  - MUY_ALTO.

Esto fue importante porque cada fuente usa su propia lógica interna y había que llevarlas a una estructura común.

### 3.4 Integración espacial

Después se construyó una capa unificada a nivel de zonas del atlas.

Ahí se hizo lo siguiente:

- se reconstruyeron geometrías a partir de `geo_shape`,
- se agregaron los riesgos de sismo e inundación por zona,
- se cruzaron las fracturas con las zonas del atlas,
- se generaron variables nuevas que resumen la exposición combinada.

## 4. Qué tan limpio quedó el dato

Los datos quedaron en muy buen estado para análisis exploratorio.

### Limpieza por dataset

- Sismo: 4,908 filas originales, 4,878 limpias.
- Inundación: 4,908 originales, 4,878 limpias.
- Laderas: 4,908 originales, 4,878 limpias.
- RUSE: 31,589 originales, 31,543 limpias.
- IMU: 50,790 originales, 50,675 limpias.
- Fracturas: 6,965 originales, 6,965 limpias.

Esto significa que no hubo pérdidas graves de información. La limpieza eliminó sobre todo casos muy extremos o inconsistentes, pero conservó casi todo el universo de datos.

## 5. Qué variables nuevas se construyeron

Para poder responder la pregunta de análisis, se generaron indicadores derivados:

- `sismo_count_alto_riesgo`: cuántas observaciones por zona se clasifican como alto riesgo sísmico.
- `inundacion_count_alto_riesgo`: cuántas observaciones por zona se clasifican como alto riesgo de inundación.
- `fracturas_count`: cuántas fracturas intersectan cada zona.
- `indice_exposicion_sismo_fracturas`: índice combinado de exposición a sismo y fracturas.
- `exposicion_sismo_fracturas_categoria`: categoría final del índice combinado.
- `sismo_alto`: indicador binario de sismo alto.
- `fracturas_altas`: indicador binario de fracturas altas.
- `inundacion_alta`: indicador binario de inundación alta.

Estas variables permitieron comparar zonas de una forma mucho más clara que mirando todo por separado.

## 6. Qué encontraron las correlaciones

Esta fue la parte más importante del análisis.

### 6.1 Sismo e inundación

La correlación Pearson entre sismo e inundación fue de 0.8473.

Esto se interpreta como una relación positiva alta. En palabras sencillas: donde el riesgo sísmico es más alto, también tiende a ser más alto el riesgo de inundación.

### 6.2 Fracturas e inundación

La correlación Pearson entre fracturas e inundación fue de 0.0636.

Esto es una relación muy débil. Es decir, las fracturas por sí solas no explican mucho la variación de inundación.

La correlación de Spearman fue todavía más baja, 0.0279, lo que refuerza que la relación directa entre fracturas e inundación no es fuerte por sí misma.

### 6.3 Sismo + fracturas e inundación

La correlación Pearson del índice combinado de sismo + fracturas contra inundación fue de 0.7926.

La correlación de Spearman fue de 0.7523.

Esto sí muestra una asociación fuerte. En términos prácticos, cuando una zona combina exposición sísmica y presencia de fracturas, en los datos limpios también aparece una tendencia más marcada a inundación.

## 7. Qué significa esto en lenguaje simple

La lectura más sencilla es esta:

- no parece que las fracturas por sí solas expliquen mucho la inundación,
- pero cuando las fracturas se leen junto con la exposición sísmica, sí aparece una relación clara con inundación,
- y esa relación es fuerte en los datos limpios.

Dicho de otra manera: las zonas que ya son vulnerables por sismo y además tienen fracturas tienden a coincidir con zonas donde también hay más riesgo de inundación.

Esto no significa que uno cause al otro. Lo que significa es que el territorio está mostrando una superposición de vulnerabilidades.

## 8. Hallazgos por grupos

Se construyeron grupos para comparar zonas según su nivel de exposición.

El grupo más importante fue:

- Sismo alto + fracturas altas

Ese grupo tuvo un promedio de inundación de 0.9677.

Eso es muy alto. En lenguaje llano, casi todas esas zonas aparecen con inundación alta.

Por comparación:

- Sismo bajo + fracturas altas tuvo un promedio de inundación de 0.1355.

Eso es mucho menor.

La comparación entre grupos sugiere que el factor que realmente eleva el patrón de inundación no es solo la presencia de fracturas, sino la combinación con una exposición sísmica alta.

## 9. Qué dicen las tablas de contingencia

La tabla de contingencia mostró que en el grupo de sismo alto + fracturas altas aparecen muchísimas más zonas con inundación alta que en el grupo contrario.

Esto ayuda a reforzar la misma idea:

- la coincidencia de sismo alto y fracturas altas está asociada con inundación alta,
- mientras que fracturas solas no muestran el mismo peso explicativo.

## 10. Cómo interpretar visualmente el resultado

Las gráficas del notebook ayudan a entender la lectura:

- la distribución del sismo muestra zonas con distintos niveles de intensidad,
- la distribución de fracturas muestra dónde se concentran esas líneas o rupturas geológicas,
- el índice combinado concentra mejor la lectura territorial,
- y la comparación por grupo muestra que donde coinciden sismo alto y fracturas altas, la inundación promedio sube mucho.

## 11. Qué sí podemos concluir y qué no

### Sí podemos concluir

- Los datos ya están limpios y listos para análisis.
- Existe una asociación fuerte entre la exposición combinada de sismo + fracturas y la tendencia a inundación.
- El patrón espacial no parece aleatorio.
- Las zonas con sismo alto y fracturas altas tienden a coincidir con zonas de mayor inundación.

### No podemos concluir

- No podemos afirmar causalidad.
- No podemos decir que las fracturas “causen” inundaciones.
- No podemos decir que el sismo “genere” inundación.

Lo que sí podemos decir es que las capas de riesgo están concentrándose en los mismos territorios, y eso es relevante para priorización territorial.

## 12. Estado actual del proyecto

Con lo que ya está hecho, el proyecto tiene completados los pasos siguientes:

- definición del problema,
- recolección y limpieza de datos,
- exploración y análisis de datos,
- parte importante del preprocesamiento y feature engineering,
- interpretación y visualización.

Todavía no se ha entrado a modelado predictivo, entrenamiento o evaluación de modelo.

## 13. Archivo principal generado

El archivo más importante para continuar trabajando es:

- `output/ageb_riesgos_combinados.csv`

Ese archivo concentra las variables ya limpias y unificadas, y es el mejor punto de partida si después quieres hacer modelado, mapas finales o priorización de zonas.

## 14. Resumen final

En términos simples, el análisis dice lo siguiente:

la Ciudad de México muestra una superposición clara entre zonas con exposición sísmica alta, presencia de fracturas y tendencia a inundación. La relación más útil no es mirar cada riesgo aislado, sino combinarlos para ver dónde coinciden. En tus datos limpios, esa coincidencia sí aparece con fuerza y justifica seguir trabajando el tema como una lectura territorial de vulnerabilidad acumulada.
