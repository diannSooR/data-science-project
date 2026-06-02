# Explicación súper sencilla del análisis de riesgo antes de la regresión lineal

## ¿Qué se quiso hacer?

El objetivo del análisis fue encontrar qué zonas de la ciudad tienen más riesgo ante fenómenos como sismos, inundaciones y problemas en laderas, usando información geográfica y estadística por zona AGEB.[cite:18][cite:20]
Cada AGEB puede imaginarse como una pequeña zona o vecindario estadístico al que se le asignaron variables como tipo de suelo, riesgo sísmico, riesgo de inundación, fracturas del terreno, población y uso de servicios de emergencia.[cite:18][cite:20]

## ¿Por qué primero hubo que limpiar los datos?

Antes de hacer mapas o modelos, se detectaron varios problemas en la versión anterior de los datos, especialmente en v2.[cite:20]
Había variables que casi no cambiaban entre zonas, valores faltantes en tipo de suelo, y outliers muy grandes en fracturas que podían deformar el análisis completo.[cite:20][cite:18]

Para corregir esto, se creó una versión mejorada llamada v3.[cite:20]
En esa versión se recalcularon variables, se imputó el tipo de suelo faltante por la moda municipal, se recortaron valores extremos y se volvieron a normalizar las variables importantes en una sola escala comparable.[cite:20]

## ¿Qué es el índice de riesgo compuesto?

Como había muchas variables distintas, se construyó un índice de riesgo compuesto para resumir todo en un solo número por zona.[cite:20][cite:18]
Ese índice combina factores como riesgo sísmico, riesgo de inundación, fracturas, tipo de suelo, afectación estimada y vulnerabilidad, dando más peso a algunas variables que a otras.[cite:20]

El resultado es un valor continuo entre 0 y 1, donde los números más altos representan mayor riesgo relativo.[cite:18][cite:20]
Después, ese índice se dividió en cinco niveles: Bajo, Bajo-Medio, Medio, Medio-Alto y Alto, repartiendo casi el mismo número de AGEB en cada categoría.[cite:18][cite:6]

## ¿Qué mostraron los mapas?

Los mapas de riesgo general y del índice compuesto muestran un patrón espacial muy claro: la zona centro-oriente concentra muchos valores altos de riesgo, mientras que el sur y el poniente tienen más zonas de riesgo bajo o medio-bajo.[cite:16][cite:15]
Esto indica que el riesgo no está distribuido de manera uniforme, sino que forma corredores o cinturones territoriales bien definidos.[cite:15][cite:16]

Los mapas por tipo de riesgo permiten entender por qué pasa eso.[cite:17][cite:18]
El riesgo sísmico y el de inundación se concentran con fuerza en la franja central y oriental, mientras que el riesgo por laderas aparece más en zonas con relieve accidentado, sobre todo al sur y poniente.[cite:17][cite:18]

El análisis por tipo de suelo también refuerza esta lectura territorial.[cite:9][cite:18]
Las zonas de suelo arenoso presentan medianas más altas del índice compuesto, seguidas por las de suelo mixto, mientras que las zonas de roca tienden a concentrar los niveles más bajos de riesgo.[cite:9][cite:18]

## ¿Qué relación se vio entre variables?

La matriz de correlación muestra que el índice de riesgo compuesto está muy relacionado con variables físicas del territorio, sobre todo con tipo de suelo, riesgo de inundación, severidad de inundación y riesgo sísmico.[cite:2][cite:18]
En cambio, la población total casi no muestra relación fuerte con el índice compuesto, lo que sugiere que el índice mide más la peligrosidad física del lugar que el simple tamaño de la población.[cite:2][cite:8][cite:18]

Esto es importante porque evita interpretar el modelo como si solo estuviera detectando dónde vive más gente.[cite:8][cite:18]
Más bien, el índice está capturando una combinación de condiciones del terreno y exposición a amenazas naturales.[cite:2][cite:18]

## ¿Para qué sirvió K-means?

Además del índice compuesto, se usó K-means para agrupar las zonas en cuatro perfiles de riesgo parecidos entre sí: Bajo, Medio-Bajo, Medio-Alto y Alto.[cite:19]
Este modelo no predice daños futuros directamente, pero sí permite segmentar el territorio en grupos con características de riesgo similares.[cite:19][cite:13]

La comparación de métricas entre versiones muestra que v3 mejora frente a v2 para el clustering, aunque no supera a v1 en separación geométrica pura.[cite:7][cite:19]
Aun así, v3 se considera una mejor opción práctica porque conserva más zonas, más variables y una estructura territorial más coherente para interpretar el riesgo urbano.[cite:11][cite:19]

El mapa de clusters de v3 confirma el mismo patrón espacial observado antes: el riesgo alto se concentra en el centro-oriente y el riesgo bajo domina en buena parte del sur y poniente.[cite:13][cite:11]
Esto da consistencia al análisis, porque tanto el índice compuesto como el clustering apuntan hacia la misma lectura territorial.[cite:13][cite:15][cite:16]

## ¿Qué se puede concluir antes de la regresión?

Antes de entrar a la regresión lineal, ya se puede concluir que la versión v3 tiene datos mejor preparados, un índice compuesto más interpretable y una clasificación espacial del riesgo más coherente que la versión anterior.[cite:20][cite:18][cite:19]
También se puede afirmar que el riesgo urbano observado en este análisis depende sobre todo de condiciones físicas del territorio, especialmente suelo, sismo e inundación, y no solamente de cuánta población vive en una zona.[cite:2][cite:8][cite:18]

Con esa base, la regresión lineal se usa después como una validación adicional para comprobar si ese índice de riesgo también se relaciona con una variable observable del mundo real, como el uso de servicios de emergencia.[cite:22][cite:33]
