# Proyecto_programacion_II
## Descripción

Esta aplicación interactiva analiza el conjunto Bank Failures and Assistance Data, publicado por la Corporación Federal de Seguro de Depósitos de Estados Unidos (FDIC). La base reúne información sobre instituciones bancarias que quebraron o recibieron asistencia, incluidos sus activos, las pérdidas estimadas, la fecha del evento y el tipo de resolución.

El proyecto presenta tres análisis:

1. Activos, pérdidas y tipo de resolución: compara las pérdidas registradas con los activos de los bancos. Incluye gráficos de dispersión, histogramas y densidades, además de resúmenes con observaciones, medias y medianas.
2. Evolución temporal: muestra la cantidad de quiebras bancarias y las pérdidas económicas por año, junto con gráficos, interpretaciones y una tabla de resumen.
3. Distribución geográfica: compara la cantidad de quiebras y la pérdida estimada promedio por estado o territorio mediante un mapa interactivo y un gráfico de dispersión.

## Archivos necesarios

Se deben mantener estos archivos en la misma carpeta:

- app.p`: código de la aplicación.
- Failures_10_26_2026.csv: conjunto de datos utilizado por la aplicación.

## Instrucciones

1. Instalar los paquetes matplotlib.pyplot, numpy, pandas, folium, folium.plugins, plotly.graph_objects, shinywidgets y shiny
2. Abrir el archivo y correr "!shiny run --reload --port 8003 app.py"
3. Observar el análisis
