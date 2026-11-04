from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, render, ui
from shiny import ui

#Archivo 
BASE_DATOS = pd.read_csv(Path(__file__).with_name("Failures_10_26_2026.csv"))

##############################################################################################
#Funciones para la pregunta 1 
RESOLUCIONES = BASE_DATOS["RESTYPE1"].value_counts().head(3).index.tolist()
ACTIVOS_P33, ACTIVOS_P67 = BASE_DATOS["QBFASSET"].quantile([1 / 3, 2 / 3])
BASE_DATOS["Tamaño de activos"] = np.select(
    [BASE_DATOS["QBFASSET"] <= ACTIVOS_P33, BASE_DATOS["QBFASSET"] <= ACTIVOS_P67],
    ["Bajo", "Medio"],
    default="Alto",
)


def puntos(resolucion):
    filas = BASE_DATOS.loc[BASE_DATOS["RESTYPE1"] == resolucion, ["QBFASSET", "COST"]].dropna()
    for columna in ("QBFASSET", "COST"):
        limite_inferior, limite_superior = filas[columna].quantile([0.25, 0.75])
        rango = limite_superior - limite_inferior
        filas = filas[filas[columna].between(limite_inferior - 1.5 * rango, limite_superior + 1.5 * rango)]
    return filas


def crear_tabla(encabezados, filas, nombre_clase="tabla-datos"):
    return ui.tags.table(
        ui.tags.thead(ui.tags.tr(*[ui.tags.th(encabezado) for encabezado in encabezados])),
        ui.tags.tbody(*[ui.tags.tr(*[ui.tags.td(valor_celda) for valor_celda in fila]) for fila in filas]),
        class_=nombre_clase,
    )


def calcular_densidad(valores, rejilla):
    ancho_banda = 1.06 * np.std(valores) * len(valores) ** -0.2 or 1
    return [
        sum(math.exp(-0.5 * ((coordenada_x - valor) / ancho_banda) ** 2) for valor in valores)
        / (len(valores) * ancho_banda * math.sqrt(2 * math.pi))
        for coordenada_x in rejilla
    ]


def crear_grafico_dispersion(seleccion):
    resoluciones = RESOLUCIONES if seleccion == "__todos__" else [seleccion]
    colores = {"PA": "#3578b8", "PI": "#d88b36", "PO": "#52a36b"}
    figura, eje = plt.subplots(figsize=(8, 5))

    for resolucion in resoluciones:
        datos_grafico = puntos(resolucion).sort_values("QBFASSET")
        if datos_grafico.empty:
            continue
        color = colores.get(resolucion, "#3578b8")
        eje.scatter(datos_grafico["QBFASSET"] / 1000, datos_grafico["COST"] / 1000, color=color, alpha=0.5, label=resolucion)
        suavizado = datos_grafico.rolling(max(3, len(datos_grafico) // 6), center=True, min_periods=1).mean()
        eje.plot(suavizado["QBFASSET"] / 1000, suavizado["COST"] / 1000, color=color, linewidth=2)

    eje.set_xlabel("Activos (millones de dólares)")
    eje.set_ylabel("Pérdidas (millones de dólares)")
    eje.ticklabel_format(axis="both", style="plain", useOffset=False)
    eje.legend()
    figura.tight_layout()
    return figura


def conclusiones(seleccion):
    resoluciones = RESOLUCIONES if seleccion == "__todos__" else [seleccion]
    textos = {
        "PA": (
            "Entre mayores son los activos, mayores son las pérdidas. Aunque la curva se estabiliza para los bancos "
            "de mayor tamaño, existe demasiada variabilidad (bancos con activos parecidos muestran pérdidas muy "
            "distintas). Se trata de bancos grandes."
        ),
        "PI": (
            "Las pérdidas aumentan con activos menores, pero después la curva se aplana y baja ligeramente. El patrón "
            "de aumento es menos claro que en PA, lo cual puede deberse a la escasez de datos. Se trata de bancos medianos."
        ),
        "PO": (
            "La curva muestra una tendencia ascendente, especialmente hacia los bancos con más activos. Sin embargo, "
            "no puede ser concluyente debido a la falta de observaciones. Se trata de bancos pequeños."
        ),
    }
    return crear_tabla(
        ["Tipo", "Conclusión"],
        [(resolucion, textos[resolucion]) for resolucion in resoluciones],
        "tabla-datos tabla-conclusiones",
    )


def crear_grafico_histograma(resolucion):
    figura, ejes = plt.subplots(1, 3, figsize=(12, 4))
    colores = ["#4385bd", "#d88b36", "#52a36b"]

    for eje, tamano, color in zip(ejes, ["Bajo", "Medio", "Alto"], colores):
        valores = BASE_DATOS.loc[
            (BASE_DATOS["RESTYPE1"] == resolucion) & (BASE_DATOS["Tamaño de activos"] == tamano),
            "COST",
        ].dropna().to_numpy() / 1000

        if len(valores):
            intervalos = 100 if resolucion == "PA" and tamano == "Alto" else 18
            eje.hist(valores, bins=intervalos, density=True, color=color, alpha=0.38)
            rejilla = np.linspace(valores.min(), valores.max() or valores.min() + 1, 100)
            eje.plot(rejilla, calcular_densidad(valores, rejilla), color=color)

        eje.set_title(tamano)
        eje.set_xlabel("Pérdidas (millones de dólares)")
        if resolucion == "PA" and tamano == "Alto":
            eje.set_xlim(0, 2000)
    ejes[0].set_ylabel("Densidad")
    figura.tight_layout()
    return figura


def resumen(resolucion):
    filas = []
    for tamano in ["Bajo", "Medio", "Alto"]:
        valores = BASE_DATOS.loc[
            (BASE_DATOS["RESTYPE1"] == resolucion) & (BASE_DATOS["Tamaño de activos"] == tamano),
            "COST",
        ].dropna()
        filas.append((
            tamano,
            str(len(valores)),
            f"{valores.mean() / 1000:,.2f}" if len(valores) else "—",
            f"{valores.median() / 1000:,.2f}" if len(valores) else "—",
        ))
    return filas


##############################################################################################
# Funciones para la pregunta 2

def datos_quiebras_por_anio(anio_inicio, tipo_periodo="decada"):

    if tipo_periodo == "completo":
        anio_inicio = 1992
        anio_final = 2026
    else:
        anio_final = anio_inicio + 9

    datos = BASE_DATOS[
        (BASE_DATOS["RESTYPE"] == "FAILURE") &
        (BASE_DATOS["FAILYR"] >= anio_inicio) &
        (BASE_DATOS["FAILYR"] <= anio_final)
    ].copy()

    resumen = datos.groupby("FAILYR").agg(
        cantidad_quiebras=("FAILYR", "size"),
        perdidas_totales=("COST", "sum")
    ).reset_index()

    resumen = resumen.set_index("FAILYR")

    resumen = resumen.reindex(
        range(anio_inicio, anio_final + 1),
        fill_value=0
    )

    resumen = resumen.reset_index()
    resumen = resumen.rename(columns={"index": "FAILYR"})

    resumen["perdidas_totales"] = resumen["perdidas_totales"] / 1000

    return resumen
##############################################################################################
# Interfaz de la aplicación
interfaz_aplicacion = ui.page_fluid(
    ui.tags.style("""
        body { background: #f7f8fa; color: #202735; }
        .seccion-contenido { background: white; border: 1px solid #e2e6ec;
          border-radius: 10px; padding: 1.1rem 1.25rem; margin: 1rem 0; }
        .tabla-datos { width: 100%; border-collapse: collapse; }
        .tabla-datos th, .tabla-datos td { border-bottom: 1px solid #e7eaf0;
          padding: .55rem .65rem; text-align: left; }
        .tabla-conclusiones { table-layout: fixed; }
        .tabla-conclusiones-histogramas { table-layout: fixed; }
        .tabla-conclusiones-histogramas th, .tabla-conclusiones-histogramas td { width: 100%; }
        .tabla-conclusiones th:first-child, .tabla-conclusiones td:first-child { width: 18%; }
        .tabla-conclusiones td:last-child { overflow-wrap: anywhere; white-space: normal; vertical-align: top; }
    """),
    ui.h1("Cierres y resoluciones bancarias de la Corporación Federal de Seguro de Depósitos"),

#Descripción de la base de datos
    ui.navset_tab(
        ui.nav_panel(
            "Descripción de los datos",
            ui.tags.section(
                ui.h3("Contexto del conjunto de datos"),
                ui.p(
                    "Se trabaja con el conjunto de datos denominado Datos sobre quiebras bancarias y asistencia, "
                    "proveniente de la Corporación Federal de Seguro de Depósitos (FDIC), una agencia del "
                    "gobierno de los Estados Unidos responsable de asegurar los depósitos bancarios y "
                    "administrar procesos relacionados con instituciones financieras en dificultades."
                ),
                ui.p(
                    "El conjunto registra datos desde  1992 hasta el 2011, contando con 866 observaciones en total. "
                    "Sin embargo, hubo 14 valores faltantes, por lo que la base cuenta con 852 observaciones válidas."
                ),
                ui.h3("Diccionario de la pregunta 1"),
                ui.h4("Tipos de resoluciones"),
                ui.tags.ul(
                    ui.tags.li(
                        "Compra y asunción (PA): otro banco asume los depósitos asegurados y no asegurados "
                        "y puede adquirir algunos activos y pasivos."
                    ),
                    ui.tags.li(
                        "Compra y asunción de depósitos asegurados (PI): otro banco asume únicamente los "
                        "depósitos asegurados."
                    ),
                    ui.tags.li(
                        "Pago (PO): la FDIC paga directamente a los depositantes asegurados y liquida los "
                        "activos del banco."
                    ),
                ),
                ui.h4("Tamaño de activos"),
                ui.tags.ul(
                    ui.tags.li("Bajo: hasta 85,43 millones de dólares."),
                    ui.tags.li("Medio: más de 85,43 millones y hasta 286,04 millones de dólares."),
                    ui.tags.li("Alto: más de 286,04 millones de dólares."),
                ),
                class_="seccion-contenido",
            ),
        ui.tags.section(
ui.h3("Diccionario de la pregunta 2"),

ui.h4("Variables utilizadas"),

        ui.tags.ul(
            ui.tags.li(
                "Año de quiebra (FAILYR): año en que se registró "
                "la quiebra de la institución bancaria."
            ),
            ui.tags.li(
                "Pérdidas económicas (COST): costo económico "
                "asociado con la quiebra bancaria. "
                "En los gráficos se expresa en millones de dólares."
            ),
            ui.tags.li(
                "Cantidad de quiebras: número de instituciones "
                "bancarias que quebraron durante cada año."
            )
        ),

ui.h4("Períodos históricos"),

        ui.tags.ul(
            ui.tags.li("Antes de la crisis financiera: 1992-2006."),
            ui.tags.li("Durante la crisis financiera: 2007-2012."),
            ui.tags.li("Después de la crisis financiera: 2013-2026.")
        ),
        class_="seccion-contenido",
    ),
),
#Pregunta 1
ui.nav_panel(
            "Pregunta 1",
            ui.tags.section(
                ui.h3(ui.strong(
                    "Relación de las pérdidas registradas según el tamaño de los activos "
                    "y el tipo de resolución"
                )),
                #Gráficos de dispersión
                ui.h4("Dispersión de los activos y pérdidas por tipo de resolución"),
                ui.input_select(
                    "resolucion_dispersion",
                    "Tipos de resolución para el diagrama de dispersión",
                    choices={"__todos__": "Todos", **{codigo: codigo for codigo in RESOLUCIONES}},
                    selected="__todos__",
                ),
                ui.layout_columns(
                    ui.output_plot("grafico_dispersion"),
                    ui.output_ui("tabla_conclusiones"),
                    col_widths=(8, 4),
                    fill=False,
                    gap="1rem",
                ),
                #Histogramas
                ui.h4("Distribución de las pérdidas por tamaño de activos"),
                ui.input_select(
                    "resolucion_histograma",
                    "Tipo de resolución para el histograma",
                    choices={codigo: codigo for codigo in RESOLUCIONES},
                    selected=RESOLUCIONES[0],
                ),
                ui.output_plot("grafico_histograma"),
                ui.output_ui("tabla_resumen"),
                ui.output_ui("conclusiones_histogramas"),
                class_="seccion-contenido",
            ),
        ),
        ui.nav_panel(
    "Pregunta 2",

    ui.h3(ui.strong("Evolución temporal de las quiebras bancarias")),
            
    ui.p(
        "¿Cómo ha evolucionado la cantidad de quiebras bancarias "
        "en Estados Unidos y qué relación presenta con las pérdidas "
        "económicas registradas durante el período de estudio?"
    ),
    ui.input_radio_buttons(
        "tipo_periodo",
        "Seleccione el tipo de análisis:",
    {
        "decada": "Períodos de 10 años",
        "completo": "Período completo (1992–2026)"
    },
    selected="decada",
    inline=True
),
    ui.panel_conditional(
    "input.tipo_periodo === 'decada'",
    ui.input_slider(
        "anio_inicio",
        "Seleccione el período de 10 años:",
        min=1992,
        max=2017,
        value=1992,
        step=1,
        sep=""
    )
),

    ui.h4("Cantidad de quiebras bancarias por año"),

    ui.layout_columns(
        ui.card(
            ui.output_plot("grafico_quiebras")
        ),
        ui.card(
            ui.h5("Interpretación de las quiebras bancarias"),
            ui.div(
                ui.output_text("interpretacion_quiebras"),
                style="white-space: pre-line; line-height: 1.8;"
            )
        ),
        col_widths=(8, 4)
    ),

    ui.h4("Pérdidas económicas totales por año"),

    ui.layout_columns(
        ui.card(
            ui.output_plot("grafico_perdidas")
        ),
        ui.card(
            ui.h5("Interpretación de las pérdidas económicas"),
            ui.div(
                ui.output_text("interpretacion_perdidas"),
                style="white-space: pre-line; line-height: 1.8;"
            )
        ),
        col_widths=(8, 4)
    ),

    ui.h4("Relación entre quiebras y pérdidas económicas por período histórico"),

    ui.layout_columns(
        ui.card(
            ui.output_plot("grafico_dispersion_historica")
        ),
        ui.card(
            ui.h5("Interpretación de la relación entre quiebras y pérdidas"),
            ui.div(
                ui.output_text("interpretacion_dispersion"),
                style="white-space: pre-line; line-height: 1.8;"
            )
        ),
        col_widths=(8, 4)
    ),

    ui.h4("Resumen de quiebras y pérdidas por año"),
    ui.output_data_frame("tabla_quiebras"),

),
        ui.nav_panel("Pregunta 3", ui.p("Espacio reservado para definir la tercera pregunta del trabajo en grupo.")),
        id="pestanas_principales",
        selected="Descripción de los datos",
    ),
)


def servidor(entrada, salida, sesion):

    @render.text
    def interpretacion_quiebras():

        tipo_periodo = entrada.tipo_periodo()

        # Obtener todos los años para comparar las etapas históricas
        resumen = datos_quiebras_por_anio(1992, "completo")

        if tipo_periodo == "decada":

            periodos = [
                (1992, 2006, "Antes de la crisis"),
                (2007, 2012, "Durante la crisis financiera"),
                (2013, 2026, "Después de la crisis")
            ]

            interpretaciones = []
            resultados = []

            for inicio, final, nombre in periodos:

                grupo = resumen[
                    (resumen["FAILYR"] >= inicio) &
                    (resumen["FAILYR"] <= final)
                ]

                total = int(grupo["cantidad_quiebras"].sum())
                promedio = grupo["cantidad_quiebras"].mean()

                maximo = grupo.loc[grupo["cantidad_quiebras"].idxmax()]
                anio_maximo = int(maximo["FAILYR"])
                cantidad_maxima = int(maximo["cantidad_quiebras"])

                interpretaciones.append(
                    f"{nombre} ({inicio}-{final}): "
                    f"Se registraron {total} quiebras bancarias, "
                    f"con un promedio de {promedio:.2f} quiebras anuales. "
                    f"El año con mayor cantidad fue {anio_maximo}, "
                    f"con {cantidad_maxima} quiebras."
                )

                resultados.append((nombre, promedio))

            mayor = max(resultados, key=lambda x: x[1])
            menor = min(resultados, key=lambda x: x[1])

            comparacion = (
                "\n\nDiferencias relevantes: "
                f"La etapa con mayor promedio anual de quiebras fue "
                f"{mayor[0]}, mientras que la de menor promedio fue "
                f"{menor[0]}. "
                "Esta comparación permite identificar diferencias "
                "en la frecuencia de las quiebras bancarias entre "
                "las tres etapas históricas."
            )

            return "\n\n".join(interpretaciones) + comparacion

        else:

            total = int(resumen["cantidad_quiebras"].sum())
            promedio = resumen["cantidad_quiebras"].mean()

            maximo = resumen.loc[resumen["cantidad_quiebras"].idxmax()]
            anio_maximo = int(maximo["FAILYR"])
            cantidad_maxima = int(maximo["cantidad_quiebras"])

            return (
                f"Durante el período completo de 1992 a 2026 "
                f"se registraron {total} quiebras bancarias, "
                f"con un promedio de {promedio:.2f} quiebras por año. "
                f"El año con mayor cantidad fue {anio_maximo}, "
                f"con {cantidad_maxima} quiebras. "
                "La evolución temporal permite identificar períodos "
                "de mayor concentración de quiebras, especialmente "
                "en los años asociados con la crisis financiera, "
                "así como etapas posteriores con una menor frecuencia."
            )
    @render.text
    def interpretacion_perdidas():

        tipo_periodo = entrada.tipo_periodo()
        resumen = datos_quiebras_por_anio(1992, "completo")

        if tipo_periodo == "decada":

            periodos = [
                (1992, 2006, "Antes de la crisis"),
                (2007, 2012, "Durante la crisis financiera"),
                (2013, 2026, "Después de la crisis")
            ]

            interpretaciones = []
            resultados = []

            for inicio, final, nombre in periodos:

                grupo = resumen[
                    (resumen["FAILYR"] >= inicio) &
                    (resumen["FAILYR"] <= final)
                ]

                total = grupo["perdidas_totales"].sum()
                promedio = grupo["perdidas_totales"].mean()

                maximo = grupo.loc[grupo["perdidas_totales"].idxmax()]
                anio_maximo = int(maximo["FAILYR"])
                perdida_maxima = maximo["perdidas_totales"]

                interpretaciones.append(
                    f"{nombre} ({inicio}-{final}): "
                    f"Las pérdidas económicas acumuladas fueron de "
                    f"{total:,.2f} millones de dólares, "
                    f"con un promedio anual de {promedio:,.2f} millones. "
                    f"El año con mayores pérdidas fue {anio_maximo}, "
                    f"con {perdida_maxima:,.2f} millones."
                )

                resultados.append((nombre, promedio))

            mayor = max(resultados, key=lambda x: x[1])
            menor = min(resultados, key=lambda x: x[1])

            diferencias = (
                f"Diferencias relevantes: La etapa con mayores pérdidas "
                f"económicas promedio por año fue {mayor[0]}, "
                f"mientras que la de menores pérdidas fue {menor[0]}. "
                f"Esto evidencia diferencias en el impacto económico "
                f"de las quiebras bancarias entre las tres etapas históricas."
            )

            return "\n\n".join(interpretaciones + [diferencias])

        else:

            total = resumen["perdidas_totales"].sum()
            promedio = resumen["perdidas_totales"].mean()

            maximo = resumen.loc[resumen["perdidas_totales"].idxmax()]
            anio_maximo = int(maximo["FAILYR"])
            perdida_maxima = maximo["perdidas_totales"]

            return (
                f"Durante el período completo de 1992 a 2026, "
                f"las pérdidas económicas acumuladas fueron de "
                f"{total:,.2f} millones de dólares, "
                f"con un promedio anual de {promedio:,.2f} millones.\n\n"
                f"El año con mayores pérdidas fue {anio_maximo}, "
                f"con {perdida_maxima:,.2f} millones de dólares.\n\n"
                f"La evolución temporal muestra que las pérdidas "
                f"no fueron constantes a lo largo del período. "
                f"Se observan años con pérdidas considerablemente "
                f"más elevadas que otros, lo que permite identificar "
                f"momentos de mayor impacto económico asociado "
                f"a las quiebras bancarias."
            )
    @render.text
    def interpretacion_dispersion():

        tipo_periodo = entrada.tipo_periodo()
        resumen = datos_quiebras_por_anio(1992, "completo")

        if tipo_periodo == "decada":

            periodos = [
                (1992, 2006, "Antes de la crisis"),
                (2007, 2012, "Durante la crisis financiera"),
                (2013, 2026, "Después de la crisis")
            ]

            interpretaciones = []

            for inicio, final, nombre in periodos:

                grupo = resumen[
                    (resumen["FAILYR"] >= inicio) &
                    (resumen["FAILYR"] <= final)
                ]

                total_quiebras = grupo["cantidad_quiebras"].sum()
                total_perdidas = grupo["perdidas_totales"].sum()

                interpretaciones.append(
                    f"{nombre} ({inicio}-{final}): "
                    f"Se registraron {total_quiebras:.0f} quiebras "
                    f"bancarias, con pérdidas económicas acumuladas "
                    f"de {total_perdidas:.2f} millones de dólares. "
                    f"Estos resultados permiten observar la evolución "
                    f"de las quiebras bancarias y las pérdidas económicas "
                    f"durante esta etapa histórica."
                )

            return "\n\n".join(interpretaciones)

        else:

            total_quiebras = resumen["cantidad_quiebras"].sum()
            total_perdidas = resumen["perdidas_totales"].sum()

            anio_mas_quiebras = resumen.loc[
                resumen["cantidad_quiebras"].idxmax(), "FAILYR"
            ]

            anio_mas_perdidas = resumen.loc[
                resumen["perdidas_totales"].idxmax(), "FAILYR"
            ]

            return (
                f"Durante el período completo de 1992 a 2026 "
                f"se registraron {total_quiebras:.0f} quiebras "
                f"bancarias, con pérdidas económicas acumuladas "
                f"de {total_perdidas:.2f} millones de dólares.\n\n"
                f"El año con mayor cantidad de quiebras bancarias "
                f"fue {anio_mas_quiebras}, mientras que el año "
                f"con mayores pérdidas económicas fue "
                f"{anio_mas_perdidas}.\n\n"
                f"Estos resultados permiten observar la evolución "
                f"de las quiebras bancarias y las pérdidas económicas "
                f"a lo largo del período de estudio. "
                f"Es importante considerar que las pérdidas "
                f"económicas no dependen únicamente de la cantidad "
                f"de instituciones que quebraron, sino también "
                f"de su magnitud."
            )
            
    @render.ui
    def tabla_conclusiones():
        return conclusiones(entrada.resolucion_dispersion())

    @render.plot
    def grafico_dispersion():
        return crear_grafico_dispersion(entrada.resolucion_dispersion())

    @render.plot
    def grafico_histograma():
        return crear_grafico_histograma(entrada.resolucion_histograma())

    @render.ui
    def tabla_resumen():
        return crear_tabla(
            ["Tamaño de activos", "Observaciones", "Media de las pérdidas (millones de dólares)", "Mediana de las pérdidas (millones de dólares)"],
            resumen(entrada.resolucion_histograma()),
        )

    @render.ui
    def conclusiones_histogramas():
        return crear_tabla(
            ["Conclusiones"],
            [("Las distribuciones de las pérdidas para los bancos con activos bajos y medios poseen una "
              "distribución más marcada y muy asimétrica hacia la derecha, diferenciándose de los bancos "
              "con activos grandes, que no cuentan con una distribución apreciable, lo cual es probablemente "
              "debido a la escasez de datos",)],
            "tabla-datos tabla-conclusiones-histogramas",
        )

    # Gráfico de barras para la pregunta 2

    @render.plot
    def grafico_quiebras():

        anio_inicio = entrada.anio_inicio()
            
        tipo_periodo = entrada.tipo_periodo()
        
        resumen = datos_quiebras_por_anio(anio_inicio, tipo_periodo)

        fig, ax = plt.subplots(figsize=(10, 3.5))

        ax.bar(
            resumen["FAILYR"],
            resumen["cantidad_quiebras"],
            color="steelblue"
        )

        ax.set_title("Cantidad de quiebras bancarias por año")

        ax.set_xlabel("Año")
        ax.set_ylabel("Cantidad de quiebras")

        ax.set_xticks(resumen["FAILYR"][::5] if tipo_periodo == "completo" else resumen["FAILYR"])

        plt.xticks(rotation=45)

        plt.tight_layout()

        return fig

    
    # Gráfico de líneas de pérdidas económicas para la pregunta 2

    @render.plot
    def grafico_perdidas():

        anio_inicio = entrada.anio_inicio()
        tipo_periodo = entrada.tipo_periodo()

        resumen = datos_quiebras_por_anio(anio_inicio, tipo_periodo)
        fig, ax = plt.subplots(figsize=(10, 3.5))

        ax.plot(
            resumen["FAILYR"],
            resumen["perdidas_totales"],
            marker="o",
            color="darkred",
            linewidth=2
        )

        ax.set_title("Pérdidas económicas por año")

        ax.set_xlabel("Año")
        ax.set_ylabel("Pérdidas totales (millones de dólares)")

        ax.set_xticks(resumen["FAILYR"][::5] if tipo_periodo == "completo" else resumen["FAILYR"])

        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)

        plt.tight_layout()

        return fig
        
    # Gráfico de dispersión por períodos históricos para la pregunta 2

    @render.plot
    def grafico_dispersion_historica():

        anio_inicio = entrada.anio_inicio()
        tipo_periodo = entrada.tipo_periodo()

        resumen = datos_quiebras_por_anio(anio_inicio, tipo_periodo)

        fig, ax = plt.subplots(figsize=(10, 3.5))

        periodos = [
            (1992, 2006, "Antes de la crisis (1992-2006)", "steelblue"),
            (2007, 2012, "Crisis financiera (2007-2012)", "darkred"),
            (2013, 2026, "Después de la crisis (2013-2026)", "seagreen")
        ]

        for inicio, final, nombre, color in periodos:

            grupo = resumen[
                (resumen["FAILYR"] >= inicio) &
                (resumen["FAILYR"] <= final)
            ]

            if not grupo.empty:
                ax.scatter(
                    grupo["cantidad_quiebras"],
                    grupo["perdidas_totales"],
                    label=nombre,
                    color=color,
                    s=70,
                    alpha=0.8
                )

        ax.set_title("Relación entre quiebras y pérdidas económicas")
        ax.set_xlabel("Cantidad de quiebras bancarias")
        ax.set_ylabel("Pérdidas económicas (millones de dólares)")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig
        
    # Tabla interactiva para la pregunta 2

    @render.data_frame
    def tabla_quiebras():

        anio_inicio = entrada.anio_inicio()
        tipo_periodo = entrada.tipo_periodo()
        resumen = datos_quiebras_por_anio(anio_inicio, tipo_periodo)
        
        resumen["perdidas_totales"] = resumen["perdidas_totales"].round(2)

        resumen = resumen.rename(columns={
            "FAILYR": "Año",
            "cantidad_quiebras": "Cantidad de quiebras",
            "perdidas_totales": "Pérdidas totales (millones de dólares)"
        })

        resumen["Pérdidas totales (millones de dólares)"] = resumen["Pérdidas totales (millones de dólares)"].map(lambda x: f"{x:.2f}")

        return render.DataGrid(resumen, filters=False)

app = App(interfaz_aplicacion, servidor)
