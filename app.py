from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, render, ui

#Archivo 
DATA = pd.read_csv(Path(__file__).with_name("Failures_10_26_2026.csv"))

##############################################################################################
#Funciones para la pregunta 1 
RESOLUTIONS = DATA["RESTYPE1"].value_counts().head(3).index.tolist()
ASSET_Q1, ASSET_Q2 = DATA["QBFASSET"].quantile([1 / 3, 2 / 3])
DATA["Tamaño de activos"] = np.select(
    [DATA["QBFASSET"] <= ASSET_Q1, DATA["QBFASSET"] <= ASSET_Q2],
    ["Bajo", "Medio"],
    default="Alto",
)


def points(resolution):
    rows = DATA.loc[DATA["RESTYPE1"] == resolution, ["QBFASSET", "COST"]].dropna()
    for column in ("QBFASSET", "COST"):
        low, high = rows[column].quantile([0.25, 0.75])
        spread = high - low
        rows = rows[rows[column].between(low - 1.5 * spread, high + 1.5 * spread)]
    return rows


def table(headers, rows, class_name="data-table"):
    return ui.tags.table(
        ui.tags.thead(ui.tags.tr(*[ui.tags.th(h) for h in headers])),
        ui.tags.tbody(*[ui.tags.tr(*[ui.tags.td(v) for v in row]) for row in rows]),
        class_=class_name,
    )


def density(values, grid):
    bandwidth = 1.06 * np.std(values) * len(values) ** -0.2 or 1
    return [
        sum(math.exp(-0.5 * ((x - value) / bandwidth) ** 2) for value in values)
        / (len(values) * bandwidth * math.sqrt(2 * math.pi))
        for x in grid
    ]


def scatter_figure(selection):
    resolutions = RESOLUTIONS if selection == "__all__" else [selection]
    colors = {"PA": "#3578b8", "PI": "#d88b36", "PO": "#52a36b"}
    fig, ax = plt.subplots(figsize=(8, 5))

    for resolution in resolutions:
        data = points(resolution).sort_values("QBFASSET")
        if data.empty:
            continue
        color = colors.get(resolution, "#3578b8")
        ax.scatter(data["QBFASSET"] / 1000, data["COST"] / 1000, color=color, alpha=0.5, label=resolution)
        smooth = data.rolling(max(3, len(data) // 6), center=True, min_periods=1).mean()
        ax.plot(smooth["QBFASSET"] / 1000, smooth["COST"] / 1000, color=color, linewidth=2)

    ax.set_xlabel("Activos (millones de dólares)")
    ax.set_ylabel("Pérdidas (millones de dólares)")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.legend()
    fig.tight_layout()
    return fig


def conclusions(selection):
    resolutions = RESOLUTIONS if selection == "__all__" else [selection]
    texts = {
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
    return table(
        ["Tipo", "Conclusión"],
        [(resolution, texts[resolution]) for resolution in resolutions],
        "data-table conclusion-table",
    )


def histogram_figure(resolution):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4385bd", "#d88b36", "#52a36b"]

    for ax, size, color in zip(axes, ["Bajo", "Medio", "Alto"], colors):
        values = DATA.loc[
            (DATA["RESTYPE1"] == resolution) & (DATA["Tamaño de activos"] == size),
            "COST",
        ].dropna().to_numpy() / 1000

        if len(values):
            bins = 100 if resolution == "PA" and size == "Alto" else 18
            ax.hist(values, bins=bins, density=True, color=color, alpha=0.38)
            grid = np.linspace(values.min(), values.max() or values.min() + 1, 100)
            ax.plot(grid, density(values, grid), color=color)

        ax.set_title(f"{size} (n={len(values)})")
        ax.set_xlabel("Pérdidas (millones de dólares)")
        if resolution == "PA" and size == "Alto":
            ax.set_xlim(0, 2000)
    axes[0].set_ylabel("Densidad")
    fig.tight_layout()
    return fig


def summary(resolution):
    rows = []
    for size in ["Bajo", "Medio", "Alto"]:
        values = DATA.loc[
            (DATA["RESTYPE1"] == resolution) & (DATA["Tamaño de activos"] == size),
            "COST",
        ].dropna()
        rows.append((
            size,
            str(len(values)),
            f"{values.mean() / 1000:,.2f}" if len(values) else "—",
            f"{values.median() / 1000:,.2f}" if len(values) else "—",
        ))
    return rows


##############################################################################################


app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background: #f7f8fa; color: #202735; }
        .content-section { background: white; border: 1px solid #e2e6ec;
          border-radius: 10px; padding: 1.1rem 1.25rem; margin: 1rem 0; }
        .data-table { width: 100%; border-collapse: collapse; }
        .data-table th, .data-table td { border-bottom: 1px solid #e7eaf0;
          padding: .55rem .65rem; text-align: left; }
        .conclusion-table { table-layout: fixed; }
        .conclusion-table th:first-child, .conclusion-table td:first-child { width: 18%; }
        .conclusion-table td:last-child { overflow-wrap: anywhere; white-space: normal; vertical-align: top; }
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
                        "Purchase and Assumption (PA): otro banco asume los depósitos asegurados y no asegurados "
                        "y puede adquirir algunos activos y pasivos."
                    ),
                    ui.tags.li(
                        "Purchase and Assumption of insured deposits (PI): otro banco asume únicamente los "
                        "depósitos asegurados."
                    ),
                    ui.tags.li(
                        "Payout (PO): la FDIC paga directamente a los depositantes asegurados y liquida los "
                        "activos del banco."
                    ),
                ),
                ui.h4("Tamaño de activos"),
                ui.tags.ul(
                    ui.tags.li("Bajo: hasta 85,43 millones de dólares."),
                    ui.tags.li("Medio: más de 85,43 millones y hasta 286,04 millones de dólares."),
                    ui.tags.li("Alto: más de 286,04 millones de dólares."),
                ),
                class_="content-section",
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
                    "scatter_resolution",
                    "Tipos de resolución para el diagrama de dispersión",
                    choices={"__all__": "Todos", **{code: code for code in RESOLUTIONS}},
                    selected="__all__",
                ),
                ui.layout_columns(
                    ui.output_plot("scatter_plot"),
                    ui.output_ui("conclusions_table"),
                    col_widths=(8, 4),
                    fill=False,
                    gap="1rem",
                ),
                #Histogramas
                ui.h4("Distribución de las pérdidas por tamaño de activos"),
                ui.input_select(
                    "resolution",
                    "Tipo de resolución para el histograma",
                    choices={code: code for code in RESOLUTIONS},
                    selected=RESOLUTIONS[0],
                ),
                ui.output_plot("histogram_plot"),
                ui.output_ui("summary_table"),
                class_="content-section",
            ),
        ),
        ui.nav_panel("Pregunta 2", ui.p("Espacio reservado para definir la segunda pregunta del trabajo en grupo.")),
        ui.nav_panel("Pregunta 3", ui.p("Espacio reservado para definir la tercera pregunta del trabajo en grupo.")),
        id="main_tabs",
        selected="Descripción de los datos",
    ),
)


def server(input, output, session):
    @render.ui
    def conclusions_table():
        return conclusions(input.scatter_resolution())

    @render.plot
    def scatter_plot():
        return scatter_figure(input.scatter_resolution())

    @render.plot
    def histogram_plot():
        return histogram_figure(input.resolution())

    @render.ui
    def summary_table():
        return table(
            ["Tamaño de activos", "Observaciones", "Media de las pérdidas (millones de dólares)", "Mediana de las pérdidas (millones de dólares)"],
            summary(input.resolution()),
        )


app = App(app_ui, server)
