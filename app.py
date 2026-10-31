from shiny import App, ui

######################################################################################################
# Estética

def html_table(headers, rows):
    return ui.tags.table(
        ui.tags.thead(ui.tags.tr(*[ui.tags.th(h, scope="col") for h in headers])),
        ui.tags.tbody(
            *[
                ui.tags.tr(*[ui.tags.td(value) for value in row])
                for row in rows
            ]
        ),
        class_="data-table",
    )


def section(title, *content):
    return ui.tags.section(
        ui.h3(title),
        *content,
        class_="content-section",
    )


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body { background: #f7f8fa; color: #202735; }
        .app-header { padding: 2rem 0 1rem; }
        .app-header h1 { font-size: 1.8rem; font-weight: 650; margin-bottom: .35rem; }
        .app-subtitle { color: #5c6678; margin-bottom: 1.25rem; }
        .content-section { background: #fff; border: 1px solid #e2e6ec;
          border-radius: 10px; padding: 1.1rem 1.25rem; margin: 1rem 0; }
        .content-section h3 { font-size: 1.1rem; margin: 0 0 .8rem; }
        .data-table { width: 100%; border-collapse: collapse; margin: .4rem 0; }
        .data-table th, .data-table td { border-bottom: 1px solid #e7eaf0;
          padding: .55rem .65rem; text-align: left; vertical-align: top; }
        .data-table th { font-weight: 600; color: #394457; }
        .note { color: #5c6678; font-size: .94rem; }
        .placeholder { color: #5c6678; padding: .5rem 0; }
        @media (max-width: 640px) {
          .content-section { padding: .9rem; }
          .data-table th, .data-table td { padding: .45rem .35rem; }
        }
        """
    ),
    ui.tags.header(
        ui.h1("Cierres y resoluciones bancarias de la Corporación Federal de Seguro de Depósitos"),
        class_="app-header",
    ),

######################################################################################################
    #Descripción de los datos

    ui.navset_tab(
        ui.nav_panel(
            "Descripción de los datos",
            section(
        "Contexto del conjunto de datos",
        ui.tags.p(
            "Se trabaja con el conjunto de datos denominado Datos sobre quiebras bancarias y asistencia, "
            "proveniente de la Corporación Federal de Seguro de Depósitos (FDIC), una agencia del "
            "gobierno de los Estados Unidos responsable de asegurar los depósitos bancarios y "
            "administrar procesos relacionados con instituciones financieras en dificultades."
        ),
        ui.tags.p(
            "El conjunto registra datos desde  1992 hasta el 2011, contando con 866 observaciones en total. "
            "Sin embargo, hubo 14 valores faltantes, por lo que la base cuenta con 852 observaciones válidas."
        ),
    ),
),
######################################################################################################
     #Pregunta 1    
        ui.nav_panel(
            "Pregunta 1",
            section(
                "Relación de las pérdidas registradas de los cierres bancarios según el tamaño de los activos y el tipo de resolución",
                ui.tags.p(
                    "Esta pestaña queda preparada para incorporar el análisis correspondiente.",
                    class_="placeholder",
                ),
            ),
        ),
        ui.nav_panel(
            "Pregunta 2",
            section(
                "Pregunta 2",
                ui.tags.p("Espacio reservado para definir la segunda pregunta del trabajo en grupo.", class_="placeholder"),
            ),
        ),
        ui.nav_panel(
            "Pregunta 3",
            section(
                "Pregunta 3",
                ui.tags.p("Espacio reservado para definir la tercera pregunta del trabajo en grupo.", class_="placeholder"),
            ),
        ),
        id="main_tabs",
        selected="Descripción de los datos",
    ),
)


def server(input, output, session):
    pass


app = App(app_ui, server)
