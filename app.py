from shiny.express import input, render, ui

ui.input_slider("n", "Número", 0, 100, 50)

@render.text
def resultado():
    return f"Elegiste {input.n()}"
