import pandas as pd
import plotly.graph_objects as go
import os

def plotly_series_ia():
    df = pd.read_csv("results/metricas/comparacion_series_ia.csv")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['semana'], y=df['real'],
                             mode='lines+markers', name='Real'))
    fig.add_trace(go.Scatter(x=df['semana'], y=df['prophet'],
                             mode='lines+markers', name='Prophet'))
    fig.add_trace(go.Scatter(x=df['semana'], y=df['random_forest'],
                             mode='lines+markers', name='Random Forest'))
    fig.add_trace(go.Scatter(x=df['semana'], y=df['ann'],
                             mode='lines+markers', name='ANN'))

    fig.update_layout(title="Comparación de Predicciones (IA) vs Real - Interactivo",
                      xaxis_title="Semana",
                      yaxis_title="Demanda")

    os.makedirs("results/graficos", exist_ok=True)
    fig.write_html("results/graficos/comparacion_series_ia_interactivo.html")

    print("\n✅ Gráfico interactivo guardado en results/graficos/comparacion_series_ia_interactivo.html")
    fig.show()

if __name__ == "__main__":
    plotly_series_ia()
