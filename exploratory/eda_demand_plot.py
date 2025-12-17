# exploratory/eda_demand_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def run_eda():
    # Leer CSV
    df = pd.read_csv('data/demanda_semanal.csv', sep=';')
    df.columns = df.columns.str.strip()
    df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
    df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
    df = df.dropna()

    print("Primeras filas del dataset:")
    print(df.head())

    # Crear carpetas si no existen
    os.makedirs('results/graficos', exist_ok=True)
    os.makedirs('results/metricas', exist_ok=True)

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df['semana'], df['demanda'], marker='o')
    plt.title('Demanda semanal')
    plt.xlabel('Semana')
    plt.ylabel('Demanda')
    plt.grid(True)
    plt.savefig('results/graficos/demanda_semanal.png', dpi=300)
    plt.show()

    # Resumen estadístico
    print("\nResumen estadístico:")
    print(df['demanda'].describe())
    df['demanda'].describe().to_csv('results/metricas/resumen_estadistico.csv')

