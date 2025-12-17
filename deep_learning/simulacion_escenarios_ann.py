import pandas as pd
import os
import matplotlib.pyplot as plt

def generar_escenarios_ann():
    # Cargar predicciones de ANN
    df = pd.read_csv('results/metricas/ann_predicciones.csv')

    # Crear escenarios A–E con ±10% y ±20%
    df['Escenario_A'] = df['demanda_predicha'] * 0.80  # -20%
    df['Escenario_B'] = df['demanda_predicha'] * 0.90  # -10%
    df['Escenario_C'] = df['demanda_predicha']         # base
    df['Escenario_D'] = df['demanda_predicha'] * 1.10  # +10%
    df['Escenario_E'] = df['demanda_predicha'] * 1.20  # +20%

    # Crear carpeta si no existe
    os.makedirs('results/escenarios', exist_ok=True)

    # Guardar archivo CSV
    df.to_csv('results/escenarios/ann_escenarios.csv', index=False)

    # Mostrar resumen en consola
    print("\n✅ Escenarios A–E generados con éxito.")
    print("📁 Guardado en: results/escenarios/ann_escenarios.csv\n")
    print(df[['semana', 'demanda_real', 'Escenario_A', 'Escenario_B', 'Escenario_C', 'Escenario_D', 'Escenario_E']].round(2))

    # Gráfico comparativo
    plt.figure(figsize=(12, 6))
    plt.plot(df['semana'], df['demanda_real'], label='Real', marker='o', linestyle='--')
    plt.plot(df['semana'], df['Escenario_A'], label='Escenario A (-20%)')
    plt.plot(df['semana'], df['Escenario_B'], label='Escenario B (-10%)')
    plt.plot(df['semana'], df['Escenario_C'], label='Escenario C (Base)')
    plt.plot(df['semana'], df['Escenario_D'], label='Escenario D (+10%)')
    plt.plot(df['semana'], df['Escenario_E'], label='Escenario E (+20%)')
    plt.title('Simulación de Escenarios Logísticos – Modelo ANN')
    plt.xlabel('Semana')
    plt.ylabel('Unidades de Demanda')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/escenarios/ann_escenarios_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generar_escenarios_ann()

