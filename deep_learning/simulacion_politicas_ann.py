import pandas as pd
import os
import matplotlib.pyplot as plt

def generar_politicas_ann():
    # Cargar predicciones de ANN
    df = pd.read_csv('results/metricas/ann_predicciones.csv')

    # Definir 3 políticas logísticas
    df['Politica_A'] = df['demanda_predicha'] * 1.20  # reposición alta (+20%)
    df['Politica_B'] = df['demanda_predicha']         # reposición media (base)
    df['Politica_C'] = df['demanda_predicha'] * 0.80  # reposición conservadora (-20%)

    # Crear carpetas de salida
    os.makedirs('results/escenarios', exist_ok=True)
    os.makedirs('results/graficos', exist_ok=True)

    # Guardar archivo CSV
    df.to_csv('results/escenarios/ann_politicas.csv', index=False)

    # Mostrar resumen en terminal
    print("\n✅ Políticas A–B–C generadas exitosamente.")
    print("📁 Guardado en: results/escenarios/ann_politicas.csv\n")
    print(df[['semana', 'demanda_real', 'Politica_A', 'Politica_B', 'Politica_C']].round(2))

    # ============================
    # 📊 GENERAR GRÁFICO POLÍTICAS
    # ============================
    plt.figure(figsize=(12, 6))

    plt.plot(df['semana'], df['demanda_real'], marker='o', label='Demanda Real')
    plt.plot(df['semana'], df['Politica_A'], marker='x', label='Política A (+20%)')
    plt.plot(df['semana'], df['Politica_B'], marker='s', label='Política B (Base)')
    plt.plot(df['semana'], df['Politica_C'], marker='d', label='Política C (-20%)')

    plt.title('Simulación de Políticas Logísticas – Modelo ANN (Keras)')
    plt.xlabel('Semana')
    plt.ylabel('Unidades de demanda / reposición')
    plt.grid(True)
    plt.legend()

    output_path = 'results/graficos/ann_politicas.png'
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"📊 Gráfico generado y guardado en: {output_path}\n")

if __name__ == "__main__":
    generar_politicas_ann()
