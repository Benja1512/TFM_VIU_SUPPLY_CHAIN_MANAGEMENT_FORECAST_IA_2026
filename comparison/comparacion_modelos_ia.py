import pandas as pd
import matplotlib.pyplot as plt
import os

def comparar_modelos_ia():
    # Rutas de métricas
    ruta_prophet = 'results/metricas/prophet_metricas.csv'
    ruta_rf = 'results/metricas/random_forest_metricas.csv'
    ruta_ann = 'results/metricas/ann_metricas.csv'

    # Leer métricas
    df_prophet = pd.read_csv(ruta_prophet)
    df_rf = pd.read_csv(ruta_rf)
    df_ann = pd.read_csv(ruta_ann)

    # Unir en un solo DataFrame
    df_comparacion = pd.concat([df_prophet, df_rf, df_ann], ignore_index=True)

    # Guardar CSV consolidado
    os.makedirs('results/metricas', exist_ok=True)
    df_comparacion.to_csv('results/metricas/comparacion_modelos_ia.csv', index=False)

    print("\n==============================")
    print("✅ Comparación IA completada")
    print(df_comparacion)
    print("==============================\n")

    # Crear gráfico comparativo
    os.makedirs('results/graficos', exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # MAE
    ax[0].bar(df_comparacion['modelo'], df_comparacion['MAE'], color='skyblue')
    ax[0].set_title('Comparación MAE')
    ax[0].set_ylabel('MAE')

    # RMSE
    ax[1].bar(df_comparacion['modelo'], df_comparacion['RMSE'], color='salmon')
    ax[1].set_title('Comparación RMSE')
    ax[1].set_ylabel('RMSE')

    plt.suptitle('Comparación de Modelos IA (Prophet, RF, ANN)')
    plt.savefig('results/graficos/comparacion_modelos_ia.png', dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    comparar_modelos_ia()
