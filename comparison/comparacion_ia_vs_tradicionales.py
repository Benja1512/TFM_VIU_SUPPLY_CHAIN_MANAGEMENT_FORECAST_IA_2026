import pandas as pd
import matplotlib.pyplot as plt
import os

def comparar_ia_vs_tradicionales():
    # Archivos de métricas
    rutas = {
        'SES': 'results/metricas/ses_metricas.csv',
        'MA-3': 'results/metricas/ma3_metricas.csv',
        'Prophet': 'results/metricas/prophet_metricas.csv',
        'Random Forest': 'results/metricas/random_forest_metricas.csv',
        'ANN': 'results/metricas/ann_metricas.csv'
    }

    # Cargar métricas en un único DataFrame
    metricas = []
    for modelo, ruta in rutas.items():
        df = pd.read_csv(ruta)
        df['modelo'] = modelo
        metricas.append(df)

    df_all = pd.concat(metricas, ignore_index=True)
    df_all = df_all[['modelo', 'MAE', 'RMSE']]  # columnas relevantes

    # Guardar tabla consolidada
    os.makedirs('results/metricas', exist_ok=True)
    df_all.to_csv('results/metricas/comparacion_ia_vs_tradicionales.csv', index=False)

    # ===================== Gráfico =====================
    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # MAE
    axes[0].bar(df_all['modelo'], df_all['MAE'], color='skyblue')
    axes[0].set_title('Comparación MAE')
    axes[0].set_ylabel('MAE')
    axes[0].set_xticks(range(len(df_all['modelo'])))
    axes[0].set_xticklabels(df_all['modelo'], rotation=30, ha="right")

    # RMSE
    axes[1].bar(df_all['modelo'], df_all['RMSE'], color='salmon')
    axes[1].set_title('Comparación RMSE')
    axes[1].set_ylabel('RMSE')
    axes[1].set_xticks(range(len(df_all['modelo'])))
    axes[1].set_xticklabels(df_all['modelo'], rotation=30, ha="right")

    plt.suptitle("Comparación de Modelos Tradicionales vs IA", fontsize=14)

    os.makedirs('results/graficos', exist_ok=True)
    plt.savefig('results/graficos/comparacion_ia_vs_tradicionales.png', dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\n✅ Comparación consolidada guardada en:")
    print(" - CSV: results/metricas/comparacion_ia_vs_tradicionales.csv")
    print(" - Gráfico: results/graficos/comparacion_ia_vs_tradicionales.png")

if __name__ == "__main__":
    comparar_ia_vs_tradicionales()
