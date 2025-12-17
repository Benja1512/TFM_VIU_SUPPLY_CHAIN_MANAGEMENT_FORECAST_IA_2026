import pandas as pd
import numpy as np
import os

# Parámetros
stock_seguridad = 50
stock_inicial = 100
horizonte = 12

# Ruta base
base_path = 'D:/Pronostico'

# Cargar demanda real
demanda = pd.read_csv(os.path.join(base_path, 'data', 'demanda_semanal.csv'))
demanda_real = demanda['y'][-horizonte:].values

# Rutas por modelo en orden: MA-3, SES, Prophet, RF, ANN
modelos = {
    'MA-3': 'results/ma3_forecast_series.csv',
    'SES': 'results/ses_forecast_series.csv',
    'Prophet': 'results/prophet_forecast_series.csv',
    'RandomForest': 'prophet_model/rf_forecast_series.csv',
    'ANN': 'deep_learning/ann_predicciones.csv'
}

# Guardar resultados aquí
resultados = []

for modelo, ruta_relativa in modelos.items():
    ruta = os.path.join(base_path, ruta_relativa)
    df = pd.read_csv(ruta)

    # Tomar solo predicciones finales
    if 'yhat' in df.columns:
        pred = df['yhat'].values[-horizonte:]
    elif 'y' in df.columns:
        pred = df['y'].values[-horizonte:]
    else:
        raise ValueError(f'No se encontró columna válida en {modelo}')

    # Simulación logística
    inventario = stock_inicial
    stockouts = 0
    sobrestock = 0
    nivel_servicio = 0
    inventario_final = []

    for t in range(horizonte):
        demanda_t = demanda_real[t]
        llegada = pred[t] + stock_seguridad
        inventario += llegada

        if demanda_t > inventario:
            stockouts += demanda_t - inventario
            nivel_servicio += inventario
            inventario = 0
        else:
            nivel_servicio += demanda_t
            inventario -= demanda_t

        if inventario > stock_seguridad:
            sobrestock += inventario - stock_seguridad

        inventario_final.append(inventario)

    resultados.append({
        'Modelo': modelo,
        'Stockout Total (u)': round(stockouts, 2),
        'Sobrestock Total (u)': round(sobrestock, 2),
        'Inventario Final Medio (u)': round(np.mean(inventario_final), 2),
        'Nivel de Servicio (%)': round((nivel_servicio / sum(demanda_real)) * 100, 2)
    })

# Exportar resultados
df_resultados = pd.DataFrame(resultados)
output_csv = os.path.join(base_path, 'results', 'resultados_logisticos_modelos.csv')
df_resultados.to_csv(output_csv, index=False)

print("✅ Resultados guardados en:", output_csv)
print(df_resultados)
