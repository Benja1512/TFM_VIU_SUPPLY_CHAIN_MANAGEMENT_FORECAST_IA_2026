# ============================================================
# PROPHET FORECAST (OUT-OF-SAMPLE 41–52)
# Producto: Leche Entera 1L - Retail urbano simulado
# Autor: Elaboración propia (TFM VIU)
# ============================================================

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ============================================================
# 1. CREAR CARPETA ÚNICA PARA PROPHET
# ============================================================

output_path = "results/prophet"
os.makedirs(output_path, exist_ok=True)

# ============================================================
# 2. CARGAR DATOS
# ============================================================

df = pd.read_csv('data/demanda_semanal.csv', sep=';')
df.columns = df.columns.str.strip()

df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
df = df.dropna()

# ============================================================
# 3. FORMATO PARA PROPHET
# ============================================================

df_prophet = pd.DataFrame({
    "ds": pd.date_range(start="2023-01-01", periods=len(df), freq="W"),
    "y": df["demanda"]
})

# ============================================================
# 4. DIVISIÓN TRAIN / TEST (OUT-OF-SAMPLE)
# ============================================================

train = df_prophet.iloc[:40]
test = df_prophet.iloc[40:]

# ============================================================
# 5. ENTRENAMIENTO
# ============================================================

model = Prophet()
model.fit(train)

# Horizonte 12 semanas (41–52)
future = model.make_future_dataframe(periods=12, freq='W')
forecast = model.predict(future)

forecast_test = forecast.iloc[40:52].copy()

# ============================================================
# 6. MÉTRICAS FUERA DE MUESTRA
# ============================================================

mae = mean_absolute_error(test['y'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
mape = np.mean(np.abs((test['y'] - forecast_test['yhat']) / test['y'])) * 100

# ============================================================
# 7. GUARDAR FORECAST COMPLETO 41–52
# ============================================================

df_out = pd.DataFrame({
    "Semana": range(41, 53),
    "Demanda_real": test['y'].values,
    "Prediccion_Prophet": forecast_test['yhat'].values,
    "Error_abs": np.abs(test['y'].values - forecast_test['yhat'].values)
})

df_out.to_csv(f"{output_path}/prophet_forecast_series.csv", index=False)

# ============================================================
# 8. GUARDAR MÉTRICAS
# ============================================================

df_metricas = pd.DataFrame({
    'Modelo': ['Prophet'],
    'MAE': [round(mae, 2)],
    'RMSE': [round(rmse, 2)],
    'MAPE (%)': [round(mape, 2)]
})

df_metricas.to_csv(f"{output_path}/prophet_metricas.csv", index=False)

# ============================================================
# 9. GUARDAR KPIs ADICIONALES (Resumen Ejecutivo)
# ============================================================

df_kpis = pd.DataFrame({
    "Indicador": ["MAE", "RMSE", "MAPE (%)"],
    "Valor": [round(mae, 2), round(rmse, 2), round(mape, 2)]
})

df_kpis.to_csv(f"{output_path}/kpis_prophet_41_52.csv", index=False)

# ============================================================
# 10. GRÁFICO REAL VS PREDICCIÓN 41–52
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(range(41, 53), test['y'], label='Demanda real', marker='o')
plt.plot(range(41, 53), forecast_test['yhat'], label='Prophet', marker='x')

plt.title("Prophet - Predicción Demanda Semanal (41–52)")
plt.xlabel("Semana")
plt.ylabel("Demanda")
plt.legend()
plt.grid(True)

plt.savefig(f"{output_path}/prophet_forecast.png", dpi=300)
plt.close()

# ============================================================
# 11. CONFIRMACIÓN FINAL
# ============================================================

print("\n=======================================")
print("     PROPHET OUT-OF-SAMPLE COMPLETADO")
print("=======================================")
print("📊 MAE:", round(mae, 2))
print("📊 RMSE:", round(rmse, 2))
print("📊 MAPE:", round(mape, 2), "%")
print("📁 Todo guardado en:", output_path)
print("=======================================\n")