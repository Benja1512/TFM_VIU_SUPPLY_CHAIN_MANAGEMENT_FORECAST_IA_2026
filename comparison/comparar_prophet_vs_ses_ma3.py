import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Cargar demanda original
df = pd.read_csv('data/demanda_semanal.csv', sep=';')
df.columns = df.columns.str.strip()
df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
df = df.dropna()

# 2. Cargar métricas
df_ses = pd.read_csv('results/metricas/ses_metricas.csv')
df_ma3 = pd.read_csv('results/metricas/ma3_metricas.csv')
df_prophet = pd.read_csv('results/metricas/prophet_metricas.csv')

# 3. Concatenar métricas
df_comparacion = pd.concat([df_ses, df_ma3, df_prophet])
df_comparacion.to_csv('results/metricas/comparacion_metricas.csv', index=False)
print("✅ CSV comparacion_metricas.csv guardado en results/metricas/")
print("\n📊 Comparación completa de métricas:\n")
print(df_comparacion)

# 4. Cargar series completas
df_ses_full = pd.read_csv('results/metricas/ses_forecast_series.csv')
df_prophet_full = pd.read_csv('results/metricas/prophet_forecast_series.csv')
df_prophet_full = df_prophet_full.rename(columns={'ds': 'semana', 'yhat': 'pronostico_prophet'})

df_ma3_full = df.copy()
df_ma3_full['pronostico_ma3'] = df_ma3_full['demanda'].rolling(window=3).mean()

# 5. Gráfico comparativo
plt.figure(figsize=(12, 6))
plt.plot(df['semana'], df['demanda'], label='Demanda real', marker='o', alpha=0.6)
plt.plot(df_ses_full['semana'], df_ses_full['pronostico_ses'], label='SES', linestyle='--', marker='x')
plt.plot(df_ma3_full['semana'], df_ma3_full['pronostico_ma3'], label='MA-3', linestyle='-.', marker='^')
plt.plot(df_prophet_full['semana'], df_prophet_full['pronostico_prophet'], label='Prophet', linestyle=':', marker='s')

plt.title('Comparación de modelos: SES vs MA-3 vs Prophet')
plt.xlabel('Semana')
plt.ylabel('Demanda')
plt.grid(True)
plt.legend()

# Mejora la visibilidad del eje X
plt.xticks(
    ticks=df['semana'][::4],  # una etiqueta cada 4 semanas
    rotation=45,
    ha='right'
)

plt.tight_layout()


# 6. Guardar gráfico
os.makedirs('results/graficos', exist_ok=True)
plt.savefig('results/graficos/comparacion_modelos_3.png')
print("✅ Gráfico comparacion_modelos_3.png guardado en results/graficos/")
plt.show()
