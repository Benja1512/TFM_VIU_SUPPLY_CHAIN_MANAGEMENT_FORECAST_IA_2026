import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Cargar demanda original
df = pd.read_csv('data/demanda_semanal.csv', sep=';')
df.columns = df.columns.str.strip()
df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
df = df.dropna()

# 2. Cargar métricas SES y MA-3
df_ses = pd.read_csv('results/metricas/ses_metricas.csv')
df_ma3 = pd.read_csv('results/metricas/ma3_metricas.csv')

# 3. Cargar predicciones completas si existen
df_ses_full = pd.read_csv('results/metricas/ses_forecast_series.csv') if os.path.exists('results/metricas/ses_forecast_series.csv') else None
df_ma3_full = pd.read_csv('results/metricas/ma3_forecast_series.csv') if os.path.exists('results/metricas/ma3_forecast_series.csv') else None

# 4. Gráfico comparativo
plt.figure(figsize=(12, 6))
plt.plot(df['semana'], df['demanda'], label='Demanda real', marker='o', alpha=0.6)

if df_ses_full is not None and 'pronostico_ses' in df_ses_full.columns:
    plt.plot(df_ses_full['semana'], df_ses_full['pronostico_ses'], label='SES', linestyle='--', marker='x')

if df_ma3_full is not None and 'pronostico_ma3' in df_ma3_full.columns:
    plt.plot(df_ma3_full['semana'], df_ma3_full['pronostico_ma3'], label='MA-3', linestyle='-.', marker='^')

plt.title('Comparación: SES vs MA-3')
plt.xlabel('Semana')
plt.ylabel('Demanda')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 5. Guardar gráfico antes de mostrar
os.makedirs('results/graficos', exist_ok=True)
plt.savefig('results/graficos/comparacion_ses_vs_ma3.png')

# 6. Guardar métricas combinadas antes de mostrar el gráfico
df_comparacion = pd.concat([df_ses, df_ma3])
os.makedirs('results/metricas', exist_ok=True)  # por si acaso no existe
df_comparacion.to_csv('results/metricas/comparacion_metricas.csv', index=False)

# 🔁 Mensaje de confirmación
print("✅ CSV comparacion_metricas.csv guardado en results/metricas/")
print("\n📊 Comparación de métricas:")
print(df_comparacion)

# 7. Mostrar gráfico
plt.show()
