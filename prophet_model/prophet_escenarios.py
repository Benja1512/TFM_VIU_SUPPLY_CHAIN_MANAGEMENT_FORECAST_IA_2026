import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# =====================================================
# 1. CARGAR DATOS SIMULADOS
# =====================================================

csv_input = "data/demanda_semanal.csv"   # <-- RUTA CORRECTA

if not os.path.exists(csv_input):
    raise FileNotFoundError(f"❌ ERROR: No existe el archivo de demanda en: {csv_input}")

# ⚠️ Cargar con sep=';' para leer correctamente el archivo
df = pd.read_csv(csv_input, sep=';')

# Renombrar columnas para Prophet
df = df.rename(columns={"semana": "ds", "demanda": "y"})

# Convertir 'ds' a fechas semanales
df["ds"] = pd.date_range(start='2023-01-01', periods=len(df), freq='W')

# =====================================================
# 2. ENTRENAR MODELO PROPHET
# =====================================================

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12, freq="W")
forecast = model.predict(future)

# =====================================================
# 3. CREAR ESCENARIOS DE DEMANDA
# =====================================================

escenarios = pd.DataFrame({
    "ds": forecast["ds"],
    "base_prophet": forecast["yhat"],
    "escenario_alto_20": forecast["yhat"] * 1.20,
    "escenario_bajo_20": forecast["yhat"] * 0.80,
    "escenario_pesimista": forecast["yhat_upper"] * 1.10,
    "escenario_optimista": forecast["yhat_lower"]
})

# =====================================================
# 4. MOSTRAR RESULTADOS EN TERMINAL
# =====================================================

print("\n=== ESCENARIOS PROPHET (ÚLTIMAS 12 SEMANAS) ===\n")
print(escenarios.tail(12))

# =====================================================
# 5. GUARDAR CSV
# =====================================================

os.makedirs("results/escenarios", exist_ok=True)
csv_output = "results/escenarios/escenarios_prophet.csv"
escenarios.to_csv(csv_output, index=False)
print(f"\n📁 CSV guardado en: {csv_output}")

# =====================================================
# 6. GUARDAR GRÁFICO
# =====================================================

os.makedirs("results/img", exist_ok=True)

plt.figure(figsize=(12, 6))
plt.plot(escenarios["ds"], escenarios["base_prophet"], label="Base Prophet")
plt.plot(escenarios["ds"], escenarios["escenario_alto_20"], label="Demanda +20%")
plt.plot(escenarios["ds"], escenarios["escenario_bajo_20"], label="Demanda -20%")
plt.plot(escenarios["ds"], escenarios["escenario_pesimista"], label="Pesimista (upper +10%)")
plt.plot(escenarios["ds"], escenarios["escenario_optimista"], label="Optimista (lower)")

plt.legend()
plt.title("Escenarios de Demanda basados en Prophet – Leche Entera 1L")
plt.xlabel("Semana")
plt.ylabel("Unidades")
plt.grid(True)

img_output = "results/img/escenarios_prophet.png"
plt.savefig(img_output, dpi=300)
print(f"🖼️ Gráfico guardado en: {img_output}")
plt.close()

print("\n✅ Proceso completado correctamente.\n")
esc_path = "results/escenarios/escenarios_prophet.csv"

