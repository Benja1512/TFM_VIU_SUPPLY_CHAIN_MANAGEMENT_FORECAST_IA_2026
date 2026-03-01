import pandas as pd
import os

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

INVENTARIO_INICIAL = 100
REPOSICION_SEMANAL = 250
CAPACIDAD_MAXIMA = 300

SEMANA_INICIAL = 41
NUM_SEMANAS = 12

# ============================================================
# RUTAS
# ============================================================

BASE_DIR = os.path.dirname(__file__)

forecast_file = os.path.join(
    BASE_DIR,
    "../results/random_forest/random_forest_forecast_series.csv"
)

output_dir = os.path.join(
    BASE_DIR,
    "../results/random_forest"
)

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# FUNCIÓN DE SIMULACIÓN
# ============================================================

def simular_escenario(df_forecast, factor_demanda=1.0):

    resultados = []
    inventario = INVENTARIO_INICIAL

    for i, row in df_forecast.iterrows():

        semana = SEMANA_INICIAL + i
        demanda = float(row["prediccion_rf"]) * factor_demanda

        inv_inicial = inventario
        inventario += REPOSICION_SEMANAL

        if inventario >= demanda:
            inventario -= demanda
            stockout = 0
        else:
            stockout = demanda - inventario
            inventario = 0

        sobrestock = 0
        if inventario > CAPACIDAD_MAXIMA:
            sobrestock = inventario - CAPACIDAD_MAXIMA
            inventario = CAPACIDAD_MAXIMA

        resultados.append({
            "Semana": semana,
            "Demanda": round(demanda, 1),
            "Reposicion": REPOSICION_SEMANAL,
            "Inv_Inicial": round(inv_inicial, 1),
            "Inv_Final": round(inventario, 1),
            "Sobrestock": round(sobrestock, 1),
            "Stockout": round(stockout, 1)
        })

    return pd.DataFrame(resultados)


# ============================================================
# CARGAR FORECAST
# ============================================================

df_forecast = pd.read_csv(forecast_file)

if "prediccion_rf" not in df_forecast.columns:
    raise ValueError("❌ No se encontró la columna 'prediccion_rf'")

df_forecast = df_forecast.head(NUM_SEMANAS)

# ============================================================
# SIMULAR
# ============================================================

base = simular_escenario(df_forecast, 1.0)
alto = simular_escenario(df_forecast, 1.20)
bajo = simular_escenario(df_forecast, 0.80)

# ============================================================
# GUARDAR SIMULACIONES
# ============================================================

base.to_csv(os.path.join(output_dir, "simulacion_rf_base_41_52.csv"), index=False)
alto.to_csv(os.path.join(output_dir, "simulacion_rf_alto_41_52.csv"), index=False)
bajo.to_csv(os.path.join(output_dir, "simulacion_rf_bajo_41_52.csv"), index=False)

# ============================================================
# CALCULAR KPIs
# ============================================================

def calcular_kpis(df):
    inventario_medio = df["Inv_Final"].mean()
    sobrestock_total = df["Sobrestock"].sum()
    stockout_total = df["Stockout"].sum()
    return round(inventario_medio,1), round(sobrestock_total,1), round(stockout_total,1)

kpi_base = calcular_kpis(base)
kpi_alto = calcular_kpis(alto)
kpi_bajo = calcular_kpis(bajo)

df_kpis = pd.DataFrame({
    "Escenario": ["Base", "+20%", "-20%"],
    "Inventario_final_medio": [kpi_base[0], kpi_alto[0], kpi_bajo[0]],
    "Sobrestock_total": [kpi_base[1], kpi_alto[1], kpi_bajo[1]],
    "Stockout_total": [kpi_base[2], kpi_alto[2], kpi_bajo[2]]
})

# Guardar KPIs
df_kpis.to_csv(os.path.join(output_dir, "kpis_rf_41_52.csv"), index=False)

# ============================================================
# IMPRIMIR EN TERMINAL
# ============================================================

print("\n=======================================")
print("   SIMULACIÓN LOGÍSTICA RF OK")
print("=======================================")
print(df_kpis)
print("📁 Resultados guardados en: results/random_forest")