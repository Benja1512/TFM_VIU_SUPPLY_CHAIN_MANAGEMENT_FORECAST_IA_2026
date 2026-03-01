# ============================================================
# Simulación logística Prophet (Semanas 41–52)
# Producto: Leche Entera 1L - Utrecht (Retail urbano simulado)
# Autor: Elaboración propia (TFM VIU)
# ============================================================

import pandas as pd
import os

# ------------------------------------------------------------
# 1. Cargar pronóstico Prophet OUT-OF-SAMPLE (41–52)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

forecast_file = os.path.join(
    BASE_DIR,
    "../results/prophet/prophet_forecast_series.csv"
)

df_forecast = pd.read_csv(forecast_file)

print("📂 Columnas detectadas:", df_forecast.columns)

# 🔥 Usar la columna correcta
demanda_pred = df_forecast["Prediccion_Prophet"].values
weeks = df_forecast["Semana"].values


# ------------------------------------------------------------
# 2. Parámetros logísticos
# ------------------------------------------------------------

INVENTARIO_INICIAL = 100
REPOSICION_SEMANAL = 250
CAPACIDAD_MAXIMA = 300


# ------------------------------------------------------------
# 3. Función de simulación
# ------------------------------------------------------------

def simular_escenario(demanda, weeks):

    inventario = INVENTARIO_INICIAL
    filas = []

    for semana, d in zip(weeks, demanda):

        inv_inicial = inventario

        # Reposición
        inventario += REPOSICION_SEMANAL

        # Atender demanda
        atendido = min(inventario, d)
        stockout = max(0, d - inventario)

        inventario -= atendido

        # Sobrestock
        sobrestock = max(0, inventario - CAPACIDAD_MAXIMA)

        if inventario > CAPACIDAD_MAXIMA:
            inventario = CAPACIDAD_MAXIMA

        filas.append([
            semana,
            round(d, 1),
            REPOSICION_SEMANAL,
            round(inv_inicial, 1),
            round(inventario, 1),
            round(sobrestock, 1),
            round(stockout, 1)
        ])

    df_resultado = pd.DataFrame(
        filas,
        columns=[
            "Semana",
            "Demanda",
            "Reposicion",
            "Inv_Inicial",
            "Inv_Final",
            "Sobrestock",
            "Stockout"
        ]
    )

    return df_resultado


# ------------------------------------------------------------
# 4. Escenarios ±20%
# ------------------------------------------------------------

base = simular_escenario(demanda_pred, weeks)
alto = simular_escenario(demanda_pred * 1.20, weeks)
bajo = simular_escenario(demanda_pred * 0.80, weeks)


# ------------------------------------------------------------
# 5. Guardar resultados
# ------------------------------------------------------------

output_dir = os.path.join(BASE_DIR, "../results/prophet")
os.makedirs(output_dir, exist_ok=True)

base.to_csv(os.path.join(output_dir, "simulacion_prophet_base_41_52.csv"), index=False)
alto.to_csv(os.path.join(output_dir, "simulacion_prophet_alto_41_52.csv"), index=False)
bajo.to_csv(os.path.join(output_dir, "simulacion_prophet_bajo_41_52.csv"), index=False)


# ------------------------------------------------------------
# 6. Calcular KPIs agregados
# ------------------------------------------------------------

def calcular_kpis(df):

    inventario_medio = df["Inv_Final"].mean()
    sobrestock_total = df["Sobrestock"].sum()
    stockout_total = df["Stockout"].sum()

    return round(inventario_medio, 1), round(sobrestock_total, 1), round(stockout_total, 1)


kpi_base = calcular_kpis(base)
kpi_alto = calcular_kpis(alto)
kpi_bajo = calcular_kpis(bajo)

df_kpis = pd.DataFrame({
    "Escenario": ["Base", "+20%", "-20%"],
    "Inventario_final_medio": [kpi_base[0], kpi_alto[0], kpi_bajo[0]],
    "Sobrestock_total": [kpi_base[1], kpi_alto[1], kpi_bajo[1]],
    "Stockout_total": [kpi_base[2], kpi_alto[2], kpi_bajo[2]]
})

df_kpis.to_csv(os.path.join(output_dir, "kpis_prophet_41_52.csv"), index=False)


print("\n=======================================")
print("   SIMULACIÓN LOGÍSTICA PROPHET OK")
print("=======================================")
print(df_kpis)
print("📁 Resultados guardados en: results/prophet")