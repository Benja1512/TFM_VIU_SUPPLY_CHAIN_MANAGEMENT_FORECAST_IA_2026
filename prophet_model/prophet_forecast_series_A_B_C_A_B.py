# ============================================================
# SIMULACIÓN LOGÍSTICA POR ESCENARIOS (PROPHET) - Semanas 41–52
# Producto: Leche Entera 1L - Utrecht (Retail urbano simulado)
# Autor: Elaboración propia (TFM VIU)
# ============================================================

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1) Cargar pronóstico Prophet OUT-OF-SAMPLE (41–52)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

forecast_file = os.path.join(
    BASE_DIR,
    "../results/prophet/prophet_forecast_series.csv"
)

df_forecast = pd.read_csv(forecast_file)

required_cols = {"Semana", "Prediccion_Prophet"}
missing = required_cols - set(df_forecast.columns)
if missing:
    raise ValueError(f"Faltan columnas en prophet_forecast_series.csv: {missing}")

weeks = df_forecast["Semana"].values
demanda_pred = np.round(
    df_forecast["Prediccion_Prophet"].values
).astype(int)  # 🔥 ahora trabajamos con unidades enteras

print("✅ Forecast Prophet cargado correctamente")
print("📌 Periodo:", weeks.min(), "-", weeks.max())

# ------------------------------------------------------------
# 2) Parámetros logísticos
# ------------------------------------------------------------

CAPACIDAD_MAXIMA = 300

# ------------------------------------------------------------
# 3) Políticas de reposición
# ------------------------------------------------------------

def calcular_reposicion(demanda_semana: float, politica: str) -> int:

    politica = politica.strip().lower()

    if politica == "generosa":
        mult = 1.20
    elif politica == "ajustada":
        mult = 1.00
    elif politica == "moderada":
        mult = 0.80
    elif politica == "ninguna":
        return 0
    elif politica == "insuficiente":
        mult = 0.30
    else:
        raise ValueError(f"Política no reconocida: {politica}")

    return int(np.ceil(demanda_semana * mult))

# ------------------------------------------------------------
# 4) Simulación de inventario
# ------------------------------------------------------------

def simular_escenario(demanda, weeks, inventario_inicial, politica_reposicion):

    inventario = float(inventario_inicial)
    filas = []

    for semana, d in zip(weeks, demanda):

        inv_inicial = inventario

        # Reposición dinámica
        reposicion = calcular_reposicion(d, politica_reposicion)
        inventario += reposicion

        # Atender demanda
        atendido = min(inventario, d)
        stockout = max(0.0, d - inventario)

        inventario -= atendido

        # Sobrestock
        sobrestock = max(0.0, inventario - CAPACIDAD_MAXIMA)

        if inventario > CAPACIDAD_MAXIMA:
            inventario = CAPACIDAD_MAXIMA

        filas.append([
            int(semana),
            int(d),
            int(reposicion),
            round(inv_inicial, 2),
            round(inventario, 2),
            round(sobrestock, 2),
            round(stockout, 2)
        ])

    df_resultado = pd.DataFrame(
        filas,
        columns=[
            "Semana",
            "Demanda_Forecast_Prophet",
            "Reposicion",
            "Inv_Inicial",
            "Inv_Final",
            "Sobrestock",
            "Stockout"
        ]
    )

    return df_resultado


def calcular_kpis(df):
    inventario_final_medio = df["Inv_Final"].mean()
    sobrestock_total = df["Sobrestock"].sum()
    stockout_total = df["Stockout"].sum()
    reposicion_promedio = df["Reposicion"].mean()

    return (
        round(inventario_final_medio, 2),
        round(sobrestock_total, 0),
        round(stockout_total, 0),
        round(reposicion_promedio, 0)
    )

# ------------------------------------------------------------
# 5) Definir escenarios
# ------------------------------------------------------------

escenarios = [
    {"Escenario": "OPTIMO A", "Tipo de Stock": "Alto", "Stock inicial": 80,
     "Política": "Generosa", "Comentario breve": "Máxima seguridad, pero con alto costo"},

    {"Escenario": "OPTIMO B", "Tipo de Stock": "Medio", "Stock inicial": 50,
     "Política": "Ajustada", "Comentario breve": "Equilibrado y eficiente"},

    {"Escenario": "INTERMEDIO", "Tipo de Stock": "Medio-bajo", "Stock inicial": 30,
     "Política": "Moderada", "Comentario breve": "Riesgo controlado, pero con quiebres"},

    {"Escenario": "CRITICO A", "Tipo de Stock": "Bajo", "Stock inicial": 10,
     "Política": "Ninguna", "Comentario breve": "Rotura total de stock"},

    {"Escenario": "CRITICO B", "Tipo de Stock": "Sin stock", "Stock inicial": 0,
     "Política": "Insuficiente", "Comentario breve": "Peor escenario posible"}
]

# ------------------------------------------------------------
# 6) Ejecutar simulación
# ------------------------------------------------------------

output_dir = os.path.join(BASE_DIR, "../results/prophet")
os.makedirs(output_dir, exist_ok=True)

detalles = []
resumen = []

for sc in escenarios:

    df_sim = simular_escenario(
        demanda=demanda_pred,
        weeks=weeks,
        inventario_inicial=sc["Stock inicial"],
        politica_reposicion=sc["Política"]
    )

    inv_medio, sobre_total, stockout_total, repo_prom = calcular_kpis(df_sim)

    fname = f"simulacion_prophet_{sc['Escenario'].replace(' ', '_')}_41_52.csv"
    df_sim.to_csv(os.path.join(output_dir, fname), index=False)

    detalles.append(df_sim.assign(Escenario=sc["Escenario"]))

    resumen.append({
        "Escenario": sc["Escenario"],
        "Tipo de Stock": sc["Tipo de Stock"],
        "Stock inicial (u)": sc["Stock inicial"],
        "Reposición promedio semanal (u)": repo_prom,
        "Stockout total (u)": stockout_total,
        "Sobrestock total (u)": sobre_total,
        "Comentario breve": sc["Comentario breve"]
    })

df_resumen = pd.DataFrame(resumen)

orden = ["OPTIMO A", "OPTIMO B", "INTERMEDIO", "CRITICO A", "CRITICO B"]
df_resumen["Escenario"] = pd.Categorical(df_resumen["Escenario"], categories=orden, ordered=True)
df_resumen = df_resumen.sort_values("Escenario")

df_resumen.to_csv(os.path.join(output_dir, "tabla_escenarios_prophet_41_52.csv"), index=False)

df_detalle_all = pd.concat(detalles, ignore_index=True)
df_detalle_all.to_csv(os.path.join(output_dir, "detalle_escenarios_prophet_41_52.csv"), index=False)

# ------------------------------------------------------------
# 7) Confirmación
# ------------------------------------------------------------

print("\n=======================================")
print("   ESCENARIOS LOGÍSTICOS PROPHET OK")
print("=======================================")
print(df_resumen.to_string(index=False))
print("\n📁 Archivos generados en:", output_dir)
print("=======================================\n")