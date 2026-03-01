# ============================================================
# SIMULACIÓN LOGÍSTICA POR ESCENARIOS (RANDOM FOREST)
# Semanas 41–52 | Producto: Leche Entera 1L
# Autor: Elaboración propia (TFM VIU)
# ============================================================

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1) Cargar forecast Random Forest OUT-OF-SAMPLE (41–52)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

forecast_file = os.path.join(
    BASE_DIR,
    "../results/random_forest/random_forest_forecast_series.csv"
)

df_forecast = pd.read_csv(forecast_file)

required_cols = {"semana", "prediccion_rf"}
missing = required_cols - set(df_forecast.columns)
if missing:
    raise ValueError(f"Faltan columnas en random_forest_forecast_series.csv: {missing}")

weeks = df_forecast["semana"].values
demanda_pred = np.round(df_forecast["prediccion_rf"].values).astype(int)

print("✅ Forecast Random Forest cargado correctamente")
print("📌 Periodo:", weeks.min(), "-", weeks.max())

# ------------------------------------------------------------
# 2) Parámetros logísticos
# ------------------------------------------------------------

CAPACIDAD_MAXIMA = 300

# ------------------------------------------------------------
# 3) Políticas de reposición (idénticas a Prophet)
# ------------------------------------------------------------

def calcular_reposicion(demanda_semana, politica):

    politica = politica.lower()

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
        raise ValueError("Política no reconocida")

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
        stockout = max(0, d - inventario)

        inventario -= atendido

        # Sobrestock
        sobrestock = max(0, inventario - CAPACIDAD_MAXIMA)

        if inventario > CAPACIDAD_MAXIMA:
            inventario = CAPACIDAD_MAXIMA

        filas.append([
            int(semana),
            int(d),
            int(reposicion),
            round(inv_inicial, 2),
            round(inventario, 2),
            int(round(sobrestock, 0)),
            int(round(stockout, 0))
        ])

    df_resultado = pd.DataFrame(
        filas,
        columns=[
            "Semana",
            "Demanda_Forecast_RF",
            "Reposicion",
            "Inv_Inicial",
            "Inv_Final",
            "Sobrestock",
            "Stockout"
        ]
    )

    return df_resultado

# ------------------------------------------------------------
# 5) Escenarios (mismo orden que Prophet)
# ------------------------------------------------------------

escenarios = [
    ("OPTIMO A", "Alto", 80, "Generosa", "Máxima seguridad, pero con alto costo"),
    ("OPTIMO B", "Medio", 50, "Ajustada", "Equilibrado y eficiente"),
    ("INTERMEDIO", "Medio-bajo", 30, "Moderada", "Riesgo controlado, pero con quiebres"),
    ("CRITICO A", "Bajo", 10, "Ninguna", "Rotura total de stock"),
    ("CRITICO B", "Sin stock", 0, "Insuficiente", "Peor escenario posible")
]

# ------------------------------------------------------------
# 6) Ejecutar simulación
# ------------------------------------------------------------

output_dir = os.path.join(BASE_DIR, "../results/random_forest")
os.makedirs(output_dir, exist_ok=True)

resumen = []
detalles = []

for esc, tipo, stock, politica, comentario in escenarios:

    df_sim = simular_escenario(demanda_pred, weeks, stock, politica)

    # Guardar detalle individual
    df_sim.to_csv(
        os.path.join(output_dir, f"simulacion_rf_{esc}_41_52.csv"),
        index=False
    )

    detalles.append(df_sim.assign(Escenario=esc))

    resumen.append({
        "Escenario": esc,
        "Tipo de Stock": tipo,
        "Stock inicial (u)": stock,
        "Reposición promedio semanal (u)": int(round(df_sim["Reposicion"].mean(), 0)),
        "Stockout total (u)": int(round(df_sim["Stockout"].sum(), 0)),
        "Sobrestock total (u)": int(round(df_sim["Sobrestock"].sum(), 0)),
        "Comentario breve": comentario
    })

df_resumen = pd.DataFrame(resumen)

orden = ["OPTIMO A", "OPTIMO B", "INTERMEDIO", "CRITICO A", "CRITICO B"]
df_resumen["Escenario"] = pd.Categorical(df_resumen["Escenario"], categories=orden, ordered=True)
df_resumen = df_resumen.sort_values("Escenario")

df_resumen.to_csv(
    os.path.join(output_dir, "tabla_escenarios_rf_41_52.csv"),
    index=False
)

df_detalle_all = pd.concat(detalles, ignore_index=True)
df_detalle_all.to_csv(
    os.path.join(output_dir, "detalle_escenarios_rf_41_52.csv"),
    index=False
)

# ------------------------------------------------------------
# 7) Confirmación
# ------------------------------------------------------------

print("\n=======================================")
print("   ESCENARIOS LOGÍSTICOS RANDOM FOREST OK")
print("=======================================")
print(df_resumen.to_string(index=False))
print("📁 Guardado en:", output_dir)
print("=======================================\n")