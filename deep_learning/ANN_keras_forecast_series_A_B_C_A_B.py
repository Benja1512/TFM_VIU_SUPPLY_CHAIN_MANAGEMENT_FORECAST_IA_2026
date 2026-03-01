# ============================================================
# SIMULACIÓN LOGÍSTICA POR ESCENARIOS (ANN - KERAS)
# Semanas 41–52 | Producto: Leche Entera 1L
# Autor: Elaboración propia (TFM VIU)
# ============================================================

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1) Cargar forecast ANN OUT-OF-SAMPLE (41–52)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

forecast_file = os.path.join(
    BASE_DIR,
    "../results/ann_keras/ann_keras_forecast_series.csv"
)

df_forecast = pd.read_csv(forecast_file)
df_forecast.columns = df_forecast.columns.str.strip()

weeks = df_forecast["Semana"].values
demanda_pred = np.round(
    df_forecast["Prediccion_ANN_Keras"].values
).astype(int)

print("✅ Forecast ANN cargado correctamente")
print("📌 Periodo:", weeks.min(), "-", weeks.max())

# ------------------------------------------------------------
# 2) Parámetros logísticos
# ------------------------------------------------------------

CAPACIDAD_MAXIMA = 300

# ------------------------------------------------------------
# 3) Políticas de reposición
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

        reposicion = calcular_reposicion(d, politica_reposicion)
        inventario += reposicion

        atendido = min(inventario, d)
        stockout = max(0, d - inventario)

        inventario -= atendido

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

    return pd.DataFrame(
        filas,
        columns=[
            "Semana",
            "Demanda_Forecast_ANN",
            "Reposicion",
            "Inv_Inicial",
            "Inv_Final",
            "Sobrestock",
            "Stockout"
        ]
    )

# ------------------------------------------------------------
# 5) Escenarios ORDENADOS
# ------------------------------------------------------------

escenarios = [
    ("01", "OPTIMO A", "Alto", 80, "Generosa", "Máxima seguridad, pero con alto costo"),
    ("02", "OPTIMO B", "Medio", 50, "Ajustada", "Equilibrado y eficiente"),
    ("03", "INTERMEDIO", "Medio-bajo", 30, "Moderada", "Riesgo controlado, pero con quiebres"),
    ("04", "CRITICO A", "Bajo", 10, "Ninguna", "Rotura total de stock"),
    ("05", "CRITICO B", "Sin stock", 0, "Insuficiente", "Peor escenario posible")
]

# ------------------------------------------------------------
# 6) Ejecutar simulación
# ------------------------------------------------------------

output_dir = os.path.join(BASE_DIR, "../results/ann_keras")
os.makedirs(output_dir, exist_ok=True)

resumen = []
detalles = []

for num, esc, tipo, stock, politica, comentario in escenarios:

    df_sim = simular_escenario(demanda_pred, weeks, stock, politica)

    # 🔥 Guardado numerado
    nombre_archivo = f"{num}_{esc.replace(' ', '_')}"
    df_sim.to_csv(
        os.path.join(output_dir, f"simulacion_ann_{nombre_archivo}_41_52.csv"),
        index=False
    )

    detalles.append(df_sim.assign(Escenario=esc))

    resumen.append({
        "Orden": num,
        "Escenario": esc,
        "Tipo de Stock": tipo,
        "Stock inicial (u)": stock,
        "Reposición promedio semanal (u)": int(round(df_sim["Reposicion"].mean(), 0)),
        "Stockout total (u)": int(round(df_sim["Stockout"].sum(), 0)),
        "Sobrestock total (u)": int(round(df_sim["Sobrestock"].sum(), 0)),
        "Comentario breve": comentario
    })

df_resumen = pd.DataFrame(resumen)
df_resumen = df_resumen.sort_values("Orden")

df_resumen.to_csv(
    os.path.join(output_dir, "tabla_escenarios_ann_41_52.csv"),
    index=False
)

df_detalle_all = pd.concat(detalles, ignore_index=True)
df_detalle_all.to_csv(
    os.path.join(output_dir, "detalle_escenarios_ann_41_52.csv"),
    index=False
)

print("\n=======================================")
print("   ESCENARIOS LOGÍSTICOS ANN ORDENADOS OK")
print("=======================================")
print(df_resumen.to_string(index=False))
print("📁 Guardado en:", output_dir)
print("=======================================\n")