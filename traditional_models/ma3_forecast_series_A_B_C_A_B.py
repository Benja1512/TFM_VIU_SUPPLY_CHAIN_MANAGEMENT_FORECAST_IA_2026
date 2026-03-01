# ============================================================
# SIMULACIÓN LOGÍSTICA POR ESCENARIOS (MA-3)
# Semanas 41–52 | Leche Entera 1L
# ============================================================

import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 1) CARGAR FORECAST MA3 (OUT-OF-SAMPLE 41–52)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

forecast_file = os.path.join(
    BASE_DIR,
    "../results/ma3/ma3_forecast_series.csv"
)

df_forecast = pd.read_csv(forecast_file)

# Normalizar nombres de columnas
df_forecast.columns = df_forecast.columns.str.strip().str.lower()

# Verificación básica
required_cols = ["semana", "pronostico_ma3"]
for col in required_cols:
    if col not in df_forecast.columns:
        raise ValueError(f"❌ No se encontró la columna '{col}' en el archivo.")

# Eliminar filas sin pronóstico (semanas 1 y 2)
df_forecast = df_forecast.dropna(subset=["pronostico_ma3"])

# Filtrar solo semanas 41–52
df_forecast = df_forecast[df_forecast["semana"] >= 41]

if df_forecast.empty:
    raise ValueError("❌ No se encontraron semanas 41–52.")

weeks = df_forecast["semana"].values
demanda_pred = np.round(
    df_forecast["pronostico_ma3"].values
).astype(int)

print("=======================================")
print("✅ MA-3 cargado correctamente")
print("📌 Periodo:", weeks.min(), "-", weeks.max())
print("📦 Total semanas:", len(weeks))
print("=======================================")


# ------------------------------------------------------------
# 2) PARÁMETROS
# ------------------------------------------------------------

CAPACIDAD_MAXIMA = 300


# ------------------------------------------------------------
# 3) POLÍTICAS DE REPOSICIÓN
# ------------------------------------------------------------

def calcular_reposicion(demanda, politica):

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

    return int(np.ceil(demanda * mult))


# ------------------------------------------------------------
# 4) SIMULACIÓN
# ------------------------------------------------------------

def simular_escenario(demanda, weeks, stock_inicial, politica):

    inventario = float(stock_inicial)
    filas = []

    for semana, d in zip(weeks, demanda):

        inv_inicial = inventario

        reposicion = calcular_reposicion(d, politica)
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
            int(sobrestock),
            int(stockout)
        ])

    return pd.DataFrame(
        filas,
        columns=[
            "Semana",
            "Demanda_Forecast_MA3",
            "Reposicion",
            "Inv_Inicial",
            "Inv_Final",
            "Sobrestock",
            "Stockout"
        ]
    )


# ------------------------------------------------------------
# 5) ESCENARIOS
# ------------------------------------------------------------

escenarios = [
    ("01_OPTIMO_A", "Alto", 80, "Generosa", "Máxima seguridad, alto costo"),
    ("02_OPTIMO_B", "Medio", 50, "Ajustada", "Equilibrio eficiencia-riesgo"),
    ("03_INTERMEDIO", "Medio-bajo", 30, "Moderada", "Riesgo controlado"),
    ("04_CRITICO_A", "Bajo", 10, "Ninguna", "Sin reposición → rotura total"),
    ("05_CRITICO_B", "Sin stock", 0, "Insuficiente", "Reposición insuficiente")
]


# ------------------------------------------------------------
# 6) EJECUCIÓN
# ------------------------------------------------------------

output_dir = os.path.join(BASE_DIR, "../results/ma3")
os.makedirs(output_dir, exist_ok=True)

resumen = []
detalles = []

for nombre, tipo, stock, politica, comentario in escenarios:

    df_sim = simular_escenario(demanda_pred, weeks, stock, politica)

    df_sim.to_csv(
        os.path.join(output_dir, f"simulacion_ma3_{nombre}_41_52.csv"),
        index=False
    )

    detalles.append(df_sim.assign(Escenario=nombre))

    resumen.append({
        "Escenario": nombre,
        "Tipo de Stock": tipo,
        "Stock inicial (u)": stock,
        "Reposición promedio semanal (u)": int(df_sim["Reposicion"].mean()),
        "Stockout total (u)": int(df_sim["Stockout"].sum()),
        "Sobrestock total (u)": int(df_sim["Sobrestock"].sum()),
        "Comentario breve": comentario
    })

df_resumen = pd.DataFrame(resumen)

df_resumen.to_csv(
    os.path.join(output_dir, "tabla_escenarios_ma3_41_52.csv"),
    index=False
)

df_detalle = pd.concat(detalles, ignore_index=True)
df_detalle.to_csv(
    os.path.join(output_dir, "detalle_escenarios_ma3_41_52.csv"),
    index=False
)

print("\n=======================================")
print(" ESCENARIOS LOGÍSTICOS MA-3 OK")
print("=======================================")
print(df_resumen.to_string(index=False))
print("📁 Guardado en:", output_dir)
print("=======================================\n")