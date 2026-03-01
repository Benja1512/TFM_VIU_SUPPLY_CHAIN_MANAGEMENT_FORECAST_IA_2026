import pandas as pd
import numpy as np
import os

# =====================================================
# 1️⃣ CARGAR DATOS
# =====================================================
df = pd.read_csv("data/demanda_semanal.csv", sep=";")
df.columns = df.columns.str.strip().str.lower()
df = df.sort_values("semana").reset_index(drop=True)

# =====================================================
# 2️⃣ SPLIT OUT-OF-SAMPLE
# =====================================================
train = df[df["semana"] <= 40].copy()
test  = df[df["semana"] >= 41].copy()

# =====================================================
# 3️⃣ MA3 ROLLING
# =====================================================
def ma3_out_of_sample(train_data, test_data):
    history = list(train_data["demanda"])
    forecasts = []

    for real_value in test_data["demanda"]:
        forecast = np.mean(history[-3:])
        forecasts.append(forecast)
        history.append(real_value)

    return forecasts

test["Base"] = ma3_out_of_sample(train, test)
test["-20"]  = (test["Base"] - 20).clip(lower=0)
test["+20"]  = test["Base"] + 20

# =====================================================
# 4️⃣ SIMULACIÓN INVENTARIO
# =====================================================
def simular_escenario(data, forecast_col, stock_inicial=250):
    stock = stock_inicial
    resultados = []
    stockout_total = 0
    sobrestock_total = 0
    demanda_total = data["demanda"].sum()

    for _, row in data.iterrows():
        semana = row["semana"]
        demanda = row["demanda"]
        reposicion = row[forecast_col]
        inv_inicial = stock

        stock = stock + reposicion - demanda

        if stock < 0:
            stockout = abs(stock)
            sobrestock = 0
            stockout_total += stockout
            stock = 0
        else:
            stockout = 0
            sobrestock = stock
            sobrestock_total += sobrestock

        inv_final = stock

        resultados.append([
            semana,
            demanda,
            reposicion,
            inv_inicial,
            inv_final,
            sobrestock,
            stockout
        ])

    nivel_servicio = 100 * (1 - stockout_total / demanda_total)

    tabla = pd.DataFrame(
        resultados,
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

    return tabla, stockout_total, sobrestock_total, nivel_servicio


# Generar escenarios
tabla_base, so_b, ss_b, ns_b = simular_escenario(test, "Base")
tabla_m20, so_m, ss_m, ns_m = simular_escenario(test, "-20")
tabla_p20, so_p, ss_p, ns_p = simular_escenario(test, "+20")

# =====================================================
# 5️⃣ GUARDAR TABLAS DETALLADAS
# =====================================================
output_path = "results/ma3"
os.makedirs(output_path, exist_ok=True)

tabla_base.to_csv(f"{output_path}/simulacion_ma3_base_41_52.csv", index=False)
tabla_m20.to_csv(f"{output_path}/simulacion_ma3_-20_41_52.csv", index=False)
tabla_p20.to_csv(f"{output_path}/simulacion_ma3_+20_41_52.csv", index=False)

# =====================================================
# 6️⃣ TABLA KPI RESUMIDA
# =====================================================
tabla_kpi = pd.DataFrame({
    "Escenario": ["Base", "-20", "+20"],
    "Stockout_total": [so_b, so_m, so_p],
    "Sobrestock_total": [ss_b, ss_m, ss_p],
    "Nivel_servicio_%": [ns_b, ns_m, ns_p]
})

tabla_kpi.to_csv(f"{output_path}/tabla_kpi_ma3_41_52.csv", index=False)

print("\n📊 KPI MA3 (41–52)")
print(tabla_kpi)

print("\n✅ Todo guardado en results/ma3/")