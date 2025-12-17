import pandas as pd
import os
import matplotlib.pyplot as plt

# === 1. Función de simulación logística por escenario ===

def simular_logistica(demanda, escenario, recepcion=250, inventario_inicial=100, capacidad_maxima=300):
    inventario = inventario_inicial
    resultados = []

    for semana, d in enumerate(demanda, start=1):
        disponible = inventario + recepcion
        stockout = max(0, d - disponible)
        inventario_final = max(0, disponible - d)
        sobrestock = max(0, inventario_final - capacidad_maxima)

        resultados.append({
            "Semana": semana,
            "Demanda pronosticada": round(d, 1),
            "Inventario inicial": round(inventario, 1),
            "Recepción": recepcion,
            "Inventario final": round(inventario_final, 1),
            "Stockout": round(stockout, 1),
            "Sobrestock": round(sobrestock, 1),
        })

        inventario = inventario_final

    return pd.DataFrame(resultados)

# === 2. Leer archivo de escenarios generado por Prophet ===

ruta_csv = "results/escenarios/escenarios_prophet.csv"

if not os.path.exists(ruta_csv):
    raise FileNotFoundError(f"❌ ERROR: No se encuentra el archivo de escenarios: {ruta_csv}")

df = pd.read_csv(ruta_csv)

# Asegurarse de renombrar si los nombres no coinciden
df = df.rename(columns={
    "base_prophet": "base",
    "escenario_alto_20": "alto",
    "escenario_bajo_20": "bajo",
    "escenario_pesimista": "pesimista",
    "escenario_optimista": "optimista"
})

df = df.tail(12).reset_index(drop=True)

# === 3. Simulación por escenario ===

escenarios = ["base", "alto", "bajo", "pesimista", "optimista"]
resumen = []

for esc in escenarios:
    print(f"\n📦 Escenario logístico: {esc.upper()}")
    demanda = df[esc]
    tabla = simular_logistica(demanda, esc)

    print(tabla.to_string(index=False))

    resumen.append({
        "Escenario": esc,
        "Inventario final medio (u)": round(tabla["Inventario final"].mean(), 1),
        "Sobrestock total (u)": round(tabla["Sobrestock"].sum(), 1),
        "Stockout total (u)": round(tabla["Stockout"].sum(), 1)
    })

    # === 3b. Guardar gráfico por escenario ===
    plt.figure(figsize=(10, 6))
    plt.plot(tabla["Semana"], tabla["Inventario final"], label="Inventario final", marker="o")
    plt.plot(tabla["Semana"], tabla["Demanda pronosticada"], label="Demanda pronosticada", linestyle="--", marker="x")
    plt.fill_between(tabla["Semana"], 0, tabla["Stockout"], label="Stockout", color="red", alpha=0.3)
    plt.fill_between(tabla["Semana"], 0, tabla["Sobrestock"], label="Sobrestock", color="green", alpha=0.2)
    plt.title(f"Simulación logística – Escenario: {esc.upper()}")
    plt.xlabel("Semana")
    plt.ylabel("Unidades")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("results/img", exist_ok=True)
    plt.savefig(f"results/img/escenario_{esc}.png")
    plt.close()

# === 4. Mostrar resumen final ===

df_resumen = pd.DataFrame(resumen)
print("\n📊 Indicadores logísticos comparativos:\n")
print(df_resumen.to_string(index=False))

# === 5. Guardar resumen CSV ===

os.makedirs("results/politicas", exist_ok=True)
df_resumen.to_csv("results/politicas/politicas_prophet.csv", index=False)
print("\n📁 CSV guardado en: results/politicas/politicas_prophet.csv")
