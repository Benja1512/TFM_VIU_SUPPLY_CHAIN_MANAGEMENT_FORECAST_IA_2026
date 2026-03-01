import pandas as pd
import os

# ============================================================
# SIMULACIÓN LOGÍSTICA OPERATIVA – ANN_Keras (Base-Stock)
# Escenarios: Base / +20% / -20%
# Horizonte: Semanas 41–52
# Política: Nivel objetivo = Forecast + Stock Seguridad
# ============================================================

def simular_escenario(df,
                      factor_demanda=1.0,
                      inventario_inicial=250,
                      stock_seguridad=50):

    inventario = inventario_inicial
    resultados = []

    sobrestock_total = 0
    stockout_total = 0

    for _, row in df.iterrows():

        semana = int(row["Semana"])
        demanda_real = float(row["Demanda_real"])
        demanda = demanda_real * factor_demanda
        forecast = float(row["Prediccion_ANN_Keras"])

        # ---------------------------------------------------
        # Política Base-Stock REAL
        # ---------------------------------------------------
        nivel_objetivo = forecast + stock_seguridad
        reposicion = max(0, nivel_objetivo - inventario)

        inv_inicial = inventario
        inventario += reposicion

        # Ventas
        ventas = min(inventario, demanda)
        stockout = max(0, demanda - inventario)

        inventario -= ventas

        # Sobrestock = inventario excedente
        sobrestock = max(0, inventario)

        sobrestock_total += sobrestock
        stockout_total += stockout

        resultados.append({
            "Semana": semana,
            "Demanda": round(demanda, 1),
            "Reposicion": round(reposicion, 1),
            "Inv_Inicial": round(inv_inicial, 1),
            "Inv_Final": round(inventario, 1),
            "Sobrestock": round(sobrestock, 1),
            "Stockout": round(stockout, 1)
        })

    df_resultados = pd.DataFrame(resultados)

    resumen = {
        "Inventario_final_medio": df_resultados["Inv_Final"].mean(),
        "Sobrestock_total": sobrestock_total,
        "Stockout_total": stockout_total
    }

    return df_resultados, resumen


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n=======================================")
    print("🚀 Simulación logística ANN_Keras (Base-Stock)")
    print("=======================================\n")

    BASE_DIR = os.path.dirname(__file__)

    forecast_file = os.path.join(
        BASE_DIR,
        "../results/ann_keras/ann_keras_forecast_series.csv"
    )

    output_dir = os.path.join(
        BASE_DIR,
        "../results/ann_keras"
    )

    os.makedirs(output_dir, exist_ok=True)

    df_forecast = pd.read_csv(forecast_file)

    print("✅ Archivo ANN_Keras cargado correctamente.")
    print("📌 Columnas:", df_forecast.columns)

    escenarios = {
        "Base": 1.0,
        "+20%": 1.20,
        "-20%": 0.80
    }

    resumenes = []

    for nombre, factor in escenarios.items():

        print(f"\n📌 Ejecutando escenario: {nombre}")

        df_sim, resumen = simular_escenario(
            df_forecast,
            factor_demanda=factor,
            inventario_inicial=250,
            stock_seguridad=50
        )

        output_csv = os.path.join(
            output_dir,
            f"simulacion_ann_keras_{nombre}_41_52.csv"
        )

        df_sim.to_csv(output_csv, index=False)

        resumenes.append({
            "Escenario": nombre,
            "Inventario_final_medio": round(resumen["Inventario_final_medio"], 2),
            "Sobrestock_total": round(resumen["Sobrestock_total"], 2),
            "Stockout_total": round(resumen["Stockout_total"], 2)
        })

    # KPI FINAL
    df_kpis = pd.DataFrame(resumenes)

    kpi_file = os.path.join(
        output_dir,
        "kpis_ann_keras_41_52.csv"
    )

    df_kpis.to_csv(kpi_file, index=False)

    print("\n=======================================")
    print("   SIMULACIÓN LOGÍSTICA ANN_KERAS OK")
    print("=======================================")
    print(df_kpis.to_string(index=False))
    print("📁 Resultados guardados en: results/ann_keras\n")