# ============================================================
# RANDOM FOREST FORECAST (OUT-OF-SAMPLE 41–52)
# Con variables rezagadas (lags)
# Producto: Leche Entera 1L - Retail urbano simulado
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Función MAPE
# ============================================================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

def run_random_forest_forecast():

    print("\n==============================")
    print("   RANDOM FOREST (con lags)")
    print("==============================\n")

    # ==========================================================
    # 1. Cargar datos
    # ==========================================================

    df = pd.read_csv("data/demanda_semanal.csv", sep=";")
    df.columns = df.columns.str.strip()

    df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
    df["demanda"] = pd.to_numeric(df["demanda"], errors="coerce")
    df = df.dropna()

    # ==========================================================
    # 2. Crear variables rezagadas (lags)
    # ==========================================================

    df["lag1"] = df["demanda"].shift(1)
    df["lag2"] = df["demanda"].shift(2)
    df["mes"] = (df["semana"] % 52) // 4 + 1

    df = df.dropna()

    features = ["lag1", "lag2", "mes"]
    target = "demanda"

    # ==========================================================
    # 3. División OUT-OF-SAMPLE (40 train / 41–52 test)
    # ==========================================================

    train = df[df["semana"] <= 40]
    test  = df[df["semana"] >= 41]

    # ==========================================================
    # 4. Entrenamiento del modelo
    # ==========================================================

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(train[features], train[target])

    test = test.copy()
    test["pred"] = model.predict(test[features])

    # ==========================================================
    # 5. Crear carpeta de salida
    # ==========================================================

    output_path = "results/random_forest"
    os.makedirs(output_path, exist_ok=True)

    # ==========================================================
    # 6. Guardar forecast 41–52
    # ==========================================================

    forecast_series = test[["semana", "demanda", "pred"]].copy()
    forecast_series.columns = ["semana", "demanda_real", "prediccion_rf"]

    forecast_series.to_csv(
        f"{output_path}/random_forest_forecast_series.csv",
        index=False
    )

    print("✅ Forecast exportado correctamente.")

    # ==========================================================
    # 7. Gráfico
    # ==========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(test["semana"], test["demanda"],
             label="Demanda real", marker="o")

    plt.plot(test["semana"], test["pred"],
             label="Predicción RF (lags)", marker="x")

    plt.title("Random Forest con Lags - Predicción (41–52)")
    plt.xlabel("Semana")
    plt.ylabel("Demanda")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"{output_path}/random_forest_forecast.png",
        dpi=300
    )

    plt.close()

    print("✅ Gráfico guardado correctamente.")

    # ==========================================================
    # 8. Métricas
    # ==========================================================

    mae = mean_absolute_error(test["demanda"], test["pred"])
    rmse = np.sqrt(mean_squared_error(test["demanda"], test["pred"]))
    mape = mean_absolute_percentage_error(test["demanda"], test["pred"])

    metrics_df = pd.DataFrame({
        "Modelo": ["Random Forest (lags)"],
        "MAE": [round(mae, 2)],
        "RMSE": [round(rmse, 2)],
        "MAPE (%)": [round(mape, 2)]
    })

    metrics_df.to_csv(
        f"{output_path}/random_forest_metricas.csv",
        index=False
    )

    print("\n📊 MÉTRICAS OUT-OF-SAMPLE:")
    print(metrics_df)

    print("\n==============================")
    print("   RANDOM FOREST COMPLETADO")
    print("==============================\n")


if __name__ == "__main__":
    run_random_forest_forecast()