import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# ANN_Keras – Forecast semanal demanda retail
# Producto: Leche entera 1L (Utrecht)
# Train: semanas 1–40
# Test:  semanas 41–52
# Output: results/ann_keras/
# ============================================================

def run_ann_keras_forecast():

    print("\n=======================================")
    print("   🚀 EJECUTANDO ANN_Keras")
    print("=======================================\n")

    # ------------------------------------------------------------
    # 1. Rutas organizadas por modelo (robustas)
    # ------------------------------------------------------------
    BASE_DIR = os.path.dirname(__file__)

    data_file = os.path.join(BASE_DIR, "../data/demanda_semanal.csv")

    output_dir = os.path.join(BASE_DIR, "../results/ann_keras")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 2. Cargar datos (ruta segura)
    # ------------------------------------------------------------
    df = pd.read_csv(data_file, sep=";")
    df.columns = df.columns.str.strip()

    df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
    df["demanda"] = pd.to_numeric(df["demanda"], errors="coerce")
    df = df.dropna()

    print("✅ Datos cargados correctamente.")
    print(f"📄 Archivo leído desde: {data_file}")

    # ------------------------------------------------------------
    # 3. Feature Engineering temporal
    # ------------------------------------------------------------
    df["mes"] = (df["semana"] % 52) // 4 + 1
    df["dia"] = (df["semana"] % 7) + 1

    X = df[["semana", "mes", "dia"]]
    y = df["demanda"]

    # ------------------------------------------------------------
    # 4. Split temporal EXACTO (TFM)
    # ------------------------------------------------------------
    X_train = X.iloc[:40]
    y_train = y.iloc[:40]

    X_test = X.iloc[40:]
    y_test = y.iloc[40:]

    print("\n📌 Split temporal aplicado:")
    print("Train semanas:", int(X_train["semana"].min()), "-", int(X_train["semana"].max()))
    print("Test semanas :", int(X_test["semana"].min()), "-", int(X_test["semana"].max()))

    # ------------------------------------------------------------
    # 5. Normalización
    # ------------------------------------------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # 6. Arquitectura ANN
    # ------------------------------------------------------------
    model = Sequential([
        Dense(64, activation="relu", input_shape=(3,)),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    # ------------------------------------------------------------
    # 7. Entrenamiento
    # ------------------------------------------------------------
    print("\n🚀 Entrenando red neuronal...\n")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=8,
        callbacks=[early_stop],
        verbose=0
    )

    print("✅ Entrenamiento completado.")

    # ------------------------------------------------------------
    # 8. Predicción
    # ------------------------------------------------------------
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    # ------------------------------------------------------------
    # 9. Métricas estadísticas
    # ------------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics_df = pd.DataFrame({
        "Modelo": ["ANN_Keras"],
        "MAE": [round(mae, 2)],
        "RMSE": [round(rmse, 2)],
        "MAPE": [round(mape, 2)]
    })

    # ------------------------------------------------------------
    # 10. Guardar métricas y forecast series (en results/ann_keras)
    # ------------------------------------------------------------
    metrics_path = os.path.join(output_dir, "ann_keras_metricas.csv")
    metrics_df.to_csv(metrics_path, index=False)

    forecast_df = pd.DataFrame({
        "Semana": X_test["semana"].values,
        "Demanda_real": y_test.values,
        "Prediccion_ANN_Keras": y_pred
    })

    forecast_path = os.path.join(output_dir, "ann_keras_forecast_series.csv")
    forecast_df.to_csv(forecast_path, index=False)

    # ------------------------------------------------------------
    # 11. Gráfico Forecast
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(X_test["semana"], y_test.values, marker="o", label="Demanda real")
    plt.plot(X_test["semana"], y_pred, marker="x", label="ANN_Keras")

    plt.title("Forecast Semanal – ANN_Keras")
    plt.xlabel("Semana")
    plt.ylabel("Demanda")
    plt.grid(True)
    plt.legend()

    forecast_png = os.path.join(output_dir, "ann_keras_forecast.png")
    plt.savefig(forecast_png, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 12. Gráfico Error
    # ------------------------------------------------------------
    errores = y_test.values - y_pred

    plt.figure(figsize=(10, 4))
    plt.plot(X_test["semana"], errores, marker="o")
    plt.title("Error de Predicción – ANN_Keras")
    plt.xlabel("Semana")
    plt.ylabel("Error")
    plt.grid(True)

    error_png = os.path.join(output_dir, "ann_keras_errores.png")
    plt.savefig(error_png, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 13. Imprimir resultado en terminal
    # ------------------------------------------------------------
    print("\n=======================================")
    print("      ANN_KERAS OUT-OF-SAMPLE OK")
    print("=======================================")
    print(metrics_df)
    print("📁 Resultados guardados en: results/ann_keras")
    print(f"📄 Métricas: {metrics_path}")
    print(f"📄 Forecast : {forecast_path}")
    print(f"🖼️ Gráfico  : {forecast_png}")
    print(f"🖼️ Error    : {error_png}\n")


# ============================================================
if __name__ == "__main__":
    run_ann_keras_forecast()