import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def run_ann_forecast():
    # Leer datos
    df = pd.read_csv('data/demanda_semanal.csv', sep=';')
    df.columns = df.columns.str.strip()
    df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
    df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
    df = df.dropna()

    # Feature engineering
    df['mes'] = (df['semana'] % 52) // 4 + 1
    df['dia'] = (df['semana'] % 7) + 1

    # Features y target
    X = df[['semana', 'mes', 'dia']]
    y = df['demanda']

    # Split 80/20 sin mezclar (serie temporal)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Modelo ANN
    model = Sequential()
    model.add(Dense(64, input_dim=3, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # salida

    model.compile(optimizer='adam', loss='mse')

    # Entrenar
    model.fit(X_train, y_train, epochs=200, verbose=0)

    # Predecir
    y_pred = model.predict(X_test).flatten()

    # Crear carpetas si no existen
    os.makedirs('results/graficos', exist_ok=True)
    os.makedirs('results/metricas', exist_ok=True)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics_df = pd.DataFrame({
        'modelo': ['ANN'],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape]
    })
    metrics_df.to_csv('results/metricas/ann_metricas.csv', index=False)

    # Mostrar métricas
    print("\n==============================")
    print("✅ ANN terminado.")
    print(metrics_df.to_string(index=False))
    print("==============================\n")

    # Guardar predicciones
    semanas_test = X_test['semana'].values
    pred_df = pd.DataFrame({
        'semana': semanas_test,
        'demanda_real': y_test.values,
        'demanda_predicha': y_pred
    })
    pred_df.to_csv('results/metricas/ann_predicciones.csv', index=False)

    # Gráfico de predicción
    plt.figure(figsize=(10, 6))
    plt.plot(semanas_test, y_test.values, label='Real', marker='o')
    plt.plot(semanas_test, y_pred, label='Predicción ANN', marker='x')
    plt.title('Predicción de Demanda Semanal – Modelo ANN (Keras)')
    plt.xlabel('Semana')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/graficos/ann_forecast.png', dpi=300)
    plt.show()
    plt.close()

    # Gráfico de errores
    errores = y_test.values - y_pred
    plt.figure(figsize=(10, 4))
    plt.plot(semanas_test, errores, marker='o', color='red')
    plt.title('Error de Predicción – ANN')
    plt.xlabel('Semana')
    plt.ylabel('Error (Real - Predicho)')
    plt.grid(True)
    plt.savefig('results/graficos/ann_errores.png', dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    run_ann_forecast()
