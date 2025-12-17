import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Prophet
from prophet import Prophet

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# ANN (Keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

def comparar_series_ia():
    # Leer datos
    df = pd.read_csv('data/demanda_semanal.csv', sep=';')
    df.columns = df.columns.str.strip()
    df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
    df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
    df = df.dropna()

    # Feature engineering
    df['mes'] = (df['semana'] % 52) // 4 + 1
    df['dia'] = (df['semana'] % 7) + 1

    # Split 80/20
    X = df[['semana', 'mes', 'dia']]
    y = df['demanda']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    semanas_test = X_test['semana'].values

    # ===================== Prophet =====================
    # Crear columna fecha ficticia a partir de semana (semana 1 = 2020-01-01)
    start_date = pd.to_datetime("2020-01-01")
    df_prophet = df[['semana', 'demanda']].copy()
    df_prophet['ds'] = start_date + pd.to_timedelta(df_prophet['semana'] - 1, unit="W")
    df_prophet = df_prophet.rename(columns={'demanda': 'y'})

    m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(df_prophet.iloc[:len(X_train)])  # entrenar solo con train

    # Fechas de test
    future = pd.DataFrame({'ds': df_prophet['ds'].iloc[-len(y_test):]})
    forecast = m.predict(future)
    y_pred_prophet = forecast['yhat'].values

    # ===================== Random Forest =====================
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    # ===================== ANN (Keras) =====================
    model_ann = Sequential()
    model_ann.add(Dense(64, input_dim=3, activation='relu'))
    model_ann.add(Dense(32, activation='relu'))
    model_ann.add(Dense(1))
    model_ann.compile(optimizer='adam', loss='mse')
    model_ann.fit(X_train, y_train, epochs=200, verbose=0)
    y_pred_ann = model_ann.predict(X_test).flatten()

    # ===================== Consolidar resultados =====================
    df_result = pd.DataFrame({
        'semana': semanas_test,
        'real': y_test.values,
        'prophet': y_pred_prophet,
        'random_forest': y_pred_rf,
        'ann': y_pred_ann
    })

    os.makedirs('results/metricas', exist_ok=True)
    df_result.to_csv('results/metricas/comparacion_series_ia.csv', index=False)

    # ===================== Gráfico =====================
    plt.figure(figsize=(12, 6))
    plt.plot(df_result['semana'], df_result['real'], label="Real", marker='o', linewidth=2, color='black')
    plt.plot(df_result['semana'], df_result['prophet'], label="Prophet", marker='x', linestyle='--')
    plt.plot(df_result['semana'], df_result['random_forest'], label="Random Forest", marker='s', linestyle='--')
    plt.plot(df_result['semana'], df_result['ann'], label="ANN", marker='d', linestyle='--')

    plt.title("Comparación de Predicciones (IA) vs Demanda Real")
    plt.xlabel("Semana")
    plt.ylabel("Demanda")
    plt.legend()
    plt.grid(True)

    os.makedirs('results/graficos', exist_ok=True)
    plt.savefig('results/graficos/comparacion_series_ia.png', dpi=300)
    plt.show()
    plt.close()

    print("\n✅ Comparación de series (líneas) generada en results/graficos/comparacion_series_ia.png")

if __name__ == "__main__":
    comparar_series_ia()
