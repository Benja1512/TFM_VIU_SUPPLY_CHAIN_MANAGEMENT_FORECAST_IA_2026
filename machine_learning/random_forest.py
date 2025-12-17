import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_random_forest_forecast():
    # Leer datos
    df = pd.read_csv('data/demanda_semanal.csv', sep=';')
    df.columns = df.columns.str.strip()
    df['semana'] = pd.to_numeric(df['semana'], errors='coerce')
    df['demanda'] = pd.to_numeric(df['demanda'], errors='coerce')
    df = df.dropna()

    # Feature Engineering
    df['mes'] = (df['semana'] % 52) // 4 + 1
    df['dia'] = (df['semana'] % 7) + 1

    # Entrenamiento y test (80/20)
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    features = ['semana', 'mes', 'dia']
    target = 'demanda'

    # Modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[features], train[target])

    test = test.copy()
    test['pred'] = model.predict(test[features])

    # Crear carpetas
    os.makedirs('results/graficos', exist_ok=True)
    os.makedirs('results/metricas', exist_ok=True)

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(test['semana'], test['demanda'], label='Real', marker='o')
    plt.plot(test['semana'], test['pred'], label='Predicción RF', marker='x')
    plt.title('Random Forest - Predicción de Demanda')
    plt.xlabel('Semana')
    plt.ylabel('Demanda')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/graficos/random_forest_forecast.png', dpi=300)
    plt.show()
    plt.close()

    # Métricas
    mae = mean_absolute_error(test['demanda'], test['pred'])
    rmse = np.sqrt(mean_squared_error(test['demanda'], test['pred']))
    mape = mean_absolute_percentage_error(test['demanda'], test['pred'])

    metrics_df = pd.DataFrame({
        'modelo': ['Random Forest'],
        'MAE': [mae],
        'RMSE': [rmse],
        'MAPE': [mape]
    })
    metrics_df.to_csv('results/metricas/random_forest_metricas.csv', index=False)

    print("Random Forest terminado.")
    print(metrics_df)

if __name__ == "__main__":
    run_random_forest_forecast()
