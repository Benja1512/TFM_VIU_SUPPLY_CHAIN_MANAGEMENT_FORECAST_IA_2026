import pandas as pd

def simular_escenario(df, col_reposicion, nombre_escenario):
    inventario = 0
    resultados = []

    for i, row in df.iterrows():
        semana = row['semana']
        demanda = row['demanda_real']
        reposicion = row[col_reposicion]

        inventario_final = inventario + reposicion - demanda

        stockout = 0 if inventario + reposicion >= demanda else demanda - (inventario + reposicion)
        sobrestock = inventario_final if inventario_final > 0 else 0

        resultados.append({
            'Semana': semana,
            'Demanda (u)': round(demanda, 2),
            'Reposición (u)': round(reposicion, 2),
            'Inventario Final (u)': round(inventario_final, 2),
            'Stockout (u)': round(stockout, 2),
            'Sobrestock (u)': round(sobrestock, 2)
        })

        inventario = max(inventario_final, 0)  # No puede ser negativo

    df_resultado = pd.DataFrame(resultados)
    print(f"\n📦 {nombre_escenario} – Resultados de simulación logística:\n")
    print(df_resultado.to_string(index=False))

    return df_resultado

def ejecutar_simulacion_ann():
    df = pd.read_csv('results/escenarios/ann_escenarios.csv')

    escenarios = ['Escenario_A', 'Escenario_B', 'Escenario_C', 'Escenario_D', 'Escenario_E']
    nombres = ['Escenario A (conservador)',
               'Escenario B (ligeramente conservador)',
               'Escenario C (equilibrado)',
               'Escenario D (ligeramente agresivo)',
               'Escenario E (agresivo)']

    for esc, nombre in zip(escenarios, nombres):
        simular_escenario(df, esc, nombre)

if __name__ == "__main__":
    ejecutar_simulacion_ann()
