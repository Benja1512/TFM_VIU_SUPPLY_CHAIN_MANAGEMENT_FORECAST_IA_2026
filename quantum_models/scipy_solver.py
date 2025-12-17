# scipy_solver.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

variables = ["x_AT1", "x_AT2", "x_BT1", "x_BT2"]
c = [2, 3, 4, 1]

# Restricciones: Ax <= b
A_ub = [
    [1, 1, 0, 0],  # capacidad A
    [0, 0, 1, 1],  # capacidad B
    [-1, 0, -1, 0],  # demanda T1
    [0, -1, 0, -1]   # demanda T2
]
b_ub = [2, 2, -2, -2]

bounds = [(0, 1)] * 4  # binario relajado
integrality = [1, 1, 1, 1]

# Resolver
res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
              method='highs', integrality=integrality)

results_dir = "quantum_models/results"
os.makedirs(results_dir, exist_ok=True)

if res.success:
    valores = [round(v) for v in res.x]
    costo_total = res.fun

    # CSV
    df = pd.DataFrame([{
        "metodo": "SciPy",
        "costo": costo_total,
        "x_AT1": valores[0],
        "x_AT2": valores[1],
        "x_BT1": valores[2],
        "x_BT2": valores[3],
    }])
    df.to_csv(os.path.join(results_dir, "scipy_output.csv"), index=False)

    # Gráfico
    plt.figure(figsize=(6, 4))
    bars = plt.bar(variables, valores, color='darkorange', edgecolor='black')
    plt.ylim(0, 1.2)
    plt.title("SciPy - Asignación de Stock")
    plt.ylabel("Valor (0 o 1)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for i, val in enumerate(valores):
        plt.text(i, val + 0.05, f"{int(val)}", ha='center', va='bottom', fontsize=10)

    img_path = os.path.join(results_dir, "scipy_solution.png")
    plt.savefig(img_path)
    plt.show()

    # Consola
    print("=== SOLUCIÓN SCIPY ===")
    for var, val in zip(variables, valores):
        print(f"{var} = {val}")
    print(f"costo total = {costo_total}")

else:
    print("No se encontró solución válida.")
