# brute_force_solver.py

import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt

# Variables binarias
variables = ["x_AT1", "x_AT2", "x_BT1", "x_BT2"]

# Explorar todas las combinaciones binarias (2^4)
soluciones = []
for bits in itertools.product([0, 1], repeat=4):
    x_AT1, x_AT2, x_BT1, x_BT2 = bits

    # Restricciones
    if x_AT1 + x_AT2 > 2: continue
    if x_BT1 + x_BT2 > 2: continue
    if x_AT1 + x_BT1 < 2: continue
    if x_AT2 + x_BT2 < 2: continue

    # Función objetivo
    costo = 2 * x_AT1 + 3 * x_AT2 + 4 * x_BT1 + 1 * x_BT2

    soluciones.append({
        "x_AT1": x_AT1, "x_AT2": x_AT2,
        "x_BT1": x_BT1, "x_BT2": x_BT2,
        "costo": costo
    })

# Convertir a DataFrame
df = pd.DataFrame(soluciones)
mejor = df.sort_values(by="costo").head(1)

print("=== SOLUCIÓN FUERZA BRUTA ===")
print(mejor.to_string(index=False))

# === Guardar en CSV ===
results_dir = "quantum_models/results"
os.makedirs(results_dir, exist_ok=True)
df.to_csv(os.path.join(results_dir, "brute_force_output.csv"), index=False)

# === Guardar gráfico ===
valores = mejor.iloc[0][variables].values

plt.figure(figsize=(6, 4))
bars = plt.bar(variables, valores, color='royalblue', edgecolor='black')
plt.ylim(0, 1.2)
plt.title("Fuerza Bruta - Asignación de Stock")
plt.ylabel("Valor (0 o 1)")
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Añadir texto sobre barras
for i, val in enumerate(valores):
    plt.text(i, val + 0.05, f"{int(val)}", ha='center', va='bottom', fontsize=10)

img_path = os.path.join(results_dir, "brute_force_solution.png")
plt.savefig(img_path)
plt.show()
