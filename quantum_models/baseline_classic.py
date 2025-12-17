import os
import pandas as pd
import matplotlib.pyplot as plt
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ===================================
# 1. DEFINIR EL PROBLEMA BINARIO
# ===================================
qp = QuadraticProgram()
qp.binary_var("x_AT1")
qp.binary_var("x_AT2")
qp.binary_var("x_BT1")
qp.binary_var("x_BT2")

# Restricciones de capacidad por almacén
qp.linear_constraint({"x_AT1": 1, "x_AT2": 1}, "<=", 2, "capacidad_A")
qp.linear_constraint({"x_BT1": 1, "x_BT2": 1}, "<=", 2, "capacidad_B")

# Restricciones de demanda por tienda
qp.linear_constraint({"x_AT1": 1, "x_BT1": 1}, ">=", 2, "demanda_T1")
qp.linear_constraint({"x_AT2": 1, "x_BT2": 1}, ">=", 2, "demanda_T2")

# Función objetivo: minimizar costos de asignación
qp.minimize(linear={"x_AT1": 2, "x_AT2": 3, "x_BT1": 4, "x_BT2": 1})

# ===================================
# 2. RESOLVER CON MÉTODO CLÁSICO
# ===================================
solver = NumPyMinimumEigensolver()
classic_optimizer = MinimumEigenOptimizer(solver)
result_classic = classic_optimizer.solve(qp)

# ===================================
# 3. MOSTRAR RESULTADO EN CONSOLA
# ===================================
print("\n=== SOLUCIÓN CLÁSICA ===")
print(result_classic.prettyprint())

# ===================================
# 4. GUARDAR RESULTADOS EN CSV
# ===================================
results_dir = "quantum_models/results"
os.makedirs(results_dir, exist_ok=True)

df = pd.DataFrame([{
    "metodo": "Clásico",
    "costo": result_classic.fval,
    "x_AT1": result_classic.x[0],
    "x_AT2": result_classic.x[1],
    "x_BT1": result_classic.x[2],
    "x_BT2": result_classic.x[3],
}])
df.to_csv(os.path.join(results_dir, "baseline_output.csv"), index=False)

# ===================================
# 5. GENERAR Y GUARDAR GRÁFICO
# ===================================
variables = ["x_AT1", "x_AT2", "x_BT1", "x_BT2"]
valores = result_classic.x

plt.figure(figsize=(6, 4))
plt.bar(variables, valores, color='skyblue', edgecolor='black')
plt.ylim(0, 1.2)
plt.title("Solución Clásica - Asignación de Stock")
plt.ylabel("Valor (0 o 1)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir valores 0 o 1 encima de cada barra
for i, val in enumerate(valores):
    plt.text(i, val + 0.05, f"{int(val)}", ha='center', va='bottom', fontsize=10)

# Guardar en carpeta graficos
graficos_dir = os.path.join(results_dir, "graficos")
os.makedirs(graficos_dir, exist_ok=True)
img_path = os.path.join(graficos_dir, "baseline_solution.png")
plt.tight_layout()
plt.savefig(img_path)
plt.show()
