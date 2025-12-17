import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ===============================
# Configuración general
# ===============================
np.random.seed(123)  # ✅ Semilla moderna
results_dir = "quantum_models/results"
os.makedirs(results_dir, exist_ok=True)

# ===============================
# Definir problema (2 almacenes → 2 tiendas)
# ===============================
qp = QuadraticProgram()
qp.binary_var("x_AT1")
qp.binary_var("x_AT2")
qp.binary_var("x_BT1")
qp.binary_var("x_BT2")

# Restricciones
qp.linear_constraint({"x_AT1": 1, "x_AT2": 1}, "<=", 2, "capacidad_A")
qp.linear_constraint({"x_BT1": 1, "x_BT2": 1}, "<=", 2, "capacidad_B")

# ⚠️ Restricciones de demanda RELAJADAS
qp.linear_constraint({"x_AT1": 1, "x_BT1": 1}, ">=", 1, "demanda_T1")  # antes era >= 2
qp.linear_constraint({"x_AT2": 1, "x_BT2": 1}, ">=", 1, "demanda_T2")  # antes era >= 2

# Costos
qp.minimize(linear={"x_AT1": 2, "x_AT2": 3, "x_BT1": 4, "x_BT2": 1})

# ===============================
# Resolver con método clásico
# ===============================
classic_solver = NumPyMinimumEigensolver()
classic_optimizer = MinimumEigenOptimizer(classic_solver)
result_classic = classic_optimizer.solve(qp)

print("\n=== 🔵 SOLUCIÓN CLÁSICA (restricciones flexibles) ===")
print(result_classic.prettyprint())

# ===============================
# Resolver con QAOA
# ===============================
sampler = Sampler()
qaoa = QAOA(optimizer=COBYLA(maxiter=200), reps=2, sampler=sampler)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)
result_qaoa = qaoa_optimizer.solve(qp)

print("\n=== 🟠 SOLUCIÓN QAOA (restricciones flexibles) ===")
print(result_qaoa.prettyprint())

# ===============================
# Guardar resultados en CSV
# ===============================
df = pd.DataFrame([
    {
        "metodo": "Clásico",
        "costo": result_classic.fval,
        "x_AT1": result_classic.x[0],
        "x_AT2": result_classic.x[1],
        "x_BT1": result_classic.x[2],
        "x_BT2": result_classic.x[3],
    },
    {
        "metodo": "QAOA",
        "costo": result_qaoa.fval,
        "x_AT1": result_qaoa.x[0],
        "x_AT2": result_qaoa.x[1],
        "x_BT1": result_qaoa.x[2],
        "x_BT2": result_qaoa.x[3],
    }
])
csv_path = os.path.join(results_dir, "qaoa_vs_classic_flexible.csv")
df.to_csv(csv_path, index=False)

# ===============================
# Gráfico 1: Comparación de costos
# ===============================
plt.figure(figsize=(6, 4))
plt.bar(df["metodo"], df["costo"], color=["steelblue", "darkorange"])
plt.title("Comparación de costos (Restricciones flexibles)")
plt.ylabel("Costo total")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
img1 = os.path.join(results_dir, "qaoa_vs_classic_flexible.png")
plt.savefig(img1)
plt.show()

# ===============================
# Gráfico 2: Variables por método
# ===============================
variables = ["x_AT1", "x_AT2", "x_BT1", "x_BT2"]

plt.figure(figsize=(8, 5))
x = np.arange(len(variables))
width = 0.35

plt.bar(x - width/2, result_classic.x, width, label='Clásico', color='steelblue')
plt.bar(x + width/2, result_qaoa.x, width, label='QAOA', color='darkorange')

plt.xticks(x, variables)
plt.ylabel("Valor (0 o 1)")
plt.title("Comparación de variables (Restricciones flexibles)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
img2 = os.path.join(results_dir, "comparacion_variables_flexible.png")
plt.savefig(img2)
plt.show()

# ===============================
# Final
# ===============================
print(f"\n✅ Resultados guardados en:\n- {csv_path}\n- {img1}\n- {img2}")
