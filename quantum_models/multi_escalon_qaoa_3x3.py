import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.visualization import plot_distribution

# ===============================
# Configuración general
# ===============================
np.random.seed(123)
results_dir = "quantum_models/results"
os.makedirs(results_dir, exist_ok=True)

# ===============================
# Definir problema (3 almacenes → 3 tiendas)
# ===============================
qp = QuadraticProgram()

# Variables binarias (A, B, C → T1, T2, T3)
vars_names = [
    "x_AT1", "x_AT2", "x_AT3",
    "x_BT1", "x_BT2", "x_BT3",
    "x_CT1", "x_CT2", "x_CT3"
]
for v in vars_names:
    qp.binary_var(v)

# Restricciones: cada almacén puede enviar máx 2
qp.linear_constraint({"x_AT1": 1, "x_AT2": 1, "x_AT3": 1}, "<=", 2, "capacidad_A")
qp.linear_constraint({"x_BT1": 1, "x_BT2": 1, "x_BT3": 1}, "<=", 2, "capacidad_B")
qp.linear_constraint({"x_CT1": 1, "x_CT2": 1, "x_CT3": 1}, "<=", 2, "capacidad_C")

# Restricciones: cada tienda debe recibir al menos 2
qp.linear_constraint({"x_AT1": 1, "x_BT1": 1, "x_CT1": 1}, ">=", 2, "demanda_T1")
qp.linear_constraint({"x_AT2": 1, "x_BT2": 1, "x_CT2": 1}, ">=", 2, "demanda_T2")
qp.linear_constraint({"x_AT3": 1, "x_BT3": 1, "x_CT3": 1}, ">=", 2, "demanda_T3")

# Costos
costs = {
    "x_AT1": 3, "x_AT2": 5, "x_AT3": 3,
    "x_BT1": 2, "x_BT2": 4, "x_BT3": 3,
    "x_CT1": 4, "x_CT2": 2, "x_CT3": 2
}
qp.minimize(linear=costs)

print("Costos de transporte:", costs)

# ===============================
# Resolver con método clásico
# ===============================
classic_solver = NumPyMinimumEigensolver()
classic_optimizer = MinimumEigenOptimizer(classic_solver)
result_classic = classic_optimizer.solve(qp)

print("\n=== 🔵 SOLUCIÓN CLÁSICA ===")
print(result_classic.prettyprint())

# ===============================
# Resolver con QAOA
# ===============================
sampler = Sampler()
qaoa = QAOA(optimizer=COBYLA(maxiter=200), reps=2, sampler=sampler)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)
result_qaoa = qaoa_optimizer.solve(qp)

print("\n=== 🟠 SOLUCIÓN QAOA ===")
print(result_qaoa.prettyprint())

# ===============================
# Guardar resultados en CSV
# ===============================
df = pd.DataFrame([
    {"metodo": "Clásico", "costo": result_classic.fval, **dict(zip(vars_names, result_classic.x))},
    {"metodo": "QAOA", "costo": result_qaoa.fval, **dict(zip(vars_names, result_qaoa.x))}
])
csv_path = os.path.join(results_dir, "qaoa_vs_classic_3x3.csv")
df.to_csv(csv_path, index=False)

# ===============================
# Gráfico 1: Comparación de costos
# ===============================
plt.figure(figsize=(6, 4))
plt.bar(df["metodo"], df["costo"], color=["steelblue", "darkorange"])
plt.title("Comparación de costos (3x3): Clásico vs QAOA")
plt.ylabel("Costo total")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
img1 = os.path.join(results_dir, "qaoa_vs_classic_3x3.png")
plt.savefig(img1)

# ===============================
# Gráfico 2: Variables por método
# ===============================
x = np.arange(len(vars_names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, result_classic.x, width, label="Clásico", color="steelblue")
plt.bar(x + width/2, result_qaoa.x, width, label="QAOA", color="darkorange")

plt.xticks(x, vars_names, rotation=45)
plt.ylabel("Valor (0 o 1)")
plt.title("Comparación de variables (3x3)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
img2 = os.path.join(results_dir, "comparacion_variables_3x3.png")
plt.savefig(img2)

# ===============================
# Gráfico 3: Histograma de probabilidades QAOA
# ===============================
probs = result_qaoa.min_eigen_solver_result.eigenstate.binary_probabilities()

plt.figure(figsize=(12, 5))
plot_distribution(probs, title="Distribución de soluciones QAOA (3×3)")
plt.tight_layout()
img3 = os.path.join(results_dir, "qaoa_distribution_histogram_3x3.png")
plt.savefig(img3)

# ===============================
# Gráfico 4: Heatmap de probabilidades QAOA
# ===============================
bitstrings = list(probs.keys())
values = list(probs.values())
matrix_size = int(np.ceil(np.sqrt(len(values))))

# Rellenamos la matriz cuadrada
prob_matrix = np.zeros((matrix_size, matrix_size))
for i, p in enumerate(values):
    r, c = divmod(i, matrix_size)
    prob_matrix[r, c] = p

plt.figure(figsize=(8, 6))
sns.heatmap(prob_matrix, cmap="viridis", annot=False, cbar=True)
plt.title("Heatmap de probabilidades QAOA (3×3)")
plt.tight_layout()
img4 = os.path.join(results_dir, "qaoa_distribution_heatmap_3x3.png")
plt.savefig(img4)

# ===============================
# Mostrar todas las imágenes juntas
# ===============================
plt.show()

# ===============================
# Final
# ===============================
print(f"\n✅ Resultados guardados en:\n- {csv_path}\n- {img1}\n- {img2}\n- {img3}\n- {img4}")
