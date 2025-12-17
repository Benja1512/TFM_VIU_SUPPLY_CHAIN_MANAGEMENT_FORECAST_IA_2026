import os
import pandas as pd
import matplotlib.pyplot as plt
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals  # ✅ CORREGIDO
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ================================
# 1. Preparar backend moderno (Sampler)
# ================================
algorithm_globals.random_seed = 123
sampler = Sampler()

# ================================
# 2. Definir problema binario
# ================================
qp = QuadraticProgram()
qp.binary_var("x_AT1")
qp.binary_var("x_AT2")
qp.binary_var("x_BT1")
qp.binary_var("x_BT2")

qp.linear_constraint({"x_AT1": 1, "x_AT2": 1}, "<=", 2, "capacidad_A")
qp.linear_constraint({"x_BT1": 1, "x_BT2": 1}, "<=", 2, "capacidad_B")
qp.linear_constraint({"x_AT1": 1, "x_BT1": 1}, ">=", 2, "demanda_T1")
qp.linear_constraint({"x_AT2": 1, "x_BT2": 1}, ">=", 2, "demanda_T2")

qp.minimize(linear={"x_AT1": 2, "x_AT2": 3, "x_BT1": 4, "x_BT2": 1})

# ================================
# 3. Ejecutar QAOA con Sampler
# ================================
qaoa = QAOA(optimizer=COBYLA(maxiter=200), reps=2, sampler=sampler)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)
result_qaoa = qaoa_optimizer.solve(qp)

# ================================
# 4. Mostrar resultado
# ================================
print("\n=== SOLUCIÓN QAOA ===")
print(result_qaoa.prettyprint())

# ================================
# 5. Guardar resultado en CSV
# ================================
results_dir = "quantum_models/results"
os.makedirs(results_dir, exist_ok=True)

df = pd.DataFrame([{
    "metodo": "QAOA",
    "costo": result_qaoa.fval,
    "x_AT1": result_qaoa.x[0],
    "x_AT2": result_qaoa.x[1],
    "x_BT1": result_qaoa.x[2],
    "x_BT2": result_qaoa.x[3],
}])
df.to_csv(os.path.join(results_dir, "qaoa_output.csv"), index=False)

# ================================
# 6. Guardar gráfico
# ================================
graficos_dir = os.path.join(results_dir, "graficos")
os.makedirs(graficos_dir, exist_ok=True)

variables = ["x_AT1", "x_AT2", "x_BT1", "x_BT2"]
valores = result_qaoa.x

plt.figure(figsize=(6, 4))
bars = plt.bar(variables, valores, color='mediumseagreen', edgecolor='black')
plt.ylim(0, 1.2)
plt.title("Solución QAOA - Asignación de Stock")
plt.ylabel("Valor (0 o 1)")
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i, val in enumerate(valores):
    plt.text(i, val + 0.05, f"{int(val)}", ha='center', va='bottom', fontsize=10)

img_path = os.path.join(graficos_dir, "qaoa_solution.png")
plt.savefig(img_path)
plt.show()
