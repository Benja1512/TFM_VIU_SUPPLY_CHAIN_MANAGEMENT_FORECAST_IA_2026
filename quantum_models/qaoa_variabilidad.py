import os
import random
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BackendSampler

# ===============================
# CONFIGURACIÓN
# ===============================
simulador = AerSimulator()
num_corridas = 10  # 🔁 Número de simulaciones

results_dir = "quantum_models/results"
csv_path = os.path.join(results_dir, "qaoa_variabilidad_aer.csv")
img_path = os.path.join(results_dir, "graficos/qaoa_variabilidad_aer.png")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.dirname(img_path), exist_ok=True)


# ===============================
# DEFINIR PROBLEMA
# ===============================
def crear_problema():
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
    return qp


# ===============================
# EJECUTAR VARIAS SIMULACIONES
# ===============================
resultados = []
for i in range(1, num_corridas + 1):
    print(f"🔁 Corrida {i}")

    seed = random.randint(1, 10000)
    sampler = BackendSampler(backend=simulador, options={"shots": 1024, "seed_simulator": seed})

    qaoa = QAOA(optimizer=COBYLA(maxiter=200), reps=2, sampler=sampler)
    optimizer = MinimumEigenOptimizer(qaoa)

    problema = crear_problema()
    resultado = optimizer.solve(problema)

    print("   ➔ Costo:", resultado.fval)
    print(f"   ➔ x_AT1={resultado.x[0]} | x_AT2={resultado.x[1]} | x_BT1={resultado.x[2]} | x_BT2={resultado.x[3]}")
    print("-" * 60)

    resultados.append({
        "corrida": i,
        "semilla": seed,
        "costo": resultado.fval,
        "x_AT1": resultado.x[0],
        "x_AT2": resultado.x[1],
        "x_BT1": resultado.x[2],
        "x_BT2": resultado.x[3],
    })


# ===============================
# GUARDAR Y GRAFICAR
# ===============================
df = pd.DataFrame(resultados)
df.to_csv(csv_path, index=False)
print(f"\n✅ Resultados guardados en: {csv_path}")

plt.figure(figsize=(8, 4))
plt.bar(df["corrida"], df["costo"], color='orchid', edgecolor='black')
plt.xlabel("Corrida QAOA con AerSimulator")
plt.ylabel("Costo total")
plt.title("Variabilidad cuántica realista en QAOA (AerSimulator + shots)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
for i, val in enumerate(df["costo"]):
    plt.text(i + 1, val + 0.3, f"{val:.1f}", ha='center')
plt.tight_layout()
plt.savefig(img_path)
plt.show()
print(f"📊 Gráfico guardado en: {img_path}")
