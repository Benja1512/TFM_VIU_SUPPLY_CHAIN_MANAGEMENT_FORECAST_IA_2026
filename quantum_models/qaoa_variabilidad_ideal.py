import time
import os
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance, algorithm_globals

# =====================================================
# Semilla para reproducibilidad
# =====================================================
algorithm_globals.random_seed = 42


# =====================================================
# Problema binario simple (4 variables)
# =====================================================
def crear_problema_inventario():
    qp = QuadraticProgram("Inventario_QAOA_Ideal")

    qp.binary_var("x_AT1")
    qp.binary_var("x_AT2")
    qp.binary_var("x_BT1")
    qp.binary_var("x_BT2")

    qp.minimize(
        linear=[5, 3, 4, 6],
        quadratic={(0, 2): 2, (1, 3): 2}
    )
    return qp


# =====================================================
# QAOA IDEAL (sin ruido)
# =====================================================
def ejecutar_qaoa_ideal(num_corridas=10):
    print("\n🚀 QAOA IDEAL sin ruido (AerSimulator clásico)\n")

    backend = AerSimulator()  # 🚫 sin ruido
    quantum_instance = QuantumInstance(backend, shots=2048)

    qaoa = QAOA(
        optimizer=COBYLA(maxiter=100),
        reps=6,
        quantum_instance=quantum_instance
    )

    solver = MinimumEigenOptimizer(qaoa)
    qp = crear_problema_inventario()

    resultados = []

    for i in range(num_corridas):
        print(f"🔁 Corrida {i + 1}/{num_corridas}")
        inicio = time.time()
        res = solver.solve(qp)
        fin = time.time()

        resultados.append({
            "corrida": i + 1,
            "costo": res.fval,
            "tiempo_s": round(fin - inicio, 3),
            **{var.name: val for var, val in zip(qp.variables, res.x)}
        })

        print(f"   ✔ Costo={res.fval:.3f} | x={res.x}")

    return pd.DataFrame(resultados)


# =====================================================
# Guardado de resultados
# =====================================================
def guardar_resultados(df):
    base = "quantum_models/results"
    graf = os.path.join(base, "graficos")
    os.makedirs(graf, exist_ok=True)

    csv_path = os.path.join(base, "qaoa_variabilidad_ideal.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["corrida"], df["costo"], marker="o", label="Costo")
    plt.plot(df["corrida"], df["tiempo_s"], marker="x", label="Tiempo")
    plt.title("QAOA ideal (sin ruido) – Variabilidad")
    plt.xlabel("Corrida")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    img_path = os.path.join(graf, "qaoa_variabilidad_ideal.png")
    plt.savefig(img_path)
    plt.close()

    print(f"\n💾 CSV: {csv_path}")
    print(f"📈 Gráfico: {img_path}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    df = ejecutar_qaoa_ideal(num_corridas=10)
    print("\n📊 Resultados:")
    print(df)
    guardar_resultados(df)
    print("\n✅ QAOA IDEAL finalizado correctamente")
