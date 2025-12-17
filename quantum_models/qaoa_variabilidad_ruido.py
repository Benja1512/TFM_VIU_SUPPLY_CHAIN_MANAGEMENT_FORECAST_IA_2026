import time
import os
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# ✅ QAOA clásico (antiguo)
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance, algorithm_globals

# =====================================================
# SEMILLA
# =====================================================
algorithm_globals.random_seed = 42


# =====================================================
# PROBLEMA BINARIO
# =====================================================
def crear_problema_inventario():
    qp = QuadraticProgram("Inventario_QAOA")
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
# RUIDO
# =====================================================
def crear_noise_model():
    noise_model = NoiseModel()
    error_1q = depolarizing_error(0.01, 1)
    error_2q = depolarizing_error(0.05, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "ry", "rz"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    readout_error = ReadoutError([[0.95, 0.05],
                                  [0.10, 0.90]])
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


# =====================================================
# QAOA + RUIDO
# =====================================================
def ejecutar_qaoa_variabilidad(num_corridas=10):
    print("\n🚀 QAOA con VARIABILIDAD y RUIDO CUÁNTICO (Qiskit clásico)\n")

    noise_model = crear_noise_model()
    backend = AerSimulator(noise_model=noise_model)

    # ✅ quantum_instance clásico
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
            "x_AT1": res.x[0],
            "x_AT2": res.x[1],
            "x_BT1": res.x[2],
            "x_BT2": res.x[3],
        })

        print(f"   ✔ Costo={res.fval:.3f} | x={res.x}")

    return pd.DataFrame(resultados)


# =====================================================
# GUARDAR CSV + GRAF
# =====================================================
def guardar_resultados(df):
    base = "quantum_models/results"
    graf = os.path.join(base, "graficos")
    os.makedirs(graf, exist_ok=True)

    csv_path = os.path.join(base, "qaoa_variabilidad_ruido.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["corrida"], df["costo"], marker="o", label="Costo")
    plt.plot(df["corrida"], df["tiempo_s"], marker="x", label="Tiempo")
    plt.title("Variabilidad de QAOA bajo ruido cuántico")
    plt.xlabel("Corrida")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    img_path = os.path.join(graf, "qaoa_variabilidad_ruido.png")
    plt.savefig(img_path)
    plt.close()

    print(f"\n💾 CSV: {csv_path}")
    print(f"📈 Gráfico: {img_path}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    df = ejecutar_qaoa_variabilidad(num_corridas=10)
    print("\n📊 Resultados:")
    print(df)
    guardar_resultados(df)
    print("\n✅ VARIABILIDAD CON RUIDO CUÁNTICO COMPLETADA")
