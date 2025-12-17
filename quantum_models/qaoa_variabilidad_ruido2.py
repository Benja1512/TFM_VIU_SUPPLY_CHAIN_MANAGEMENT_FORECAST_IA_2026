import time
import os
import pandas as pd
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance, algorithm_globals

# =====================================================
# SEMILLA
# =====================================================
algorithm_globals.random_seed = 42


# =====================================================
# NUEVO PROBLEMA BINARIO + DIFÍCIL
# =====================================================
def crear_problema_inventario():
    qp = QuadraticProgram("Inventario_QAOA_Dificil")

    # Variables binarias (6 en lugar de 4)
    for var in ["x1", "x2", "x3", "x4", "x5", "x6"]:
        qp.binary_var(name=var)

    # Función objetivo más compleja (coeficientes positivos y negativos)
    qp.minimize(
        linear=[3, -2, 4, -1, 5, -3],
        quadratic={
            ("x1", "x2"): 2,
            ("x2", "x3"): -3,
            ("x3", "x4"): 1,
            ("x4", "x5"): 2,
            ("x5", "x6"): -2,
            ("x1", "x6"): 3
        }
    )
    return qp


# =====================================================
# RUIDO CUÁNTICO
# =====================================================
def crear_noise_model():
    noise_model = NoiseModel()

    error_1q = depolarizing_error(0.03, 1)  # más ruido
    error_2q = depolarizing_error(0.08, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, ["rx", "ry", "rz"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    readout_error = ReadoutError([[0.92, 0.08],
                                  [0.15, 0.85]])
    noise_model.add_all_qubit_readout_error(readout_error)

    return noise_model


# =====================================================
# QAOA CON RUIDO Y VARIABILIDAD
# =====================================================
def ejecutar_qaoa_variabilidad(num_corridas=10):
    print("\n🚀 QAOA DIFICULTAD EXTRA con RUIDO CUÁNTICO\n")

    noise_model = crear_noise_model()
    backend = AerSimulator(noise_model=noise_model)

    quantum_instance = QuantumInstance(backend, shots=2048)

    qaoa = QAOA(
        optimizer=COBYLA(maxiter=100),
        reps=3,  # menos profundidad para notar el ruido
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
            **{var: val for var, val in zip(qp.variables, res.x)}
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

    csv_path = os.path.join(base, "qaoa_variabilidad_ruido2.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["corrida"], df["costo"], marker="o", label="Costo")
    plt.plot(df["corrida"], df["tiempo_s"], marker="x", label="Tiempo")
    plt.title("Variabilidad de QAOA con mayor dificultad (ruido cuántico)")
    plt.xlabel("Corrida")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    img_path = os.path.join(graf, "qaoa_variabilidad_ruido2.png")
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
    print("\n✅ SIMULACIÓN DE VARIABILIDAD RUIDO2 COMPLETADA")
