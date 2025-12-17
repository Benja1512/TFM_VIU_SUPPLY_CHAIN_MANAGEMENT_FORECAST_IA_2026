import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeVigo

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import BackendSampler

from qiskit.utils import algorithm_globals

# ===============================
# CONFIGURACIÓN GLOBAL
# ===============================
algorithm_globals.random_seed = 42

NUM_QUBITS = 10
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# CREAR HAMILTONIANO QUBO (10 qubits)
# ===============================
def crear_hamiltoniano():
    paulis = []
    coeffs = []

    # Términos lineales
    for i in range(NUM_QUBITS):
        z = ["I"] * NUM_QUBITS
        z[i] = "Z"
        paulis.append("".join(z))
        coeffs.append(-1.0)

    # Términos cuadráticos (más dificultad)
    for i in range(NUM_QUBITS - 1):
        zz = ["I"] * NUM_QUBITS
        zz[i] = "Z"
        zz[i + 1] = "Z"
        paulis.append("".join(zz))
        coeffs.append(0.8)

    return SparsePauliOp(paulis, coeffs)

# ===============================
# EJECUTAR QAOA CON RUIDO
# ===============================
def ejecutar_qaoa_variabilidad(num_corridas=10):
    backend = FakeVigo()
    noise_model = NoiseModel.from_backend(backend)

    simulator = AerSimulator(
        noise_model=noise_model,
        basis_gates=noise_model.basis_gates
    )

    sampler = BackendSampler(backend=simulator)

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=120),
        reps=6
    )

    H = crear_hamiltoniano()

    resultados = []

    for i in range(num_corridas):
        print(f"▶ Corrida {i + 1}/{num_corridas}")

        start = time.time()
        result = qaoa.compute_minimum_eigenvalue(H)
        tiempo = time.time() - start

        energia = np.real(result.eigenvalue)
        resultados.append({
            "corrida": i + 1,
            "energia": energia,
            "tiempo_seg": tiempo
        })

        print(f"   Energía = {energia:.4f} | Tiempo = {tiempo:.2f}s")

    df = pd.DataFrame(resultados)
    df.to_csv(f"{OUTPUT_DIR}/qaoa_ruido_10q.csv", index=False)

    # Gráfico
    plt.figure()
    plt.plot(df["corrida"], df["energia"], marker="o")
    plt.xlabel("Corrida")
    plt.ylabel("Energía")
    plt.title("Variabilidad QAOA con ruido (10 qubits)")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/qaoa_ruido_10q.png")
    plt.close()

    print("\n✅ Resultados guardados en /results")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("\n🚀 QAOA con 10 QUBITS y RUIDO CUÁNTICO REALISTA\n")
    ejecutar_qaoa_variabilidad(num_corridas=10)
