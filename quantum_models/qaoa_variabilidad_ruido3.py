# qaoa_variabilidad_ruido3.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.providers.fake_provider import FakeVigo
from qiskit_aer.noise import NoiseModel
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.utils import QuantumInstance
from qiskit.algorithms.minimum_eigen_solvers import QAOA
from qiskit.algorithms.optimizers import COBYLA

print("\n🚀 QAOA VARIABILIDAD RUIDO3 – 8 qubits con ruido cuántico\n")


# Crear un QUBO más difícil (8 qubits)
def crear_problema_optimizacion():
    qp = QuadraticProgram()
    for i in range(8):
        qp.binary_var(name=f"x{i}")

    # Función objetivo con más interacciones
    linear = {f"x{i}": -1 for i in range(8)}
    quadratic = {(f"x{i}", f"x{i + 1}"): -1 for i in range(7)}
    quadratic[("x0", "x7")] = -1  # conexión circular para más complejidad

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


# Ejecutar múltiples corridas QAOA con ruido
def ejecutar_qaoa_variabilidad(num_corridas=10):
    qp = crear_problema_optimizacion()
    qubo = QuadraticProgramToQubo().convert(qp)

    backend = Aer.get_backend("qasm_simulator")
    fake_backend = FakeVigo()
    noise_model = NoiseModel.from_backend(fake_backend)

    quantum_instance = QuantumInstance(backend=backend,
                                       noise_model=noise_model,
                                       shots=2048,
                                       optimization_level=3,
                                       seed_simulator=42,
                                       seed_transpiler=42)

    optimizer = COBYLA(maxiter=100)
    qaoa = QAOA(optimizer=optimizer, reps=3, quantum_instance=quantum_instance)
    solver = MinimumEigenOptimizer(qaoa)

    resultados = []
    for i in range(num_corridas):
        print(f"🔁 Corrida {i + 1}/{num_corridas}")
        start = time.time()
        result = solver.solve(qubo)
        end = time.time()

        x = result.x
        costo = result.fval
        print(f"   ✔ Costo={costo:.3f} | x={x}")

        fila = {
            "corrida": i + 1,
            "costo": costo,
            "tiempo_s": round(end - start, 3),
        }
        for j in range(len(x)):
            fila[f"x{j + 1} (binary)"] = x[j]
        resultados.append(fila)

    df = pd.DataFrame(resultados)
    return df


# Ejecutar y guardar resultados
df = ejecutar_qaoa_variabilidad(num_corridas=10)

# Mostrar tabla
print("\n📊 Resultados:")
print(df)

# Guardar CSV y gráfico
df.to_csv("quantum_models/results/qaoa_variabilidad_ruido3.csv", index=False)

plt.figure(figsize=(10, 4))
plt.plot(df["corrida"], df["costo"], marker='o', linestyle='--', label="Costo")
plt.xlabel("Corrida")
plt.ylabel("Costo")
plt.title("Variabilidad de QAOA con ruido (8 qubits)")
plt.grid(True)
plt.tight_layout()
plt.savefig("quantum_models/results/graficos/qaoa_variabilidad_ruido3.png")
plt.close()

print("\n💾 CSV: quantum_models/results/qaoa_variabilidad_ruido3.csv")
print("📈 Gráfico: quantum_models/results/graficos/qaoa_variabilidad_ruido3.png")
print("\n✅ SIMULACIÓN COMPLETADA")
