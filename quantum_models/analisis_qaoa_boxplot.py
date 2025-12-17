import numpy as np
import matplotlib.pyplot as plt

energias = np.array([
    -5.6664,
    -4.8336,
    -4.6797,
    -3.6836,
    -3.1602,
    -3.7637,
    -5.3313,
    -4.0156,
    -4.3270,
    -4.3437
])

media = np.mean(energias)
desviacion = np.std(energias)

print(f"Media de energía: {media:.4f}")
print(f"Desviación estándar: {desviacion:.4f}")

# Guardar estadísticas
with open("quantum_models/results/estadisticas_qaoa_10q.txt", "w") as f:
    f.write(f"Media de energía: {media:.4f}\n")
    f.write(f"Desviación estándar: {desviacion:.4f}\n")

# Boxplot
plt.figure()
plt.boxplot(energias, vert=False)
plt.xlabel("Energía")
plt.title("Variabilidad de QAOA con ruido cuántico (10 qubits)")
plt.grid(True)

# Guardar figura
plt.savefig("quantum_models/results/boxplot_qaoa_10q.png", dpi=300)
plt.show()
