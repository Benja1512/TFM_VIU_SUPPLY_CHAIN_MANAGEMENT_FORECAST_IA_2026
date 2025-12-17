import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Rutas y validaciones
# ===============================
results_dir = "quantum_models/results"
baseline_file = os.path.join(results_dir, "baseline_output.csv")
qaoa_file = os.path.join(results_dir, "qaoa_output.csv")
final_csv = os.path.join(results_dir, "qaoa_vs_classic.csv")
final_plot = os.path.join(results_dir, "qaoa_vs_classic.png")

# Verificar existencia de archivos
if not os.path.exists(baseline_file) or not os.path.exists(qaoa_file):
    raise FileNotFoundError("Faltan uno o ambos archivos: baseline_output.csv o qaoa_output.csv")

# ===============================
# Leer y combinar resultados
# ===============================
df_baseline = pd.read_csv(baseline_file)
df_qaoa = pd.read_csv(qaoa_file)
df = pd.concat([df_baseline, df_qaoa], ignore_index=True)

# Ordenar por nombre de método para consistencia
df = df.sort_values(by="metodo")  # Clásico antes que QAOA

# Guardar CSV combinado
df.to_csv(final_csv, index=False)

# ===============================
# Gráfico comparativo con etiquetas
# ===============================
plt.figure(figsize=(6, 4))
bars = plt.bar(df["metodo"], df["costo"], color=["steelblue", "darkorange"])
plt.title("Comparación de costos: Clásico vs QAOA")
plt.ylabel("Costo total")

# Agregar etiquetas de valor encima de las barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom')

# Guardar gráfico
plt.tight_layout()
plt.savefig(final_plot)
plt.close()

# ===============================
# Confirmación
# ===============================
print(f"\n✅ Comparación completada.\n📁 CSV: {final_csv}\n📊 Gráfico: {final_plot}")
