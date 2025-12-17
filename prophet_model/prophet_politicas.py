import pandas as pd
import os
import matplotlib.pyplot as plt

# =====================================================
# 1. CARGAR ESCENARIOS PROPHET DESDE CSV
# =====================================================

esc_path = "results/escenarios/escenarios_prophet.csv"  # Ruta corregida

if not os.path.exists(esc_path):
    raise FileNotFoundError(f"❌ ERROR: No se encontró el archivo de escenarios en: {esc_path}")

df = pd.read_csv(esc_path)

# Usamos SOLO las últimas 12 semanas (las proyectadas)
esc = df.tail(12).reset_index(drop=True)

# =====================================================
# 2. FUNCIÓN CORREGIDA (YA NO USA SERIES)
# =====================================================

def calcular_indicadores(demanda, politica, stock_seguridad=100):
    inventario = stock_seguridad
    inventario_final = []
    sobrestock_total = 0
    stockout_total = 0

    # ⚠️ convertir a listas (evita Series)
    demanda = demanda.tolist()
    politica = politica.tolist()

    for pol, dem in zip(politica, demanda):
        diferencia = pol - dem
        inventario += diferencia

        # Stockout
        if inventario < 0:
            stockout_total += abs(inventario)
            inventario = 0

        # Sobrestock
        if inventario > stock_seguridad:
            sobrestock_total += (inventario - stock_seguridad)

        inventario_final.append(inventario)

    inventario_promedio = sum(inventario_final) / len(inventario_final)

    return inventario_promedio, sobrestock_total, stockout_total


# =====================================================
# 3. CALCULAR POLÍTICAS
# =====================================================

politicas = {
    "A": esc["escenario_bajo_20"],
    "B": esc["base_prophet"],
    "C": esc["escenario_alto_20"]
}

nombres = {
    "A": "Conservadora (–20 %)",
    "B": "Equilibrada (base)",
    "C": "Agresiva (+20 %)"
}

resultados = []

for clave, serie in politicas.items():
    inv, sobre, falt = calcular_indicadores(
        demanda=esc["base_prophet"],
        politica=serie
    )
    resultados.append({
        "Política": clave,
        "Tipo de política": nombres[clave],
        "Inventario final medio (u)": round(inv, 1),
        "Sobrestock total (u)": round(sobre, 1),
        "Stockout total (u)": round(falt, 1)
    })

# =====================================================
# 4. MOSTRAR RESULTADOS
# =====================================================

df_resultados = pd.DataFrame(resultados)
print("\n📊 Indicadores logísticos comparativos:\n")
print(df_resultados.to_string(index=False))

# =====================================================
# 5. GUARDAR CSV
# =====================================================

os.makedirs("results/politicas", exist_ok=True)
output_path = "results/politicas/politicas_prophet.csv"
df_resultados.to_csv(output_path, index=False)
print(f"\n📁 CSV guardado en: {output_path}")
