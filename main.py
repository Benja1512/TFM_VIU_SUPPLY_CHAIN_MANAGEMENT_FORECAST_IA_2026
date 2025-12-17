from exploratory.eda_demand_plot import run_eda
from machine_learning.random_forest import run_random_forest_forecast

if __name__ == "__main__":
    print("¿Qué quieres ejecutar?")
    print("1. Exploración de Datos (EDA)")
    print("2. Random Forest")

    opcion = input("Escribe 1 o 2: ")

    if opcion == "1":
        run_eda()
    elif opcion == "2":
        run_random_forest_forecast()
    else:
        print("Opción inválida.")
