# quantum_models/test_ibm_runtime.py
from qiskit_ibm_provider import IBMProvider

# Usamos tu API Key recién creada
IBM_TOKEN = "4dXBB5kD8gQ_BMXD6udNg3V9ugpzXdhJQFramqmU7CPO"

# Conexión con IBM Quantum
provider = IBMProvider(token=IBM_TOKEN)

print("✅ Conectado correctamente a IBM Quantum.")
print("🔍 Backends disponibles:")
for backend in provider.backends():
    print(f" - {backend.name}")
