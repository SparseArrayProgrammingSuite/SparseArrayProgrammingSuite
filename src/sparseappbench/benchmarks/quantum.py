import numpy as np
from ..binsparse_format import BinsparseFormat

def apply_single_qubit_gate(xp, state, gate, qubit, nqubits):
    
    left = 1 << qubit
    right = 1 << (nqubits - qubit - 1)
    start_resh = xp.reshape(state, (left, 2, right))
    # gate[new, old] convention => einsum "ijk,lj->ilk"
    new_resh = xp.einsum(
        "new_resh[i, j, k] += start_resh[i, l, k] * gate[j, l]", 
        start_resh=start_resh, 
        gate=gate)
    return xp.reshape(new_resh, state.shape)

"Simulates a random quantum circuit on an n-qubit state vector, using the standard reshape + einsum gate application pattern"

class QGates:
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    all_gates = [H, X, Y, Z, S, T]
    
def benchmark_rqc_statevector(xp, state_bench, nqubits, num_layers=10):
    
    rng = np.random.default_rng(seed=42)
    single_qubit_gates = QGates.all_gates
    
    #Pre-build all the gates we will need for the circuit
    gates = []
    for q in range(nqubits):
        g_np = rng.choice(single_qubit_gates)
        g_bench = BinsparseFormat.from_numpy(g_np)
        g_xp = xp.from_benchmark(g_bench)
        gates.append(g_xp)
    
    #Load the initial state
    state = xp.from_benchmark(state_bench)
    
    #Single lazy chain: apply each gate to each qubit sequentially. (only one layer)
    state = xp.lazy(state)
    for q in range(nqubits):
        state = apply_single_qubit_gate(xp, state, gates[q], q, nqubits)
    
    #Evaluate the lazy chain
    state = xp.compute(state)
    
    final_state_bench = xp.to_benchmark(state)
    return final_state_bench


#Data gen 

def dg_single_layer_large():
    "Small instance of 10 qubits, 1 layer"
    nqubits = 40
    dim = 1 << nqubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j #|000...0
    state_bin = BinsparseFormat.from_numpy(state)
    return state_bin, nqubits


def dg_single_layer_small():
    "Small instance of 10 qubits, 1 layer"
    nqubits = 10
    dim = 1 << nqubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j #|000...0
    state_bin = BinsparseFormat.from_numpy(state)
    return state_bin, nqubits

def dg_single_layer_tiny():
    "Tiny instance of 5 qubits, 1 layer"
    nqubits = 5
    dim = 1 << nqubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j #|000...0
    state_bin = BinsparseFormat.from_numpy(state)
    return state_bin, nqubits
    
    