from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from entry import Gate, Circuit
def build_teleportation_circuit() -> Circuit:
    """
    Builds a quantum teleportation circuit.
    
    Protocol:
    1. Prepare state |ψ⟩ on q0 (we'll use |+⟩ = H|0⟩)
    2. Create Bell pair between q1 (Alice) and q2 (Bob)
    3. Alice entangles q0 with q1 (CNOT + H)
    4. Alice measures q0 and q1
    5. Bob applies corrections based on measurement results
    
    Qubits:
    - q0: Message qubit (state to teleport)
    - q1: Alice's entangled qubit
    - q2: Bob's entangled qubit (receives teleported state)
    """
    circuit = Circuit(num_qubits=3)
    
    # Step 1: Prepare message state |ψ⟩ = |+⟩ on q0
    prepare_h = Gate("H", [0])
    circuit.add(prepare_h)
    print(f"✓ Prepared |+⟩ state on q0 at t={prepare_h.t}")
    
    # Step 2: Create Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2 on q1-q2
    bell_h = Gate("H", [1])
    bell_cnot = Gate("CNOT", [1, 2])
    circuit.add(bell_h)
    circuit.add(bell_cnot)
    print(f"✓ Created Bell pair on q1-q2 at t={bell_h.t}, {bell_cnot.t}")
    
    # Step 3: Alice's entangling operations
    alice_cnot = Gate("CNOT", [0, 1])
    alice_h = Gate("H", [0])
    circuit.add(alice_cnot)
    circuit.add(alice_h)
    print(f"✓ Alice's entanglement at t={alice_cnot.t}, {alice_h.t}")
    
    # Step 4: Alice's measurements
    measure_0 = Gate("M", [0])
    measure_1 = Gate("M", [1])
    circuit.add(measure_0)
    circuit.add(measure_1)
    print(f"✓ Alice measures at t={measure_0.t}, {measure_1.t}")
    
    # Step 5: Bob's conditional corrections (classical control)
    # X correction if q1 measured |1⟩
    bob_x = Gate("X", [2], metadata={"classical_control": "q1"})
    bob_x.depends_on = [measure_1]
    
    # Z correction if q0 measured |1⟩
    bob_z = Gate("Z", [2], metadata={"classical_control": "q0"})
    bob_z.depends_on = [measure_0]
    
    circuit.add(bob_x)
    circuit.add(bob_z)
    print(f"✓ Bob's corrections at t={bob_x.t}, {bob_z.t}")
    
    return circuit


def build_ghz_state(num_qubits: int = 4) -> Circuit:
    """
    Build GHZ state |GHZ⟩ = (|000...0⟩ + |111...1⟩)/√2
    Great for testing phi manifold on maximally entangled states!
    """
    circuit = Circuit(num_qubits=num_qubits)
    
    # Hadamard on first qubit
    h = Gate("H", [0])
    circuit.add(h)
    
    # CNOT cascade
    for i in range(num_qubits - 1):
        cnot = Gate("CNOT", [i, i + 1])
        circuit.add(cnot)
    
    return circuit


def build_vqe_ansatz(num_qubits: int = 4, depth: int = 2) -> Circuit:
    """
    Build a hardware-efficient VQE ansatz.
    Perfect for phi manifold visualization of parameterized circuits!
    """
    circuit = Circuit(num_qubits=num_qubits)
    
    import numpy as np
    
    for layer in range(depth):
        # Layer of single-qubit rotations
        for q in range(num_qubits):
            theta = np.random.uniform(0, 2*np.pi)
            ry = Gate("RY", [q], params=[theta])
            circuit.add(ry)
        
        # Layer of entangling gates
        for q in range(num_qubits - 1):
            cnot = Gate("CNOT", [q, q + 1])
            circuit.add(cnot)
        
        # Ring closure
        if num_qubits > 2:
            ring_cnot = Gate("CNOT", [num_qubits - 1, 0])
            circuit.add(ring_cnot)
    
    return circuit


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM TELEPORTATION CIRCUIT")
    print("=" * 60)
    
    teleport = build_teleportation_circuit()
    
    print(f"\n{teleport}")
    print(f"\nCircuit Depth: {teleport.depth}")
    print(f"Total Gates: {teleport.size}")
    
    print("\n" + "=" * 60)
    print("TIME STEP BREAKDOWN")
    print("=" * 60)
    for t in range(teleport.depth):
        gates = teleport.get_time_slice(t)
        if gates:
            gate_info = [f"{g.name}({g.qubits})" for g in gates]
            print(f"t={t}: {', '.join(gate_info)}")
    
    print("\n" + "=" * 60)
    print("CIRCUIT DIAGRAM")
    print("=" * 60)
    print(teleport.visualize())
    
    print("\n" + "=" * 60)
    print("GRID STATE (for debugging)")
    print("=" * 60)
    for q in range(teleport.num_qubits):
        row = [g.name if g else "─" for g in teleport.grid[q]]
        print(f"q{q}: {row}")
    
    # Bonus circuits for hackathon demo
    print("\n\n" + "=" * 60)
    print("BONUS: GHZ STATE (4 qubits)")
    print("=" * 60)
    ghz = build_ghz_state(4)
    print(ghz)
    print(ghz.visualize())
    
    print("\n\n" + "=" * 60)
    print("BONUS: VQE ANSATZ (4 qubits, depth=2)")
    print("=" * 60)
    vqe = build_vqe_ansatz(4, 2)
    print(vqe)
    print(vqe.visualize())
    print(f"This parameterized circuit will show CRAZY phi manifold evolution!")
