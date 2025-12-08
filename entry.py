from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import torch
import numpy as np

# This will act as the Entry point for the Backend module for QtorchX
# Deliverables include: Accumulated Phi Manifold Heat Map, Result for the given Quantum circuit, And some meta data on the 
# execution of the circuit final bitstring, qubit states, probabilities, histogram of states etc
# This will be modular for understanding and each module will be documented, or I will try to haha
# The modules will include Circuit Data Structure hosing the qubits and Gates (qubit_number, time step) classic Cirq way
# Circuit scheduler will take care of scheduling the gates without collisions and parallel measurements etc edge cases if the positions
# of the gates aren't specified by the user
# Then we got the State Vector Backend which will take care of the state vector manipulations, gate applications etc
# Then we have the Phi Manifold Extractor which will take care of extracting the phi manifold heat map from the state vector
# Finally we have the Result Aggregator which will take care of aggregating the results from the state vector and phi manifold extractor
# And yeah this results in the final state will be returned to the user along with the phi manifold heat map and meta data


class GateLibrary:
    """
    Stateless quantum gate library with 40+ gates.
    All gates returned as fresh base 2x2, 4x4, or 8x8 matrices (complex64).
    Backend handles caching and tensor product expansion to full Hilbert space.
    """
    
    @staticmethod
    def _ensure_complex(matrix: np.ndarray) -> torch.Tensor:
        """Convert numpy matrix to torch complex64 tensor"""
        return torch.tensor(matrix, dtype=torch.complex64)
    
    # ========================================================================
    # IDENTITY & PAULI GATES (4 gates)
    # ========================================================================
    
    @staticmethod
    def I() -> torch.Tensor:
        """Identity gate - 2x2"""
        return GateLibrary._ensure_complex([
            [1, 0],
            [0, 1]
        ])
    
    @staticmethod
    def X() -> torch.Tensor:
        """Pauli-X (NOT) gate - 2x2"""
        return GateLibrary._ensure_complex([
            [0, 1],
            [1, 0]
        ])
    
    @staticmethod
    def Y() -> torch.Tensor:
        """Pauli-Y gate - 2x2"""
        return GateLibrary._ensure_complex([
            [0, -1j],
            [1j, 0]
        ])
    
    @staticmethod
    def Z() -> torch.Tensor:
        """Pauli-Z gate - 2x2"""
        return GateLibrary._ensure_complex([
            [1, 0],
            [0, -1]
        ])
    
    # ========================================================================
    # HADAMARD & SQUARE ROOT GATES (4 gates)
    # ========================================================================
    
    @staticmethod
    def H() -> torch.Tensor:
        """Hadamard gate - 2x2"""
        sqrt2 = np.sqrt(2)
        return GateLibrary._ensure_complex([
            [1/sqrt2, 1/sqrt2],
            [1/sqrt2, -1/sqrt2]
        ])
    
    @staticmethod
    def SX() -> torch.Tensor:
        """√X gate (square root of X) - 2x2"""
        return GateLibrary._ensure_complex([
            [0.5+0.5j, 0.5-0.5j],
            [0.5-0.5j, 0.5+0.5j]
        ])
    
    @staticmethod
    def SY() -> torch.Tensor:
        """√Y gate - 2x2"""
        return GateLibrary._ensure_complex([
            [0.5+0.5j, -0.5-0.5j],
            [0.5+0.5j, 0.5+0.5j]
        ])
    
    @staticmethod
    def SZ() -> torch.Tensor:
        """√Z gate (same as S gate) - 2x2"""
        return GateLibrary.S()
    
    # ========================================================================
    # PHASE GATES (5 gates)
    # ========================================================================
    
    @staticmethod
    def S() -> torch.Tensor:
        """S gate (Phase gate, √Z) - 2x2"""
        return GateLibrary._ensure_complex([
            [1, 0],
            [0, 1j]
        ])
    
    @staticmethod
    def SDG() -> torch.Tensor:
        """S† gate (S-dagger, inverse of S) - 2x2"""
        return GateLibrary._ensure_complex([
            [1, 0],
            [0, -1j]
        ])
    
    @staticmethod
    def T() -> torch.Tensor:
        """T gate (π/8 gate) - 2x2"""
        return GateLibrary._ensure_complex([
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ])
    
    @staticmethod
    def TDG() -> torch.Tensor:
        """T† gate (T-dagger) - 2x2"""
        return GateLibrary._ensure_complex([
            [1, 0],
            [0, np.exp(-1j * np.pi / 4)]
        ])
    
    @staticmethod
    def P(theta: float) -> torch.Tensor:
        """Phase gate with arbitrary angle - 2x2
        P(θ) = [[1, 0], [0, e^(iθ)]]
        """
        return GateLibrary._ensure_complex([
            [1, 0],
            [0, np.exp(1j * theta)]
        ])
    
    # ========================================================================
    # ROTATION GATES (6 gates)
    # ========================================================================
    
    @staticmethod
    def RX(theta: float) -> torch.Tensor:
        """Rotation around X-axis - 2x2
        RX(θ) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return GateLibrary._ensure_complex([
            [c, -1j*s],
            [-1j*s, c]
        ])
    
    @staticmethod
    def RY(theta: float) -> torch.Tensor:
        """Rotation around Y-axis - 2x2
        RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return GateLibrary._ensure_complex([
            [c, -s],
            [s, c]
        ])
    
    @staticmethod
    def RZ(theta: float) -> torch.Tensor:
        """Rotation around Z-axis - 2x2
        RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        """
        return GateLibrary._ensure_complex([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ])
    
    @staticmethod
    def U1(lam: float) -> torch.Tensor:
        """Single-parameter U gate (equivalent to P gate) - 2x2"""
        return GateLibrary.P(lam)
    
    @staticmethod
    def U2(phi: float, lam: float) -> torch.Tensor:
        """Two-parameter U gate - 2x2
        U2(φ, λ) = (1/√2) * [[1, -e^(iλ)], [e^(iφ), e^(i(φ+λ))]]
        """
        sqrt2 = np.sqrt(2)
        return GateLibrary._ensure_complex([
            [1/sqrt2, -np.exp(1j*lam)/sqrt2],
            [np.exp(1j*phi)/sqrt2, np.exp(1j*(phi+lam))/sqrt2]
        ])
    
    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> torch.Tensor:
        """Three-parameter universal U gate - 2x2
        U3(θ, φ, λ) is the most general single-qubit gate
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return GateLibrary._ensure_complex([
            [c, -np.exp(1j*lam)*s],
            [np.exp(1j*phi)*s, np.exp(1j*(phi+lam))*c]
        ])
    
    # ========================================================================
    # TWO-QUBIT GATES (10 gates)
    # ========================================================================
    
    @staticmethod
    def CNOT() -> torch.Tensor:
        """Controlled-NOT (CX) gate - 4x4
        Control: first qubit, Target: second qubit
        """
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
    
    @staticmethod
    def CX() -> torch.Tensor:
        """Alias for CNOT"""
        return GateLibrary.CNOT()
    
    @staticmethod
    def CY() -> torch.Tensor:
        """Controlled-Y gate - 4x4"""
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ])
    
    @staticmethod
    def CZ() -> torch.Tensor:
        """Controlled-Z gate - 4x4"""
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ])
    
    @staticmethod
    def SWAP() -> torch.Tensor:
        """SWAP gate - 4x4"""
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def ISWAP() -> torch.Tensor:
        """iSWAP gate - 4x4"""
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def CH() -> torch.Tensor:
        """Controlled-Hadamard gate - 4x4"""
        sqrt2 = np.sqrt(2)
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1/sqrt2, 1/sqrt2],
            [0, 0, 1/sqrt2, -1/sqrt2]
        ])
    
    @staticmethod
    def CRX(theta: float) -> torch.Tensor:
        """Controlled-RX gate - 4x4"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -1j*s],
            [0, 0, -1j*s, c]
        ])
    
    @staticmethod
    def CRY(theta: float) -> torch.Tensor:
        """Controlled-RY gate - 4x4"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ])
    
    @staticmethod
    def CRZ(theta: float) -> torch.Tensor:
        """Controlled-RZ gate - 4x4"""
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j*theta/2), 0],
            [0, 0, 0, np.exp(1j*theta/2)]
        ])
    
    # ========================================================================
    # TWO-QUBIT PARAMETERIZED ROTATIONS (3 gates)
    # ========================================================================
    
    @staticmethod
    def RXX(theta: float) -> torch.Tensor:
        """Two-qubit XX rotation - 4x4
        RXX(θ) = exp(-iθ/2 * X⊗X)
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return GateLibrary._ensure_complex([
            [c, 0, 0, -1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [-1j*s, 0, 0, c]
        ])
    
    @staticmethod
    def RYY(theta: float) -> torch.Tensor:
        """Two-qubit YY rotation - 4x4"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return GateLibrary._ensure_complex([
            [c, 0, 0, 1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [1j*s, 0, 0, c]
        ])
    
    @staticmethod
    def RZZ(theta: float) -> torch.Tensor:
        """Two-qubit ZZ rotation - 4x4"""
        exp_pos = np.exp(1j * theta / 2)
        exp_neg = np.exp(-1j * theta / 2)
        return GateLibrary._ensure_complex([
            [exp_neg, 0, 0, 0],
            [0, exp_pos, 0, 0],
            [0, 0, exp_pos, 0],
            [0, 0, 0, exp_neg]
        ])
    
    # ========================================================================
    # THREE-QUBIT GATES (3 gates)
    # ========================================================================
    
    @staticmethod
    def TOFFOLI() -> torch.Tensor:
        """Toffoli (CCNOT) gate - 8x8
        Double-controlled NOT gate
        """
        mat = np.eye(8, dtype=np.complex64)
        mat[6, 6] = 0
        mat[7, 7] = 0
        mat[6, 7] = 1
        mat[7, 6] = 1
        return GateLibrary._ensure_complex(mat)
    
    @staticmethod
    def CCNOT() -> torch.Tensor:
        """Alias for TOFFOLI"""
        return GateLibrary.TOFFOLI()
    
    @staticmethod
    def FREDKIN() -> torch.Tensor:
        """Fredkin (CSWAP) gate - 8x8
        Controlled-SWAP gate
        """
        mat = np.eye(8, dtype=np.complex64)
        mat[5, 5] = 0
        mat[6, 6] = 0
        mat[5, 6] = 1
        mat[6, 5] = 1
        return GateLibrary._ensure_complex(mat)
    
    @staticmethod
    def CSWAP() -> torch.Tensor:
        """Alias for FREDKIN"""
        return GateLibrary.FREDKIN()
    
    # ========================================================================
    # EXOTIC/SPECIAL GATES (5 gates)
    # ========================================================================
    
    @staticmethod
    def V() -> torch.Tensor:
        """V gate (√X variant) - 2x2"""
        return GateLibrary._ensure_complex([
            [0.5+0.5j, 0.5-0.5j],
            [0.5-0.5j, 0.5+0.5j]
        ])
    
    @staticmethod
    def VDG() -> torch.Tensor:
        """V† gate - 2x2"""
        return GateLibrary._ensure_complex([
            [0.5-0.5j, 0.5+0.5j],
            [0.5+0.5j, 0.5-0.5j]
        ])
    
    @staticmethod
    def SQRT_SWAP() -> torch.Tensor:
        """√SWAP gate - 4x4"""
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 0.5+0.5j, 0.5-0.5j, 0],
            [0, 0.5-0.5j, 0.5+0.5j, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def ECR() -> torch.Tensor:
        """ECR (Echoed Cross-Resonance) gate - 4x4
        Native gate on IBM hardware
        """
        sqrt2 = np.sqrt(2)
        return GateLibrary._ensure_complex([
            [0, 0, 1/sqrt2, 1j/sqrt2],
            [0, 0, 1j/sqrt2, 1/sqrt2],
            [1/sqrt2, -1j/sqrt2, 0, 0],
            [-1j/sqrt2, 1/sqrt2, 0, 0]
        ])
    
    @staticmethod
    def DCX() -> torch.Tensor:
        """Double-CNOT gate - 4x4
        Equivalent to two CNOTs with reversed control/target
        """
        return GateLibrary._ensure_complex([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
    
    # ========================================================================
    # MAIN DISPATCHER
    # ========================================================================
    
    @staticmethod
    def get_gate(name: str, params: Optional[List[float]] = None) -> torch.Tensor:
        """
        Main entry point to get gate matrix by name.
        
        Args:
            name: Gate name (case-insensitive)
            params: Parameters for parameterized gates
            
        Returns:
            torch.Tensor: Gate matrix (2x2, 4x4, or 8x8)
            
        Raises:
            ValueError: If gate name is unknown or params are invalid
        """
        name = name.upper()
        params = params or []
        
        # Non-parameterized gates
        if name in ['I', 'X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG',
                    'SX', 'SY', 'SZ', 'V', 'VDG',
                    'CNOT', 'CX', 'CY', 'CZ', 'SWAP', 'ISWAP', 'CH',
                    'TOFFOLI', 'CCNOT', 'FREDKIN', 'CSWAP',
                    'SQRT_SWAP', 'ECR', 'DCX']:
            return getattr(GateLibrary, name)()
        
        # Parameterized gates - require params
        if name in ['P', 'U1']:
            if len(params) != 1:
                raise ValueError(f"{name} requires 1 parameter, got {len(params)}")
            return GateLibrary.P(params[0])
        
        if name in ['RX', 'RY', 'RZ']:
            if len(params) != 1:
                raise ValueError(f"{name} requires 1 parameter, got {len(params)}")
            return getattr(GateLibrary, name)(params[0])
        
        if name == 'U2':
            if len(params) != 2:
                raise ValueError(f"U2 requires 2 parameters, got {len(params)}")
            return GateLibrary.U2(params[0], params[1])
        
        if name == 'U3':
            if len(params) != 3:
                raise ValueError(f"U3 requires 3 parameters, got {len(params)}")
            return GateLibrary.U3(params[0], params[1], params[2])
        
        if name in ['CRX', 'CRY', 'CRZ', 'RXX', 'RYY', 'RZZ']:
            if len(params) != 1:
                raise ValueError(f"{name} requires 1 parameter, got {len(params)}")
            return getattr(GateLibrary, name)(params[0])
        
        raise ValueError(f"Unknown gate: {name}")
    
    @staticmethod
    def list_gates() -> Dict[str, int]:
        """Return all available gates with their dimensionality"""
        return {
            # Single-qubit (2x2)
            'I': 2, 'X': 2, 'Y': 2, 'Z': 2, 'H': 2,
            'S': 2, 'SDG': 2, 'T': 2, 'TDG': 2,
            'SX': 2, 'SY': 2, 'SZ': 2, 'V': 2, 'VDG': 2,
            'RX': 2, 'RY': 2, 'RZ': 2, 'P': 2,
            'U1': 2, 'U2': 2, 'U3': 2,
            # Two-qubit (4x4)
            'CNOT': 4, 'CX': 4, 'CY': 4, 'CZ': 4,
            'SWAP': 4, 'ISWAP': 4, 'SQRT_SWAP': 4,
            'CH': 4, 'CRX': 4, 'CRY': 4, 'CRZ': 4,
            'RXX': 4, 'RYY': 4, 'RZZ': 4,
            'ECR': 4, 'DCX': 4,
            # Three-qubit (8x8)
            'TOFFOLI': 8, 'CCNOT': 8, 'FREDKIN': 8, 'CSWAP': 8
        }

@dataclass
class Gate:
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    t: Optional[int] = None
    depends_on: Optional[List['Gate']] = None
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.qubits:
            raise ValueError(f"Gate {self.name} requires at least one qubit")
        if len(self.qubits) != len(set(self.qubits)):
            raise ValueError(f"Duplicate qubits in {self.name}: {self.qubits}")
        
class Circuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        if self.num_qubits > 24:
            raise ValueError(f"Circuit supports up to 24 qubits but given {self.num_qubits}")
        self.grid = [[] for _ in range(num_qubits)]
        self.label_counts = {}
        self.gates = []  # Keep ordered list for iteration
    
    def _ensure(self, q: int, t: int):
        """Extend grid row to include time step t"""
        while len(self.grid[q]) <= t:
            self.grid[q].append(None)
    
    def _assign_label(self, gate: Gate):
        """Generate unique label for gate based on name and qubits"""
        qubit_digits = "".join(str(q) for q in gate.qubits)
        key = (gate.name, qubit_digits)
        n = self.label_counts.get(key, 0)
        self.label_counts[key] = n + 1
        gate.label = f"Gate{gate.name}{qubit_digits}#{n}"
    
    def add(self, gate: Gate) -> int:
        """
        Add gate to circuit with automatic or manual scheduling.
        
        Features:
        - Manual placement if gate.t is set (with conflict checking)
        - Auto-scheduling finds earliest available slot
        - Handles multi-qubit gates spanning non-adjacent qubits
        - Respects dependency constraints via gate.depends_on
        
        Args:
            gate: Gate to add to circuit
            
        Returns:
            t: Time step where gate was placed
            
        Raises:
            ValueError: If manually placed gate conflicts with existing gates
        """
        qubits = gate.qubits
        
        # Validate qubit indices
        if any(q < 0 or q >= self.num_qubits for q in qubits):
            raise ValueError(
                f"Gate {gate.name} uses invalid qubits {qubits}. "
                f"Valid range: 0-{self.num_qubits - 1}"
            )
        
        self._assign_label(gate)
        
        # Manual placement
        if gate.t is not None:
            t = gate.t
            for q in qubits:
                self._ensure(q, t)
                if self.grid[q][t] is not None:
                    conflicting_gate = self.grid[q][t]
                    raise ValueError(
                        f"Qubit {q} busy at t={t} with {conflicting_gate.label}. "
                        f"Cannot place {gate.label}"
                    )
            # Place gate
            for q in qubits:
                self.grid[q][t] = gate
            self.gates.append(gate)
            return t
        
        # Auto-scheduling: find earliest valid time step
        
        # Start from latest occupied time across target qubits
        last = max((len(self.grid[q]) - 1 for q in qubits), default=-1)
        
        # For multi-qubit gates, block ALL qubits in span (handles CNOT, SWAP, etc.)
        # This prevents threading gates through the "wire" connecting control/target
        top, bot = min(qubits), max(qubits)
        for q in range(top, bot + 1):
            last = max(last, len(self.grid[q]) - 1)
        
        # Respect explicit dependencies
        for parent in (gate.depends_on or []):
            if parent.t is not None:
                last = max(last, parent.t)
        
        # Find first available slot starting from last + 1
        t = last + 1
        while True:
            # Ensure all target qubits have entries at t
            for q in qubits:
                self._ensure(q, t)
            
            # Check for conflicts
            if any(self.grid[q][t] is not None for q in qubits):
                t += 1
                continue
            
            # Check span blocking (for multi-qubit gates)
            if len(qubits) > 1:
                conflict = False
                for q in range(top, bot + 1):
                    self._ensure(q, t)
                    if self.grid[q][t] is not None:
                        # Allow if it's a single-qubit gate on a non-target qubit in span
                        existing = self.grid[q][t]
                        if q not in qubits and len(existing.qubits) == 1:
                            continue  # Safe to have single-qubit gate on "wire"
                        conflict = True
                        break
                
                if conflict:
                    t += 1
                    continue
            
            # Found valid slot
            break
        
        gate.t = t
        for q in qubits:
            self.grid[q][t] = gate
        
        self.gates.append(gate)
        return t
    
    @property
    def depth(self) -> int:
        """Circuit depth (max time steps across all qubits)"""
        return max((len(row) for row in self.grid), default=0)
    
    @property
    def size(self) -> int:
        """Total number of gates"""
        return len(self.gates)
    
    def visualize(self) -> str:
        """ASCII visualization of circuit grid"""
        lines = []
        max_t = self.depth
        
        for q in range(self.num_qubits):
            line = f"q{q}: |0⟩─"
            for t in range(max_t):
                if t < len(self.grid[q]) and self.grid[q][t] is not None:
                    gate = self.grid[q][t]
                    # Show gate only on first qubit it acts on
                    if q == min(gate.qubits):
                        params_str = f"({gate.params[0]:.2f})" if gate.params else ""
                        line += f"[{gate.name}{params_str}]─"
                    else:
                        line += "──●──" if q != min(gate.qubits) else "──■──"
                else:
                    line += "─────"
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_time_slice(self, t: int) -> List[Gate]:
        """Get all unique gates at time step t"""
        seen_ids = set()
        gates_at_t = []
        
        for q in range(self.num_qubits):
            if t < len(self.grid[q]) and self.grid[q][t] is not None:
                gate = self.grid[q][t]
                # Use id() to track uniqueness instead of hash
                if id(gate) not in seen_ids:
                    seen_ids.add(id(gate))
                    gates_at_t.append(gate)
        
        return gates_at_t

    
    def __repr__(self) -> str:
        return f"Circuit(qubits={self.num_qubits}, gates={self.size}, depth={self.depth})"
