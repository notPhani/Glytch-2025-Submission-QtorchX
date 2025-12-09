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
    BURST_WEIGHTS = {
        # Single-qubit gates - very low error (~0.01-0.1%)
        'I': 0.05,      # Identity/idle - just decoherence
        'X': 0.4,       # Pauli gates - fast, low error
        'Y': 0.4,
        'Z': 0.3,       # Z is phase-only, very fast
        'H': 0.5,       # Hadamard - moderate
        'S': 0.3,       # Phase gates - very fast
        'SDG': 0.3,
        'T': 0.35,      # π/8 gate
        'TDG': 0.35,
        'SX': 0.45,     # Square root gates
        'SY': 0.45,
        'SZ': 0.3,
        'V': 0.4,
        'VDG': 0.4,
        'P': 0.35,      # Parameterized phase
        'U1': 0.35,
        'U2': 0.55,     # More complex single-qubit
        'U3': 0.6,      # Most general single-qubit
        'RX': 0.5,      # Rotation gates - moderate
        'RY': 0.5,
        'RZ': 0.4,      # RZ is phase-only, faster
        
        # Two-qubit gates - much higher error (~0.3-1%)
        'CNOT': 2.5,    # Standard entangling gate - high burst
        'CX': 2.5,      # Alias
        'CY': 2.7,      # Slightly worse than CNOT
        'CZ': 2.3,      # CZ often cleaner than CNOT
        'SWAP': 3.0,    # SWAP = 3 CNOTs, but optimized
        'ISWAP': 2.6,   # Native on some hardware
        'SQRT_SWAP': 2.4,
        'CH': 2.8,      # Controlled-Hadamard
        'CRX': 2.9,     # Controlled rotations
        'CRY': 2.9,
        'CRZ': 2.6,     # CRZ cleaner (phase-only target)
        'RXX': 2.8,     # Two-qubit rotations
        'RYY': 2.8,
        'RZZ': 2.5,     # ZZ coupling often native
        'ECR': 2.4,     # Native IBM gate - optimized
        'DCX': 3.2,     # Double CNOT - expensive
        
        # Three-qubit gates - experimental, very high error (~5-10%)
        'TOFFOLI': 8.0,   # Decomposed into ~6-7 two-qubit gates
        'CCNOT': 8.0,     # Alias
        'FREDKIN': 9.0,   # Even worse - complex decomposition
        'CSWAP': 9.0,     # Alias
    }
    
    @staticmethod
    def _get_burst_weight(gate_name: str, num_qubits: int = 1) -> float:
        """
        Get burst weight for gate based on hardware calibration.
        
        Args:
            gate_name: Gate identifier
            num_qubits: Number of qubits gate acts on
            
        Returns:
            Burst weight (higher = more disturbance)
        """
        gate_name = gate_name.upper()
        
        # Lookup from database
        if gate_name in GateLibrary.BURST_WEIGHTS:
            return GateLibrary.BURST_WEIGHTS[gate_name]
        
        # Fallback heuristic based on qubit count
        if num_qubits == 1:
            return 0.5  # Default single-qubit
        elif num_qubits == 2:
            return 2.5  # Default two-qubit
        elif num_qubits >= 3:
            return 8.0  # Default three-qubit
        else:
            return 1.0  # Unknown
    
    
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
    def get_gate_with_metadata(
        name: str, 
        qubits: List[int],
        params: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Get gate matrix AND metadata including burst weight.
        
        Args:
            name: Gate name
            qubits: Qubits gate acts on
            params: Gate parameters
            
        Returns:
            Dict with 'matrix' and 'metadata' keys
        """
        # Get base matrix
        matrix = GateLibrary.get_gate(name, params)
        
        # Get burst weight
        burst_weight = GateLibrary._get_burst_weight(name, len(qubits))
        
        # Build metadata
        metadata = {
            'burst_weight': burst_weight,
            'num_qubits': len(qubits),
            'gate_type': 'parameterized' if params else 'static',
            'matrix_dim': matrix.shape[0]
        }
        
        return {
            'matrix': matrix,
            'metadata': metadata
        }
    
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
    @staticmethod
    def list_gates_with_burst() -> Dict[str, Dict[str, Any]]:
        """Return all gates with burst weights and metadata"""
        gates = {}
        for name, dim in GateLibrary.list_gates().items():
            num_qubits = {2: 1, 4: 2, 8: 3}.get(dim, 1)
            gates[name] = {
                'matrix_dim': dim,
                'num_qubits': num_qubits,
                'burst_weight': GateLibrary._get_burst_weight(name, num_qubits),
                'error_class': 'low' if num_qubits == 1 else 'high' if num_qubits == 2 else 'extreme'
            }
        return gates

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
        
        # Auto-populate burst_weight if not provided
        if 'burst_weight' not in self.metadata:
            self.metadata['burst_weight'] = GateLibrary._get_burst_weight(
                self.name, 
                len(self.qubits)
            )
    
    def get_burst_weight(self) -> float:
        """Get burst weight (from metadata or default)"""
        return self.metadata.get('burst_weight', 1.0)

class Circuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        if self.num_qubits > 24:
            raise ValueError(f"Circuit supports up to 24 qubits but given {self.num_qubits}")
        self.grid = [[] for _ in range(num_qubits)]
        self.label_counts = {}
        self.gates = []  # Keep ordered list for iteration
        self.metadata = {}
        
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

class PhiManifoldExtractor:
    """
    Extracts 6-channel phi manifold from quantum circuit execution.
    
    Output shape: (6, num_qubits, max_depth)
    
    Channels:
        [0] Memory: (α-λ)φ_i(t)
        [1] Spatial Diffusion: β[Lφ(t)]_i
        [2] Disturbance Diffusion: κ[LD(t)]_i
        [3] Nonlocal Bleed: ε Σ_j exp(-γd_ij)φ_j(t)
        [4] Nonlinear Saturation: ρ φ_i(t)/(1+φ_i²(t))
        [5] Stochastic Kicks: σ_i(t)(G_i(t) + M_i(t))η_i(t)
    """
    
    def __init__(
        self, 
        circuit: Circuit, 
        DecoherenceProjectionMatrix: torch.Tensor, 
        BaselinePauliOffset: torch.Tensor, 
        alpha: float = 0.9, 
        lam: float = 0.05, 
        beta: float = 0.15, 
        kappa: float = 0.1, 
        epsilon: float = 0.002, 
        gamma: float = 1.0, 
        rho: float = 0.08, 
        sigma: float = 0.05, 
        a: float = 1.0,  # Gate disturbance amplification
        b: float = 2.0,  # Measurement disturbance amplification (typically 2x gates)
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.circuit = circuit
        self.DecoherenceProjectionMatrix = DecoherenceProjectionMatrix.to(device)
        self.BaselinePauliOffset = BaselinePauliOffset.to(device)
        
        self.num_qubits = circuit.num_qubits
        self.max_time = circuit.depth
        self.device = device
        
        # Hyperparameters
        self.alpha = alpha
        self.lam = lam
        self.beta = beta
        self.kappa = kappa
        self.epsilon = epsilon
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.a = a  # Gate burst amplification
        self.b = b  # Measurement burst amplification
        
        # Storage for phi manifold (6, num_qubits, max_time)
        self.PhiManifold = torch.zeros(
            (6, self.num_qubits, self.max_time),
            dtype=torch.float32,
            device=device
        )

        self.PauliChannelField = torch.zeros(
            (3, self.num_qubits, self.max_time),
            dtype = torch.float32,
            device = device
        )
        
        # Precomputed graph structures (lazy initialization)
        self._laplacian: Optional[torch.Tensor] = None
        self._distance_matrix: Optional[torch.Tensor] = None
        self._adjacency: Optional[torch.Tensor] = None
    
    # ========================================================================
    # GRAPH STRUCTURE METHODS
    # ========================================================================
    
    def _get_laplacian(self) -> torch.Tensor:
        """
        Build graph Laplacian L = D - W from circuit connectivity.
        Edges exist between qubits connected by multi-qubit gates.
        
        Returns:
            Laplacian matrix (num_qubits, num_qubits)
        """
        if self._laplacian is not None:
            return self._laplacian
        
        n = self.num_qubits
        W = torch.zeros((n, n), device=self.device)
        
        # Build adjacency from multi-qubit gates
        for gate in self.circuit.gates:
            if len(gate.qubits) >= 2:
                # Create edges for all pairs in multi-qubit gate
                for i in range(len(gate.qubits)):
                    for j in range(i + 1, len(gate.qubits)):
                        q1, q2 = gate.qubits[i], gate.qubits[j]
                        W[q1, q2] = 1.0
                        W[q2, q1] = 1.0
        
        # Degree matrix
        D = torch.diag(W.sum(dim=1))
        
        # Laplacian L = D - W
        self._laplacian = D - W
        self._adjacency = W
        
        return self._laplacian
    
    def _get_distance_matrix(self) -> torch.Tensor:
        """
        Compute all-pairs shortest path distances using Floyd-Warshall.
        Distance is measured in number of hops on circuit graph.
        
        Returns:
            Distance matrix (num_qubits, num_qubits)
        """
        if self._distance_matrix is not None:
            return self._distance_matrix
        
        n = self.num_qubits
        
        # Initialize with infinity (unreachable)
        dist = torch.full((n, n), float('inf'), device=self.device)
        
        # Self-loops (distance 0)
        for i in range(n):
            dist[i, i] = 0.0
        
        # Add edges from circuit (distance 1 for neighbors)
        for gate in self.circuit.gates:
            if len(gate.qubits) >= 2:
                for i in range(len(gate.qubits)):
                    for j in range(i + 1, len(gate.qubits)):
                        q1, q2 = gate.qubits[i], gate.qubits[j]
                        dist[q1, q2] = 1.0
                        dist[q2, q1] = 1.0
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        # Replace unreachable (inf) with large distance
        dist[dist == float('inf')] = 10.0
        
        self._distance_matrix = dist
        return self._distance_matrix
    
    def _get_disturbance_field(self, time_step: int) -> torch.Tensor:
        """
        Compute disturbance field D_i(t) = a*G_i(t)*w_gate + b*M_i(t)*w_meas
        
        Where w_gate and w_meas are hardware-calibrated burst weights.
        
        Args:
            time_step: Current time step
            
        Returns:
            Disturbance vector (num_qubits,)
        """
        D = torch.zeros(self.num_qubits, device=self.device)
        
        gates_at_t = self.circuit.get_time_slice(time_step)
        
        for gate in gates_at_t:
            # Get hardware-calibrated burst weight
            burst = gate.get_burst_weight()
            
            for q in gate.qubits:
                if gate.name.upper() == 'M':
                    # Measurement: amplified by factor b
                    D[q] += self.b * burst
                else:
                    # Gate: amplified by factor a
                    D[q] += self.a * burst
        
        return D
    
    # ========================================================================
    # FEATURE COMPUTATION METHODS
    # ========================================================================
    
    def _compute_memory_term(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [0]: Memory term (α - λ)φ_i(t)
        
        Implements non-Markovian persistence with decay.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Memory contribution (num_qubits,)
        """
        return (self.alpha - self.lam) * phi_prev
    
    def _compute_spatial_diffusion(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [1]: Spatial diffusion β[Lφ(t)]_i
        
        Noise spreads along circuit topology via graph Laplacian.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Diffusion contribution (num_qubits,)
        """
        L = self._get_laplacian()
        return self.beta * torch.matmul(L, phi_prev)
    
    def _compute_disturbance_diffusion(self, time_step: int) -> torch.Tensor:
        """
        Channel [2]: Disturbance diffusion κ[LD(t)]_i
        
        Gate/measurement disturbances propagate along circuit graph.
        
        Args:
            time_step: Current time step
            
        Returns:
            Disturbance contribution (num_qubits,)
        """
        L = self._get_laplacian()
        D_t = self._get_disturbance_field(time_step)
        return self.kappa * torch.matmul(L, D_t)
    
    def _compute_nonlocal_bleed(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [3]: Nonlocal exponential bleed ε Σ_j exp(-γd_ij)φ_j(t)
        
        Long-range coupling with exponential distance decay.
        Creates smooth gradients in manifold.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Nonlocal contribution (num_qubits,)
        """
        dist_matrix = self._get_distance_matrix()
        
        # Compute exponential decay matrix exp(-γ*d_ij)
        decay_matrix = torch.exp(-self.gamma * dist_matrix)
        
        # Zero out diagonal (no self-interaction)
        decay_matrix.fill_diagonal_(0.0)
        
        # Weighted sum over neighbors
        return self.epsilon * torch.matmul(decay_matrix, phi_prev)
    
    def _compute_nonlinear_saturation(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [4]: Nonlinear saturation ρ φ_i(t)/(1 + φ_i²(t))
        
        Prevents runaway growth via soft saturation function.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Nonlinear contribution (num_qubits,)
        """
        return self.rho * phi_prev / (1.0 + phi_prev**2)
    
    def _compute_stochastic_kicks(self, time_step: int) -> torch.Tensor:
        """
        Channel [5]: Stochastic kicks σ(G_i(t) + M_i(t))η_i(t)
        
        Activity-modulated Gaussian noise. Idle qubits have minimal noise,
        active qubits (gates/measurements) have proportional noise.
        
        Args:
            time_step: Current time step
            
        Returns:
            Stochastic contribution (num_qubits,)
        """
        # Get disturbance field (activity indicator)
        D_t = self._get_disturbance_field(time_step)
        
        # Gaussian white noise
        eta = torch.randn(self.num_qubits, device=self.device)
        
        # Activity-modulated noise
        return self.sigma * D_t * eta
    
    # ========================================================================
    # MAIN EXTRACTION METHOD
    # ========================================================================
    
    def GetManifold(self) -> torch.Tensor:
        """
        Extract complete phi manifold by simulating coupled dynamics.
        
        Simulates the equation:
        φ_i(t+1) = tanh[(α-λ)φ_i(t) + β[Lφ(t)]_i + κ[LD(t)]_i 
                        + ε Σ_j exp(-γd_ij)φ_j(t) + ρH(φ_i(t)) 
                        + σ(G_i(t) + M_i(t))η_i(t)]
        
        The tanh ensures phi stays bounded in [-1, 1] range.
        
        Returns:
            PhiManifold: Tensor of shape (6, num_qubits, max_time)
        """
        # Initialize phi(0) with small random noise
        phi = torch.randn(self.num_qubits, device=self.device) * 0.01
        
        # Time evolution loop
        for t in range(self.max_time):
            # Compute all 6 feature channels independently
            
            # [0] Memory
            memory = self._compute_memory_term(phi)
            self.PhiManifold[0, :, t] = memory
            
            # [1] Spatial diffusion
            diffusion = self._compute_spatial_diffusion(phi)
            self.PhiManifold[1, :, t] = diffusion
            
            # [2] Disturbance diffusion
            disturbance = self._compute_disturbance_diffusion(t)
            self.PhiManifold[2, :, t] = disturbance
            
            # [3] Nonlocal bleed
            nonlocal_term = self._compute_nonlocal_bleed(phi)
            self.PhiManifold[3, :, t] = nonlocal_term
            
            # [4] Nonlinear saturation
            nonlinear = self._compute_nonlinear_saturation(phi)
            self.PhiManifold[4, :, t] = nonlinear
            
            # [5] Stochastic kicks
            stochastic = self._compute_stochastic_kicks(t)
            self.PhiManifold[5, :, t] = stochastic
            
            # Sum all contributions
            phi_next_raw = memory + diffusion + disturbance + nonlocal_term + nonlinear + stochastic
            
            # Apply tanh soft clamping to keep phi bounded
            phi = torch.tanh(phi_next_raw)
        
        return self.PhiManifold

    def get_pauli_channel(self) -> torch.Tensor:
        """
        Project 6-channel phi manifold into 3-channel Pauli error space.
        
        Formula:
            PauliChannel[p, q, t] = Σ_f W[p, f] * Φ[f, q, t] + B[p]
            
        Where:
            - Φ: PhiManifold (6, num_qubits, max_time)
            - W: DecoherenceProjectionMatrix (3, 6) 
            - B: BaselinePauliOffset (3,)
            
        Returns:
            PauliChannel: (3, num_qubits, max_time)
        """
        # Φ shape: (6, num_qubits, max_time)
        # W shape: (3, 6)
        # Want: (3, num_qubits, max_time)
        
        # Reshape for matmul: (6, num_qubits * max_time)
        num_qubits = self.PhiManifold.shape[1]
        max_time = self.PhiManifold.shape[2]
        
        phi_reshaped = self.PhiManifold.reshape(6, -1)  # (6, num_qubits * max_time)
        
        # Project: (3, 6) @ (6, num_qubits * max_time) -> (3, num_qubits * max_time)
        pauli_flat = torch.matmul(self.DecoherenceProjectionMatrix, phi_reshaped)
        
        # Reshape back: (3, num_qubits, max_time)
        pauli_channel = pauli_flat.reshape(3, num_qubits, max_time)
        
        # Add baseline offset (broadcast over qubits and time)
        pauli_channel = pauli_channel + self.BaselinePauliOffset[:, None, None]
        
        return pauli_channel
    #---Star Function :) Annotating the ideal circuit with noise channels at appropriate time steps---#
    # once this is complete the circuit can be simulated with noise at each time step
    def annotate_circuit(self) -> Circuit:
        """
        Annotate circuit gates with Pauli error probabilities from phi manifold.
        
        For each gate at time t on qubits Q, extract error probabilities
        from the manifold at locations (q, t) and store in gate.metadata.
        
        The backend can then apply noise by sampling from these probabilities
        during circuit execution.
        
        Formula for each qubit q at time t:
            p_x(q,t) = sigmoid(PauliChannel[0, q, t] - bias)
            p_y(q,t) = sigmoid(PauliChannel[1, q, t] - bias)
            p_z(q,t) = sigmoid(PauliChannel[2, q, t] - bias)
            
            Normalize: if p_total > 1, scale by (1 / p_total)
            p_i = 1 - (p_x + p_y + p_z)  # No error probability
        
        Returns:
            Circuit: Same circuit with noise_model added to gate.metadata
        """
        # Get Pauli channel (3, num_qubits, max_time)
        pauli_channel = self.get_pauli_channel().cpu()
        
        # Shift to get realistic error rates (1-5%)
        pauli_channel = pauli_channel - 3.0
        
        # Sigmoid activation
        def sigmoid(x):
            return 1.0 / (1.0 + torch.exp(-x))
        
        # Convert to probabilities
        p_x_all = sigmoid(pauli_channel[0])  # (num_qubits, max_time)
        p_y_all = sigmoid(pauli_channel[1])
        p_z_all = sigmoid(pauli_channel[2])
        
        # Statistics tracking
        total_gates_annotated = 0
        max_error_prob = 0.0
        error_distribution = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        
        # ========================================================================
        # ANNOTATE EACH GATE
        # ========================================================================
        for gate in self.circuit.gates:
            if gate.t is None:
                # Skip unscheduled gates
                continue
            
            t = gate.t
            pauli_probs = {}
            gate_max_error = 0.0
            dominant_error = 'I'
            
            # For each qubit this gate touches
            for q in gate.qubits:
                # Extract probabilities at this specific (qubit, time) location
                p_x = p_x_all[q, t].item()
                p_y = p_y_all[q, t].item()
                p_z = p_z_all[q, t].item()
                
                # Normalize to ensure sum ≤ 1
                p_total = p_x + p_y + p_z
                
                if p_total > 1.0:
                    # Scale down proportionally
                    scale = 1.0 / p_total
                    p_x *= scale
                    p_y *= scale
                    p_z *= scale
                    p_total = 1.0
                
                # Compute no-error probability
                p_i = 1.0 - p_total
                
                # Store as [p_i, p_x, p_y, p_z]
                pauli_probs[q] = [p_i, p_x, p_y, p_z]
                
                # Track dominant error for this gate
                if p_total > gate_max_error:
                    gate_max_error = p_total
                    dominant_error = max(
                        [('X', p_x), ('Y', p_y), ('Z', p_z)],
                        key=lambda x: x[1]
                    )[0]
                
                # Update global distribution
                error_distribution['X'] += p_x
                error_distribution['Y'] += p_y
                error_distribution['Z'] += p_z
            
            # Store noise model in gate metadata
            gate.metadata['noise_model'] = {
                'pauli_probs': pauli_probs,
                'source': 'phi_manifold',
                'time_step': t,
                'gate_name': gate.name,
                'dominant_error': dominant_error,
                'max_error_prob': gate_max_error,
                'burst_weight': gate.get_burst_weight()
            }
            
            total_gates_annotated += 1
            max_error_prob = max(max_error_prob, gate_max_error)
        
        # ========================================================================
        # STORE CIRCUIT-LEVEL STATISTICS
        # ========================================================================
        total_error = sum(error_distribution.values())
        
        self.circuit.metadata['noise_annotation'] = {
            'source': 'phi_manifold',
            'gates_annotated': total_gates_annotated,
            'max_error_probability': max_error_prob,
            'error_distribution': {
                'X': (error_distribution['X'] / total_error * 100) if total_error > 0 else 0,
                'Y': (error_distribution['Y'] / total_error * 100) if total_error > 0 else 0,
                'Z': (error_distribution['Z'] / total_error * 100) if total_error > 0 else 0
            },
            'hardware_preset': getattr(self, '_hardware_preset', 'superconducting'),
            'phi_manifold_shape': tuple(self.PhiManifold.shape),
            'hyperparameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'kappa': self.kappa,
                'epsilon': self.epsilon,
                'sigma': self.sigma,
                'a': self.a,
                'b': self.b
            }
        }
        
        return self.circuit
 
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_composite_manifold(self) -> torch.Tensor:
        """
        Get composite manifold (sum over all 6 feature channels).
        
        Returns:
            Composite: (num_qubits, max_time)
        """
        return self.PhiManifold.sum(dim=0)
    
    def get_feature_channel(self, channel_idx: int) -> torch.Tensor:
        """
        Get specific feature channel.
        
        Args:
            channel_idx: Index 0-5
            
        Returns:
            Feature: (num_qubits, max_time)
        """
        if not 0 <= channel_idx < 6:
            raise ValueError(f"channel_idx must be 0-5, got {channel_idx}")
        
        return self.PhiManifold[channel_idx]
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics of composite manifold"""
        composite = self.get_composite_manifold()
        
        return {
            'max': composite.max().item(),
            'min': composite.min().item(),
            'mean': composite.mean().item(),
            'std': composite.std().item(),
            'total_activity': torch.abs(composite).sum().item(),
            'peak_time': composite.abs().sum(dim=0).argmax().item(),
            'peak_qubit': composite.abs().sum(dim=1).argmax().item()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Compute contribution of each feature to total activity.
        
        Returns:
            Dict mapping feature index to percentage contribution
        """
        # Total absolute contribution per feature
        feature_totals = torch.abs(self.PhiManifold).sum(dim=(1, 2))
        
        # Normalize to percentages
        total = feature_totals.sum()
        if total == 0:
            return {f"feature_{i}": 0.0 for i in range(6)}
        
        percentages = (feature_totals / total * 100).cpu().numpy()
        
        feature_names = [
            'Memory',
            'Spatial Diffusion',
            'Disturbance',
            'Nonlocal Bleed',
            'Nonlinear Saturation',
            'Stochastic Kicks'
        ]
        
        return {name: float(pct) for name, pct in zip(feature_names, percentages)}
    
    def __repr__(self) -> str:
        return (
            f"PhiManifoldExtractor(\n"
            f"  shape=(6, {self.num_qubits}, {self.max_time})\n"
            f"  α={self.alpha:.3f}, λ={self.lam:.3f}\n"
            f"  β={self.beta:.3f}, κ={self.kappa:.3f}\n"
            f"  ε={self.epsilon:.4f}, γ={self.gamma:.3f}\n"
            f"  ρ={self.rho:.3f}, σ={self.sigma:.3f}\n"
            f"  a={self.a:.2f}, b={self.b:.2f}\n"
            f"  device={self.device}\n"
            f")"
        )
class QtorchBackend:
    """
    This is the main entry point for the Qtorch quantum computing backend.
    This will contain functions to configure the qubits, statevectors etc
    
    Then will have three main flags:
        . simulate_with_noise: bool - whether to simulate with noise or not
        . persistant_data: bool - whether to store persistant data or not
        . fusion_optimizations: bool - whether to use fusion optimizations or not 
          (this is optional for noise retained simulations)
    """
    
    def __init__(
        self, 
        simulate_with_noise: bool = False,
        persistant_data: bool = True,
        fusion_optimizations: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        circuit: Circuit = None,
        quantized_angle_precision: float = 0.001,
        parametric_cache_size: int = 1024,
        verbose: bool = False
    ):
        self.simulate_with_noise = simulate_with_noise
        self.persistant_data = persistant_data
        self.fusion_optimizations = fusion_optimizations
        self.device = device
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits if circuit else 0
        self.verbose = verbose
        
        if self.num_qubits > 24:
            raise ValueError(
                f"QtorchBackend supports up to 24 qubits but given {self.num_qubits}"
            )
        
        # Initialize statevector to |0...0⟩
        self.statevector = torch.zeros(
            (2**self.num_qubits,), 
            dtype=torch.complex64, 
            device=self.device
        )
        self.statevector[0] = 1.0 + 0.0j
        
        # Caching configuration
        self.angle_precision = quantized_angle_precision if persistant_data else None
        self.parametric_cache_size = parametric_cache_size if persistant_data else 0
        
        # Cache storage
        self.fixed_cache = {} if persistant_data else None
        self.parametric_cache = None  # Will be set by _setup_parametric_cache()
        
        # Classical register for measurement outcomes
        self.classical_register = {}
        
        # Initialize caches if persistent_data is enabled
        if self.persistant_data:
            self._precompute_fixed_gates()
            self._setup_parametric_cache(self.parametric_cache_size)
    
    def _precompute_fixed_gates(self) -> None:
        """
        Precompute and cache all non-parametric gate matrices.
        Only runs if persistent_data=True.
        
        Caches 28 static gates from the 43-gate library.
        """
        # All static (non-parametric) gates in your library
        fixed_gate_names = [
            # === Single-qubit gates (14) ===
            'I',           # Identity
            'X', 'Y', 'Z', # Pauli gates
            'H',           # Hadamard
            'S', 'SDG',    # Phase gates
            'T', 'TDG',    # π/8 gates
            'SX', 'SY', 'SZ',  # Square root gates
            'V', 'VDG',    # V gates
            
            # === Two-qubit gates (10) ===
            'CNOT', 'CX',  # Controlled-NOT (aliases)
            'CY', 'CZ',    # Controlled Pauli
            'SWAP',        # SWAP
            'ISWAP',       # iSWAP
            'SQRT_SWAP',   # √SWAP
            'CH',          # Controlled-Hadamard
            'ECR',         # Echoed Cross-Resonance
            'DCX',         # Double CNOT
            
            # === Three-qubit gates (4) ===
            'TOFFOLI', 'CCNOT',  # Toffoli (aliases)
            'FREDKIN', 'CSWAP',  # Fredkin (aliases)
        ]
        
        cached_count = 0
        failed_gates = []
        
        for gate_name in fixed_gate_names:
            try:
                # Get matrix from GateLibrary
                matrix = GateLibrary.get_gate(gate_name, [])
                
                if matrix is None:
                    failed_gates.append((gate_name, "returned None"))
                    continue
                
                # Cache with uppercase key
                self.fixed_cache[gate_name.upper()] = matrix.to(
                    dtype=torch.complex64,
                    device=self.device
                )
                cached_count += 1
                
            except Exception as e:
                failed_gates.append((gate_name, str(e)))
                if self.verbose:
                    print(f"[Warning] Failed to cache {gate_name}: {e}")
        
        if self.verbose and cached_count > 0:
            print(f"[Backend] Precomputed {cached_count} fixed gates (persistent_data=True)")
            if failed_gates:
                print(f"[Backend] Skipped {len(failed_gates)} gates:")
                for name, reason in failed_gates[:3]:  # Show first 3
                    print(f"  - {name}: {reason}")
    
    def _setup_parametric_cache(self, maxsize: int) -> None:
        """
        Setup LRU cache for parametric gates with angle quantization.
        Only runs if persistent_data=True.
        
        Parametric gates (15):
            Single-qubit: RX, RY, RZ, P, U1, U2, U3
            Two-qubit: CRX, CRY, CRZ, RXX, RYY, RZZ
        
        Args:
            maxsize: Maximum cache entries (typically 128-1024)
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=maxsize)
        def _cached_parametric_matrix(gate_name: str, quantized_params: tuple) -> torch.Tensor:
            """
            Cached computation of parametric gate matrices.
            
            Uses quantized parameters as cache key for high hit rates.
            Example: RX(0.1234567) and RX(0.1230001) both map to RX(0.123)
            
            Args:
                gate_name: Gate name (uppercase)
                quantized_params: Tuple of quantized angles
                
            Returns:
                Gate matrix on correct device
            """
            # Unpack quantized params back to list
            params_list = list(quantized_params) if quantized_params else []
            
            # Get matrix from GateLibrary
            matrix = GateLibrary.get_gate(gate_name, params_list)
            
            if matrix is None:
                raise ValueError(f"Unknown parametric gate: {gate_name}")
            
            return matrix.to(dtype=torch.complex64, device=self.device)
        
        self.parametric_cache = _cached_parametric_matrix
        
        if self.verbose:
            print(f"[Backend] LRU cache enabled (size={maxsize}, precision={self.angle_precision:.4f} rad)")
    
    def _quantize_params(self, params: Optional[List[float]]) -> Optional[tuple]:
        """
        Quantize angle parameters to nearest precision step.
        
        Increases cache hit rate by collapsing nearby angles to same key.
        
        Example with angle_precision=0.001:
            0.1234567 → 0.123
            0.9998    → 1.000
            π/4       → 0.785 (quantized)
        
        Args:
            params: List of gate parameters (angles in radians)
            
        Returns:
            Tuple of quantized parameters (hashable for cache key)
        """
        if params is None or len(params) == 0:
            return None
        
        quantized = tuple(
            round(p / self.angle_precision) * self.angle_precision
            for p in params
        )
        
        return quantized
    
    def _get_gate_matrix_cached(self, gate: Gate) -> torch.Tensor:
        """
        Get gate matrix using 2-tier caching system.
        
        Tier 1: Fixed cache (28 static gates) - O(1) lookup
        Tier 2: LRU cache (15 parametric gates) - quantized params
        Tier 3: Direct computation (fallback)
        
        Args:
            gate: Gate object
            
        Returns:
            Gate matrix (2^k, 2^k) on correct device
        """
        gate_name = gate.name.upper()
        
        if self.persistant_data:
            # ================================================================
            # TIER 1: Fixed cache (static gates)
            # ================================================================
            if gate_name in self.fixed_cache:
                return self.fixed_cache[gate_name]
            
            # ================================================================
            # TIER 2: LRU cache (parametric gates with quantization)
            # ================================================================
            if gate.params and self.parametric_cache is not None:
                # Quantize parameters for better cache hits
                quantized_params = self._quantize_params(gate.params)
                
                try:
                    return self.parametric_cache(gate_name, quantized_params)
                except Exception as e:
                    raise ValueError(f"Failed to get matrix for {gate_name}: {e}")
            
            # ================================================================
            # TIER 3: Direct computation (not in cache)
            # ================================================================
            matrix = GateLibrary.get_gate(gate_name, gate.params)
            
            if matrix is None:
                raise ValueError(f"Unknown gate: {gate_name}")
            
            matrix = matrix.to(dtype=torch.complex64, device=self.device)
            
            # If non-parametric, add to fixed cache for future use
            if not gate.params:
                self.fixed_cache[gate_name] = matrix
                if self.verbose:
                    print(f"[Cache] Added {gate_name} to fixed cache dynamically")
            
            return matrix
        
        else:
            # ================================================================
            # NO CACHING: Fresh computation every time
            # ================================================================
            matrix = GateLibrary.get_gate(gate_name, gate.params)
            
            if matrix is None:
                raise ValueError(f"Unknown gate: {gate_name}")
            
            return matrix.to(dtype=torch.complex64, device=self.device)
    
    def get_cache_stats(self) -> dict:
        """
        Get detailed statistics about cache usage.
        
        Returns:
            Dictionary with:
                - persistent_data: Whether caching is enabled
                - fixed_cache_size: Number of static gates cached
                - lru_cache: Hit rate, misses, current size
                - angle_precision: Quantization step
        """
        stats = {
            'persistent_data': self.persistant_data,
            'fixed_cache_size': len(self.fixed_cache) if self.fixed_cache else 0,
            'angle_precision': self.angle_precision,
        }
        
        # Show gate names if verbose
        if self.verbose and self.fixed_cache:
            stats['fixed_cache_gates'] = sorted(list(self.fixed_cache.keys()))
        
        # LRU cache statistics
        if self.persistant_data and self.parametric_cache is not None:
            cache_info = self.parametric_cache.cache_info()
            total_requests = cache_info.hits + cache_info.misses
            
            stats['lru_cache'] = {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'current_size': cache_info.currsize,
                'max_size': cache_info.maxsize,
                'hit_rate': (cache_info.hits / total_requests * 100) 
                           if total_requests > 0 else 0.0,
                'total_requests': total_requests
            }
        
        return stats
    
    def clear_lru_cache(self) -> None:
        """Clear LRU cache (useful for benchmarking different quantization levels)"""
        if self.parametric_cache is not None:
            self.parametric_cache.cache_clear()
            
            if self.verbose:
                print("[Backend] LRU cache cleared")
    
    def set_statevector(self, statevector: torch.Tensor) -> None:
        """
        Set custom statevector.
        
        Args:
            statevector: Complex tensor of shape (2^n,)
            
        Raises:
            ValueError: If shape or dtype is incorrect
        """
        if statevector.shape != (2**self.num_qubits,):
            raise ValueError(
                f"Statevector must have shape {(2**self.num_qubits,)}, "
                f"got {statevector.shape}"
            )
        
        if statevector.dtype not in [torch.complex64, torch.complex128]:
            raise ValueError(
                f"Statevector must have dtype torch.complex64 or torch.complex128, "
                f"got {statevector.dtype}"
            )
        
        self.statevector = statevector.to(
            dtype=torch.complex64,  # Standardize to complex64
            device=self.device
        )
        
        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(self.statevector)**2))
        if norm < 1e-10:
            raise ValueError("Statevector has zero norm (invalid state)")
        
        self.statevector = self.statevector / norm
    
    def apply_gate(self, gate: Gate) -> None:
        """
        Apply gate to statevector with optional noise.
        
        Args:
            gate: Gate instance from Circuit
            
        Raises:
            ValueError: If gate is unknown or qubits are invalid
        """
        # ====================================================================
        # VALIDATION
        # ====================================================================
        
        # Validate qubit indices
        for q in gate.qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Gate {gate.name} uses qubit {q} but circuit only has "
                    f"{self.num_qubits} qubits (indices 0-{self.num_qubits-1})"
                )
        
        # Check for duplicate qubits
        if len(gate.qubits) != len(set(gate.qubits)):
            raise ValueError(
                f"Gate {gate.name} has duplicate qubits: {gate.qubits}"
            )
        
        # ====================================================================
        # HANDLE SPECIAL GATES
        # ====================================================================
        
        # Measurement gate
        if gate.name.upper() == 'M':
            for q in gate.qubits:
                self._apply_measure(q)
            return
        
        # Classical control gates (for teleportation)
        if gate.name.upper() == 'XC':
            if gate.depends_on and len(gate.depends_on) >= 2:
                self._apply_classical_pauli(gate.qubits[0], 'X', gate.depends_on)
            return
        
        if gate.name.upper() == 'ZC':
            if gate.depends_on and len(gate.depends_on) >= 2:
                self._apply_classical_pauli(gate.qubits[0], 'Z', gate.depends_on)
            return
        
        # ====================================================================
        # GET GATE MATRIX (WITH CACHING)
        # ====================================================================
        
        U = self._get_gate_matrix_cached(gate)
        
        # Validate matrix shape
        k = len(gate.qubits)
        expected_size = 1 << k  # 2^k
        if U.shape != (expected_size, expected_size):
            raise ValueError(
                f"Gate '{gate.name}' matrix has shape {U.shape}, expected "
                f"{(expected_size, expected_size)} for {k}-qubit gate"
            )
        
        # ====================================================================
        # APPLY IDEAL GATE
        # ====================================================================
        
        self._apply_k_qubit(U, gate.qubits)
        
        # ====================================================================
        # APPLY NOISE (IF ENABLED)
        # ====================================================================
        
        if self.simulate_with_noise and 'noise_model' in gate.metadata:
            self._apply_noise_from_metadata(gate)
    
    def _apply_k_qubit(self, U: torch.Tensor, targets: List[int]) -> None:
        """
        Apply k-qubit gate U on target qubits using efficient tensor reshaping.
        
        This avoids building the full 2^n × 2^n matrix, instead using tensor
        operations for O(2^n) complexity instead of O(2^{2n}).
        
        Args:
            U: Gate matrix of shape (2^k, 2^k)
            targets: List of target qubit indices
        """
        n = self.num_qubits
        k = len(targets)
        expected_size = 1 << k
        
        # Validate matrix size
        assert U.shape == (expected_size, expected_size), \
            f"Matrix shape {U.shape} doesn't match {k}-qubit gate"
        
        # Reshape statevector to n-dimensional tensor: (2, 2, 2, ..., 2)
        psi = self.statevector.view([2] * n)
        
        # Move target qubits to the end via permutation
        # Example: n=5, targets=[1,3] → perm=[0,2,4,1,3]
        targets = list(targets)
        perm = [i for i in range(n) if i not in targets] + targets
        psi = psi.permute(perm)
        
        # Reshape to (batch_size, 2^k) where batch_size = 2^(n-k)
        batch = psi.numel() // expected_size
        psi = psi.reshape(batch, expected_size)
        
        # Apply gate via matrix multiplication
        # (batch, 2^k) @ (2^k, 2^k)^T = (batch, 2^k)
        psi = psi @ U.t()
        
        # Reshape back to n-dimensional tensor
        psi = psi.view([2] * n)
        
        # Inverse permutation to restore original qubit order
        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i
        psi = psi.permute(inv)
        
        # Flatten back to statevector
        self.statevector = psi.reshape(-1)
    
    def _apply_noise_from_metadata(self, gate: Gate) -> None:
        """
        Apply Pauli noise based on phi manifold annotation in gate metadata.
        
        For each qubit the gate touches, sample error type from probabilities
        [p_i, p_x, p_y, p_z] and apply corresponding Pauli operator.
        
        Args:
            gate: Gate with 'noise_model' in metadata
        """
        noise = gate.metadata['noise_model']
        pauli_probs = noise['pauli_probs']
        
        # Apply noise to each qubit this gate touched
        for q, probs in pauli_probs.items():
            p_i, p_x, p_y, p_z = probs
            
            # Sample which error occurs
            r = torch.rand(1, device=self.device).item()
            
            # Apply sampled Pauli error
            if r < p_x:
                # X error (bit flip)
                self._apply_single_pauli('X', q)
                
            elif r < p_x + p_y:
                # Y error (bit + phase flip)
                self._apply_single_pauli('Y', q)
                
            elif r < p_x + p_y + p_z:
                # Z error (phase flip)
                self._apply_single_pauli('Z', q)
            
            # else: no error (identity with probability p_i)
    
    def _apply_single_pauli(self, pauli_name: str, qubit: int) -> None:
        """
        Apply single Pauli gate (X, Y, or Z) to one qubit.
        
        Optimized for noise application - uses cached Pauli matrices.
        
        Args:
            pauli_name: 'X', 'Y', or 'Z'
            qubit: Target qubit index (0 to n-1)
        """
        # Get Pauli matrix (will hit fixed cache)
        pauli_name_upper = pauli_name.upper()
        
        if self.persistant_data and pauli_name_upper in self.fixed_cache:
            U = self.fixed_cache[pauli_name_upper]
        else:
            U = GateLibrary.get_gate(pauli_name, [])
            if U is None:
                raise ValueError(f"Cannot get Pauli matrix for '{pauli_name}'")
            U = U.to(dtype=torch.complex64, device=self.device)
        
        # Apply to single qubit
        self._apply_k_qubit(U, [qubit])
    
    def _apply_measure(self, q: int) -> int:
        """
        Perform Z-basis measurement of a single qubit with state collapse.
        
        Samples outcome from |ψ|² probability distribution, then projects
        the statevector onto the measured eigenspace.
        
        Args:
            q: Qubit index to measure (0 to n-1)
            
        Returns:
            Measurement outcome (0 or 1)
            
        Raises:
            ValueError: If qubit index is invalid
            RuntimeError: If state has zero norm
        """
        n = self.num_qubits
        
        # Validate qubit index
        if q < 0 or q >= n:
            raise ValueError(
                f"Cannot measure qubit {q} (circuit has {n} qubits)"
            )
        
        # Reshape statevector to tensor
        psi = self.statevector.view([2] * n)
        
        # Move measured qubit to last dimension
        perm = [i for i in range(n) if i != q] + [q]
        psi = psi.permute(perm)
        
        # Reshape to (batch, 2) where last dim is measured qubit
        psi = psi.reshape(-1, 2)
        
        # Compute probabilities for |0⟩ and |1⟩
        probs = (psi.conj() * psi).sum(dim=0).real
        p0 = float(probs[0])
        p1 = float(probs[1])
        
        # Normalize (handle numerical errors)
        total = p0 + p1
        if total < 1e-10:
            raise RuntimeError(
                f"Measurement of qubit {q} has zero probability "
                "(invalid quantum state)"
            )
        p0 /= total
        p1 /= total
        
        # Sample measurement outcome
        r = torch.rand((), device=self.device).item()
        outcome = 0 if r < p0 else 1
        
        # Collapse state: project onto |outcome⟩ subspace
        mask = torch.zeros_like(psi)
        mask[:, outcome] = 1.0
        psi = psi * mask
        
        # Renormalize collapsed state
        norm = torch.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        
        # Reshape back to tensor
        psi = psi.view([2] * n)
        
        # Inverse permutation to restore qubit order
        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i
        psi = psi.permute(inv)
        
        # Flatten back to statevector
        self.statevector = psi.reshape(-1)
        
        # Store measurement result in classical register
        self.classical_register[q] = outcome
        
        return outcome
    
    def _apply_classical_pauli(
        self, 
        target_q: int, 
        pauli: str, 
        depends_on: List[Gate]
    ) -> None:
        """
        Apply X or Z gate conditioned on classical measurement outcomes.
        
        Used for quantum teleportation protocol where Bob applies corrections
        based on Alice's measurement results.
        
        Args:
            target_q: Qubit to apply gate on
            pauli: 'X' or 'Z'
            depends_on: List of measurement gates (should be length 2)
        """
        if len(depends_on) != 2:
            return
        
        # Extract measurement gate info
        m0_gate, m1_gate = depends_on
        q0 = m0_gate.qubits[0]
        q1 = m1_gate.qubits[0]
        
        # Get measurement outcomes from classical register
        b0 = self.classical_register.get(q0, None)
        b1 = self.classical_register.get(q1, None)
        
        # If measurements haven't happened yet, skip
        if b0 is None or b1 is None:
            return
        
        # Determine if we should apply the gate
        fire = False
        if pauli == 'Z':
            # Z correction depends on first measurement
            fire = (b0 == 1)
        elif pauli == 'X':
            # X correction depends on second measurement
            fire = (b1 == 1)
        else:
            raise ValueError(f"Unknown classical Pauli: '{pauli}'")
        
        # Apply gate if condition is met
        if fire:
            self._apply_single_pauli(pauli, target_q)
    
    def reset(self) -> None:
        """Reset statevector to |0...0⟩ and clear classical register."""
        self.statevector.zero_()
        self.statevector[0] = 1.0 + 0.0j
        self.classical_register.clear()
    
    def measure_all(self) -> str:
        """
        Measure all qubits (sampling from probability distribution).
        
        Returns:
            Bitstring representing measurement outcome (e.g., "01101")
        """
        # Sample from probability distribution |ψ|²
        probs = torch.abs(self.statevector)**2
        
        # Sample one outcome
        outcome_idx = torch.multinomial(probs, 1).item()
        
        # Convert to binary string
        bitstring = format(outcome_idx, f'0{self.num_qubits}b')
        
        return bitstring
    
    def get_bloch_sphere(self, qubit_index: int) -> Dict[str, float]:
        """
        Get Bloch sphere coordinates for a single qubit.
        
        Computes expectation values ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for the qubit.
        
        Args:
            qubit_index: Qubit to compute Bloch vector for
            
        Returns:
            Dictionary with 'x', 'y', 'z' coordinates
        """
        if qubit_index < 0 or qubit_index >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        n = self.num_qubits
        psi = self.statevector
        
        # Compute ⟨Z⟩
        z_exp = 0.0
        for i in range(2**n):
            bit = (i >> (n - 1 - qubit_index)) & 1
            prob = torch.abs(psi[i])**2
            z_exp += prob.item() * (1 if bit == 0 else -1)
        
        # Save current state
        original_state = psi.clone()
        
        # Compute ⟨X⟩
        if self.persistant_data and 'X' in self.fixed_cache:
            X = self.fixed_cache['X']
        else:
            X = GateLibrary.get_gate('X', []).to(device=self.device)
        
        self._apply_k_qubit(X, [qubit_index])
        x_exp = 2 * torch.real(torch.vdot(original_state, self.statevector)).item()
        
        # Restore and compute ⟨Y⟩
        self.statevector = original_state.clone()
        
        if self.persistant_data and 'Y' in self.fixed_cache:
            Y = self.fixed_cache['Y']
        else:
            Y = GateLibrary.get_gate('Y', []).to(device=self.device)
        
        self._apply_k_qubit(Y, [qubit_index])
        y_exp = 2 * torch.real(torch.vdot(original_state, self.statevector)).item()
        
        # Restore original state
        self.statevector = original_state
        
        return {
            'x': float(x_exp),
            'y': float(y_exp),
            'z': float(z_exp)
        }
    
    def get_all_bloch_sphere(self) -> List[Dict[str, float]]:
        """
        Get Bloch sphere coordinates for all qubits.
        
        Returns:
            List of dictionaries with 'x', 'y', 'z' for each qubit
        """
        return [self.get_bloch_sphere(i) for i in range(self.num_qubits)]
    
    def execute_circuit(self, shots: int = 1) -> List[str]:
        """
        Execute circuit multiple times and return measurement outcomes.
        
        Args:
            shots: Number of times to execute the circuit
            
        Returns:
            List of measurement outcome bitstrings
        """
        if self.circuit is None:
            raise ValueError("No circuit loaded in backend")
        
        results = []
        
        for _ in range(shots):
            # Reset to initial state
            self.reset()
            
            # Apply all gates in circuit
            for gate in self.circuit.gates:
                self.apply_gate(gate)
            
            # Measure all qubits
            outcome = self.measure_all()
            results.append(outcome)
        
        return results
    
    def get_final_statevector(self) -> torch.Tensor:
        """
        Get current statevector (returns a copy on CPU).
        
        Returns:
            Complex tensor of shape (2^n,)
        """
        return self.statevector.clone().cpu()
    
    def get_histogram_data(self, shots: int = 1024) -> Dict[str, int]:
        """
        Execute circuit and return histogram of measurement outcomes.
        
        Args:
            shots: Number of circuit executions
            
        Returns:
            Dictionary mapping bitstrings to counts
        """
        # Execute circuit
        results = self.execute_circuit(shots=shots)
        
        # Count occurrences
        histogram = {}
        for outcome in results:
            histogram[outcome] = histogram.get(outcome, 0) + 1
        
        return histogram


