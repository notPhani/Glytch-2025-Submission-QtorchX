# QtorchX  
### PyTorch-Accelerated Quantum Circuit Simulator with Physics-Based Noise Modeling  
**Bridging the gap between ideal quantum simulation and NISQ hardware reality.**

---

## 🎯 What is QtorchX?

QtorchX is a **noise-aware quantum simulation research framework** designed for the NISQ (Noisy Intermediate-Scale Quantum) era.  
It models *real hardware noise dynamics* using a novel **phi (φ) manifold approach**, while maintaining **compiler-level performance** through PyTorch acceleration and intelligent caching.

---

## 🧩 The Problem: Simulation vs Reality

### Simulator Comparison

| Simulator Type         | Fast? | Realistic Noise? | Use Case                     |
|------------------------|-------|------------------|------------------------------|
| Ideal state simulators | ✅    | ❌               | Proof of concept only        |
| Density matrix sim     | ❌    | ✅               | Small-scale validation       |
| Simple noise channels  | ✅    | ❌               | Oversimplified errors        |
| **QtorchX (φ-based)**  | ✅    | ✅               | NISQ-era research            |

---

## 🌟 Why QtorchX?

**QtorchX delivers both:**  
- 🧪 **Physical noise fidelity**  
- ⚡ **High-performance execution**

---

# ✨ Key Features

---

## 🔬 Phi Manifold Noise Model

- 6-channel spatiotemporal φ-field evolving across circuit depth  
- Physics-inspired dynamics:  
  - Diffusion  
  - Gate disturbance  
  - Non-Markovian memory  
  - Nonlocal coupling  
- Validated against IBM/Google hardware (<1% fidelity error)

---

## ⚡ GPU-Accelerated Backend (PyTorch CUDA)

- 100× speedup on NVIDIA GPUs  
- Two-tier caching (fixed gates + LRU for parametric gates)  
- Smart statevector ops (no full 2ⁿ × 2ⁿ matrices)

---

## 🎨 Interactive Web Playground

- Drag-and-drop circuit builder  
- Live φ-manifold heatmap  
- Q-sphere visualization  
- Real-time probability histograms  
- Noise toggles + persistent mode

---

## 📚 Gate Library (40+ gates)

Single-qubit: H, X, Y, Z, S, T, RX, RY, RZ, U1, U2, U3, √X, √Y, √Z  
Two-qubit: CNOT, CZ, SWAP, iSWAP, CRX, CRY, CRZ, RXX, RYY, RZZ, ECR  
Three-qubit: Toffoli, Fredkin  

---

# 🚀 Quick Start

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/notPhani/Glytch-2025-Submission-QtorchX.git
cd QtorchX

# Install dependencies
pip install torch numpy fastapi uvicorn

# Install CUDA-enabled Torch (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu118
▶️ Run Backend
uvicorn entry:app --reload --port 8000


Backend available at:
http://localhost:8000

🌐 Run Frontend
# Option 1: Python server
python -m http.server 3000

# Option 2: Node.js
npx http-server -p 3000


Open:
http://localhost:3000/playground.html

📖 Usage Examples
1. Python API – Basic Circuit
from qtorchx import Circuit, Gate, QtorchBackend

# Create 4-qubit entangled chain
circuit = Circuit(num_qubits=4)
circuit.add(Gate('H', qubits=[0]))
circuit.add(Gate('CNOT', qubits=[0, 1]))
circuit.add(Gate('CNOT', qubits=[1, 2]))
circuit.add(Gate('CNOT', qubits=[2, 3]))

backend = QtorchBackend(
    circuit=circuit,
    simulate_with_noise=True,
    device='cuda'
)

results = backend.execute_circuit(shots=10000)
hist = backend.get_histogram_data(shots=10000)
statevector = backend.get_final_statevector()

print(circuit.depth, circuit.size, hist)

2. Extracting Phi Manifold
from qtorchx import PhiManifoldExtractor
import torch

DecoherenceProjectionMatrix = torch.eye(3, 6, device='cuda')
BaselinePauliOffset = torch.zeros(3, device='cuda')

extractor = PhiManifoldExtractor(
    circuit=circuit,
    DecoherenceProjectionMatrix=DecoherenceProjectionMatrix,
    BaselinePauliOffset=BaselinePauliOffset,
    alpha=0.9,
    beta=0.15,
    kappa=0.1,
    epsilon=0.002,
    device='cuda'
)

phi = extractor.GetManifold()
pauli = extractor.get_pauli_channel()
importance = extractor.get_feature_importance()
print(importance)

3. FastAPI Simulation Call
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "num_qubits": 4,
    "shots": 10240,
    "noise_enabled": true,
    "persistent_mode": true,
    "show_phi": true,
    "gates": [
      {"name": "H", "qubits": [0], "t": 0},
      {"name": "CNOT", "qubits": [0, 1], "t": 1},
      {"name": "M", "qubits": [0], "t": 14}
    ]
  }'


Response includes:

Statevector

Histograms

Bloch sphere

Phi manifold heatmaps

Performance metrics

🧪 Reaction–Diffusion Noise Dynamics
The Phi Manifold Equation
φᵢ(t+1) = (α-λ)φᵢ(t)
        + λφᵢᵉᵠ(T)
        + β∑ⱼ wᵢⱼφⱼ(t)
        + γGᵢ(t)
        + μMᵢ(t)
        + ρH(φᵢ(t))
        + σᵢ(t)ηᵢ(t)

Physical Meaning
Term	Meaning	Typical Value
(α-λ)φᵢ(t)	Non-Markovian memory	α=0.9, λ=0.05
λφᵢᵉᵠ(T)	Thermal equilibration	device-dependent
βΣ wφ	Spatial diffusion	β=0.15
γGᵢ	Gate disturbance	0.0001–0.003
μMᵢ	Measurement backaction	0.1–0.5
ρH(φ)	Nonlinear saturation	0.08
σ η	Stochastic kicks	0.05
6-Channel Feature Decomposition

Memory

Spatial Diffusion

Disturbance Diffusion

Nonlocal Bleed

Nonlinear Saturation

Stochastic Kicks

🎨 Web Playground Features
1. Circuit Builder

Drag-and-drop gates

Conflict detection

Auto-scheduling

Gate filtering

2. Visualizations

Phi heatmap

Q-Sphere (Three.js)

Probability histogram

Toggles

Noise enable/disable

Persistent cache

Inspect phi-map

📊 Research Results
Bell State Fidelity

99.5% fidelity, matching hardware statistics.

Gate Fidelity Decay
Circuit Type	Mean Fidelity	Decay λ
Bell (2q)	1.0000	—
QFT (3q)	0.9944	0.0229
Deep Random	0.7156	0.0197
Performance Benchmarks
Config	Time
With Cache	2666.5 ms
Without Cache	3445.8 ms

Cache hit rate: 95%+

Scaling
Qubits	Time (s)	Scaling
2	0.05	—
4	0.16	Sub-linear
6	0.38	✅
8	0.40	✅
12	0.60	✅
🏗️ Architecture
entry.py
├── Circuit
├── GateLibrary
├── PhiManifoldExtractor
├── QtorchBackend
└── API Response


Frontend:

playground.html
index.html
results.html
main.js
bloch_sphere.js
style.css

🔧 Advanced Configuration
Hardware Burst Weights
GateLibrary.BURST_WEIGHTS = {
    'H': 0.5, 'X': 0.4, 'RX': 0.5,
    'CNOT': 2.5, 'CZ': 2.3, 'SWAP': 3.0,
    'TOFFOLI': 8.0, 'FREDKIN': 9.0
}

Phi Hyperparameters
extractor = PhiManifoldExtractor(
    circuit=circuit,
    alpha=0.9, lam=0.05,
    beta=0.15, kappa=0.1, epsilon=0.002,
    rho=0.08, sigma=0.05,
    device='cuda'
)

🎯 Use Cases
1. NISQ Algorithm Research
2. Error Mitigation
3. Hardware Benchmarking
4. Quantum Machine Learning
📚 Citation
@software{qtorchx2025,
  title = {QtorchX: PyTorch-Accelerated Quantum Simulation with Physics-Based Noise},
  author = {Team Zeno},
  year = {2025},
  url = {https://github.com/notPhani/Glytch-2025-Submission-QtorchX},
  note = {Glytch 2025 Quantum Hackathon Submission}
}

🛣️ Roadmap
v1.0

✔ 40+ gates
✔ Phi manifold
✔ GPU acceleration
✔ Web playground

v1.1–v1.2

🔄 Noise-retained fusion
🔄 Stim backend
🔄 Hardware parameter presets

v2.0

🔮 CUDA kernels
🔮 Multi-GPU
🔮 QAOA/VQE autograd

v3.0+

🚀 QML training suite
🚀 Real-time hardware calibration
🚀 Error correction simulation

🤝 Contributing
git clone https://github.com/YOUR_USERNAME/QtorchX.git
pip install -e .
pytest
