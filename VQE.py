"""
VQE Demo - 3 Qubit Version
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict
from entry import *
import torch

EXACT_GROUND_STATE = -1.137283834
CHEMICAL_ACCURACY = 0.0036

def create_ansatz_3q(params: np.ndarray) -> Circuit:
    circuit = Circuit(num_qubits=3)
    
    circuit.add(Gate("RY", [0], [params[0]]))
    circuit.add(Gate("RY", [1], [params[1]]))
    circuit.add(Gate("RY", [2], [params[2]]))
    circuit.add(Gate("CNOT", [0, 1]))
    circuit.add(Gate("CNOT", [1, 2]))
    
    circuit.add(Gate("RY", [0], [params[3]]))
    circuit.add(Gate("RY", [1], [params[4]]))
    circuit.add(Gate("RY", [2], [params[5]]))
    circuit.add(Gate("CNOT", [0, 1]))
    circuit.add(Gate("CNOT", [1, 2]))
    
    circuit.add(Gate("RY", [0], [params[6]]))
    circuit.add(Gate("RY", [1], [params[7]]))
    circuit.add(Gate("RY", [2], [params[8]]))
    circuit.add(Gate("CNOT", [0, 1]))
    circuit.add(Gate("CNOT", [1, 2]))
    
    circuit.add(Gate("RY", [0], [params[9]]))
    circuit.add(Gate("RY", [1], [params[10]]))
    circuit.add(Gate("RY", [2], [params[11]]))
    circuit.add(Gate("CNOT", [0, 1]))
    circuit.add(Gate("CNOT", [1, 2]))
    
    circuit.add(Gate("RY", [0], [params[12]]))
    circuit.add(Gate("RY", [1], [params[13]]))
    circuit.add(Gate("RY", [2], [params[14]]))
    
    return circuit

def compute_energy_3q(params: np.ndarray, use_noise: bool = False) -> float:
    circuit = create_ansatz_3q(params)
    
    if use_noise:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        W = torch.tensor([
            [0.3, 0.2, 0.4, 0.05, 0.03, 0.02],
            [0.2, 0.3, 0.3, 0.1, 0.05, 0.05],
            [0.5, 0.5, 0.3, 0.05, 0.02, 0.03],
        ], dtype=torch.float32, device=device)
        
        B = torch.tensor([-4.0, -4.2, -3.8], dtype=torch.float32, device=device)
        
        extractor = PhiManifoldExtractor(
            circuit=circuit,
            DecoherenceProjectionMatrix=W,
            BaselinePauliOffset=B,
            device=device
        )
        
        extractor.GetManifold()
        circuit = extractor.annotate_circuit()
    
    backend = QtorchBackend(
        circuit=circuit,
        simulate_with_noise=use_noise,
        persistant_data=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    for gate in circuit.gates:
        backend.apply_gate(gate)
    
    psi = backend.statevector.cpu().numpy()
    
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    
    ZII = np.kron(np.kron(Z, I), I)
    IZI = np.kron(np.kron(I, Z), I)
    ZZI = np.kron(np.kron(Z, Z), I)
    XXI = np.kron(np.kron(X, X), I)
    
    E_const = -1.0523732
    E_ZI = -0.39793742 * np.real(np.vdot(psi, ZII @ psi))
    E_IZ = 0.39793742 * np.real(np.vdot(psi, IZI @ psi))
    E_ZZ = -0.01128010 * np.real(np.vdot(psi, ZZI @ psi))
    E_XX = 0.18093119 * np.real(np.vdot(psi, XXI @ psi))
    
    energy = E_const + E_ZI + E_IZ + E_ZZ + E_XX
    
    return energy

def vqe_3q(use_noise: bool = False, max_iterations: int = 50) -> Dict:
    
    print(f"\n{'='*70}")
    print(f"{'NOISY' if use_noise else 'IDEAL'} VQE - 3 Qubits")
    print(f"{'='*70}")
    print(f"Target: {EXACT_GROUND_STATE:.6f} Ha")
    print(f"Ansatz: 15 parameters, 4 layers\n")
    
    np.random.seed(42 if not use_noise else 123)
    params = np.random.uniform(0, 2*np.pi, 15)
    
    energy_history = []
    learning_rate = 0.08 if not use_noise else 0.04
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        
        energy = compute_energy_3q(params, use_noise)
        energy_history.append(energy)
        
        error = abs(energy - EXACT_GROUND_STATE)
        error_pct = error / abs(EXACT_GROUND_STATE) * 100
        
        progress = min(50, int((1 - min(error, 0.5)/0.5) * 50))
        bar = "█" * progress + "░" * (50 - progress)
        
        status = ""
        if error < CHEMICAL_ACCURACY:
            status = "✅"
        elif iteration > 5:
            improvement = abs(energy_history[-5] - energy_history[-1])
            if improvement < 0.001:
                status = "⚠️"
        
        print(f"Iter {iteration:2d} │ {bar} │ E = {energy:.6f} Ha │ Err = {error_pct:.2f}% {status}")
        
        if error < CHEMICAL_ACCURACY:
            print("\nConverged!")
            break
        
        if iteration >= max_iterations - 1:
            break
        
        grad = np.zeros(15)
        shift = np.pi / 2
        
        for i in range(15):
            p_plus = params.copy()
            p_plus[i] += shift
            E_plus = compute_energy_3q(p_plus, use_noise)
            
            p_minus = params.copy()
            p_minus[i] -= shift
            E_minus = compute_energy_3q(p_minus, use_noise)
            
            grad[i] = (E_plus - E_minus) / 2
        
        params = params - learning_rate * grad
        params = params % (2 * np.pi)
        
        if iteration > 10:
            recent_improvement = abs(energy_history[-10] - energy_history[-1])
            if recent_improvement < 0.0005:
                print("\nPlateau detected")
                break
    
    total_time = time.time() - start_time
    final_energy = energy_history[-1]
    final_error = abs(final_energy - EXACT_GROUND_STATE)
    
    print(f"\n{'='*70}")
    print(f"Final: {final_energy:.6f} Ha")
    print(f"Error: {final_error:.6f} Ha ({final_error/abs(EXACT_GROUND_STATE)*100:.2f}%)")
    print(f"Converged: {'✅' if final_error < CHEMICAL_ACCURACY else '❌'}")
    print(f"Time: {total_time:.1f}s")
    print(f"{'='*70}\n")
    
    return {
        'energy_history': energy_history,
        'final_energy': final_energy,
        'final_error': final_error,
        'converged': final_error < CHEMICAL_ACCURACY,
        'iterations': len(energy_history),
        'total_time': total_time,
        'use_noise': use_noise
    }

def plot_results(ideal_result: Dict, noisy_result: Dict):
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    
    ideal_energies = np.array(ideal_result['energy_history'])
    ideal_iters = np.arange(len(ideal_energies))
    
    ax1.plot(ideal_iters, ideal_energies, 'o-', linewidth=2.5, markersize=6, 
             color='#2ecc71', label='Ideal', alpha=0.9)
    
    noisy_energies = np.array(noisy_result['energy_history'])
    noisy_iters = np.arange(len(noisy_energies))
    
    ax1.plot(noisy_iters, noisy_energies, 's-', linewidth=2.5, markersize=5, 
             color='#e74c3c', label='Noisy', alpha=0.8)
    
    ax1.axhline(EXACT_GROUND_STATE, color='black', linestyle='--', linewidth=2, 
                label=f'Exact ({EXACT_GROUND_STATE:.4f} Ha)')
    
    ax1.axhline(EXACT_GROUND_STATE - CHEMICAL_ACCURACY, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7)
    ax1.axhline(EXACT_GROUND_STATE + CHEMICAL_ACCURACY, color='gray', linestyle=':', 
                linewidth=1.5, alpha=0.7, label='Chemical Accuracy')
    
    ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Energy (Hartree)', fontsize=13, fontweight='bold')
    ax1.set_title('VQE Convergence (3 Qubits)', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    table_data = [
        ['Metric', 'Ideal', 'Noisy', 'Difference'],
        ['', '', '', ''],
        ['Final Energy (Ha)', 
         f"{ideal_result['final_energy']:.6f}",
         f"{noisy_result['final_energy']:.6f}",
         f"{abs(noisy_result['final_energy'] - ideal_result['final_energy']):.6f}"],
        ['Error (Ha)',
         f"{ideal_result['final_error']:.6f}",
         f"{noisy_result['final_error']:.6f}",
         f"+{noisy_result['final_error'] - ideal_result['final_error']:.6f}"],
        ['Error (%)',
         f"{ideal_result['final_error']/abs(EXACT_GROUND_STATE)*100:.2f}%",
         f"{noisy_result['final_error']/abs(EXACT_GROUND_STATE)*100:.2f}%",
         f"+{(noisy_result['final_error'] - ideal_result['final_error'])/abs(EXACT_GROUND_STATE)*100:.2f}%"],
        ['Iterations',
         f"{ideal_result['iterations']}",
         f"{noisy_result['iterations']}",
         f"+{noisy_result['iterations'] - ideal_result['iterations']}"],
        ['Time (s)',
         f"{ideal_result['total_time']:.1f}",
         f"{noisy_result['total_time']:.1f}",
         f"+{noisy_result['total_time'] - ideal_result['total_time']:.1f}"],
        ['Converged',
         '✅' if ideal_result['converged'] else '❌',
         '✅' if noisy_result['converged'] else '❌',
         ''],
    ]
    
    table = ax2.table(cellText=table_data, cellLoc='left', loc='center', 
                      bbox=[0.0, 0.0, 1.0, 1.0])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    for i in range(2, 8):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            if j == 3 and '+' in table_data[i][j]:
                cell.set_text_props(color='#e74c3c', weight='bold')
    
    ax2.text(0.5, 0.95, 'Performance Summary', ha='center', va='top', 
             fontsize=15, fontweight='bold', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig('vqe_results_3q.png', dpi=150, bbox_inches='tight')
    print("Saved: vqe_results_3q.png\n")
    plt.show()

def main():
    print("\n" + "="*70)
    print(" "*20 + "QtorchX VQE Demo")
    print(" "*25 + "3-Qubit Test")
    print("="*70)
    
    input("Press ENTER for IDEAL...")
    ideal_result = vqe_3q(use_noise=False, max_iterations=50)
    
    time.sleep(1)
    input("\nPress ENTER for NOISY...")
    noisy_result = vqe_3q(use_noise=True, max_iterations=60)
    
    print("\nGenerating plots...")
    plot_results(ideal_result, noisy_result)
    
    print("="*70)
    print("Done!")
    print("="*70)

if __name__ == "__main__":
    main()
