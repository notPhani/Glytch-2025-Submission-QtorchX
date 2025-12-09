import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from entry import *
W = torch.tensor([
    # Memory, SpatDiff, Disturb, Nonlocal, Nonlin, Stochastic
    [0.15,    0.08,     0.25,    0.03,     0.08,    0.38],  # X-errors (stochastic-heavy)
    [0.12,    0.12,     0.20,    0.05,     0.10,    0.28],  # Y-errors (balanced)
    [0.45,    0.25,     0.50,    0.02,     0.06,    0.18]   # Z-errors (memory+disturb heavy) ✅
], dtype=torch.float32)

# Baseline should ALSO be different per Pauli type!
B = torch.tensor([-4.0, -4.5, -3.0], dtype=torch.float32)
# ============================================================================
# FIDELITY ANALYSIS BENCHMARK
# ============================================================================

def state_fidelity(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """
    Compute fidelity between two pure states.
    
    F = |⟨ψ₁|ψ₂⟩|²
    
    Args:
        state1: First statevector (2^n,)
        state2: Second statevector (2^n,)
        
    Returns:
        Fidelity in [0, 1], where 1 = identical states
    """
    overlap = torch.abs(torch.vdot(state1, state2))**2
    return overlap.item()


def trace_distance(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """
    Compute trace distance (for pure states).
    
    D = sqrt(1 - F)
    
    Returns:
        Distance in [0, 1], where 0 = identical states
    """
    fidelity = state_fidelity(state1, state2)
    return np.sqrt(1 - fidelity)


def hellinger_fidelity(hist1: Dict[str, int], hist2: Dict[str, int]) -> float:
    """
    Compute Hellinger fidelity between two probability distributions.
    
    F_H = (Σ_i sqrt(p_i * q_i))²
    
    Args:
        hist1: Histogram 1 (bitstring → count)
        hist2: Histogram 2 (bitstring → count)
        
    Returns:
        Fidelity in [0, 1]
    """
    # Get all possible outcomes
    all_outcomes = set(hist1.keys()) | set(hist2.keys())
    
    # Normalize to probabilities
    total1 = sum(hist1.values())
    total2 = sum(hist2.values())
    
    p = {outcome: hist1.get(outcome, 0) / total1 for outcome in all_outcomes}
    q = {outcome: hist2.get(outcome, 0) / total2 for outcome in all_outcomes}
    
    # Compute Hellinger fidelity
    fidelity = sum(np.sqrt(p[outcome] * q[outcome]) for outcome in all_outcomes)**2
    
    return fidelity


class FidelityBenchmark:
    """Comprehensive fidelity analysis suite"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_bell_state_fidelity(self):
        """
        Benchmark: Bell state fidelity degradation
        
        Ideal: (|00⟩ + |11⟩)/√2
        With noise: Deviates from ideal
        """
        print("\n" + "="*70)
        print("BENCHMARK 1: BELL STATE FIDELITY")
        print("="*70)
        
        # Build Bell state circuit
        circuit = Circuit(num_qubits=2)
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        
        # Annotate with noise
        extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
        annotated_circuit = extractor.annotate_circuit()
        
        # Ideal simulation
        print("\n[Ideal Simulation]")
        backend_ideal = QtorchBackend(
            circuit=circuit,
            simulate_with_noise=False,
            persistant_data=True
        )
        
        for gate in circuit.gates:
            backend_ideal.apply_gate(gate)
        
        ideal_state = backend_ideal.get_final_statevector()
        print(f"  Final state: {ideal_state}")
        print(f"  |00⟩ prob: {torch.abs(ideal_state[0])**2:.4f}")
        print(f"  |11⟩ prob: {torch.abs(ideal_state[3])**2:.4f}")
        
        # Noisy simulations
        print("\n[Noisy Simulations]")
        n_runs = 100
        fidelities = []
        
        for run in range(n_runs):
            backend_noisy = QtorchBackend(
                circuit=annotated_circuit,
                simulate_with_noise=True,
                persistant_data=True
            )
            
            for gate in annotated_circuit.gates:
                backend_noisy.apply_gate(gate)
            
            noisy_state = backend_noisy.get_final_statevector()
            fidelity = state_fidelity(ideal_state, noisy_state)
            fidelities.append(fidelity)
        
        # Statistics
        fidelities = np.array(fidelities)
        mean_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)
        min_fidelity = np.min(fidelities)
        max_fidelity = np.max(fidelities)
        
        print(f"  Mean fidelity: {mean_fidelity:.6f} ± {std_fidelity:.6f}")
        print(f"  Min fidelity:  {min_fidelity:.6f}")
        print(f"  Max fidelity:  {max_fidelity:.6f}")
        print(f"  Fidelity loss: {(1 - mean_fidelity)*100:.2f}%")
        
        self.results['bell_state'] = {
            'fidelities': fidelities,
            'mean': mean_fidelity,
            'std': std_fidelity
        }
    
    def benchmark_circuit_depth_fidelity(self):
        """
        Benchmark: Fidelity vs circuit depth
        
        Deeper circuits → more gates → more noise → lower fidelity
        """
        print("\n" + "="*70)
        print("BENCHMARK 2: FIDELITY VS CIRCUIT DEPTH")
        print("="*70)
        
        depths = [1, 2, 5, 10, 20, 50]
        mean_fidelities = []
        std_fidelities = []
        
        for depth in depths:
            # Build circuit with repeated layers
            circuit = Circuit(num_qubits=3)
            
            for layer in range(depth):
                # Random gates per layer
                circuit.add(Gate("H", [0]))
                circuit.add(Gate("RX", [1], [np.pi/4]))
                circuit.add(Gate("CNOT", [0, 1]))
                circuit.add(Gate("RY", [2], [np.pi/3]))
                circuit.add(Gate("CNOT", [1, 2]))
            
            # Annotate
            extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
            annotated_circuit = extractor.annotate_circuit()
            
            # Ideal
            backend_ideal = QtorchBackend(
                circuit=circuit,
                simulate_with_noise=False,
                persistant_data=True
            )
            for gate in circuit.gates:
                backend_ideal.apply_gate(gate)
            ideal_state = backend_ideal.get_final_statevector()
            
            # Noisy runs
            fidelities = []
            for _ in range(20):
                backend_noisy = QtorchBackend(
                    circuit=annotated_circuit,
                    simulate_with_noise=True,
                    persistant_data=True
                )
                for gate in annotated_circuit.gates:
                    backend_noisy.apply_gate(gate)
                noisy_state = backend_noisy.get_final_statevector()
                
                fidelity = state_fidelity(ideal_state, noisy_state)
                fidelities.append(fidelity)
            
            mean_fid = np.mean(fidelities)
            std_fid = np.std(fidelities)
            
            mean_fidelities.append(mean_fid)
            std_fidelities.append(std_fid)
            
            print(f"  Depth {depth:3d}: Fidelity = {mean_fid:.6f} ± {std_fid:.6f}")
        
        self.results['depth'] = {
            'depths': depths,
            'mean_fidelities': mean_fidelities,
            'std_fidelities': std_fidelities
        }
    
    def benchmark_measurement_fidelity(self):
        """
        Benchmark: Measurement histogram fidelity
        
        Compare ideal vs noisy measurement distributions
        """
        print("\n" + "="*70)
        print("BENCHMARK 3: MEASUREMENT HISTOGRAM FIDELITY")
        print("="*70)
        
        # Build GHZ state: (|000⟩ + |111⟩)/√2
        circuit = Circuit(num_qubits=3)
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        circuit.add(Gate("CNOT", [1, 2]))
        
        # Annotate
        extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
        annotated_circuit = extractor.annotate_circuit()
        
        shots = 10000
        
        # Ideal histogram
        print("\n[Ideal Distribution]")
        backend_ideal = QtorchBackend(
            circuit=circuit,
            simulate_with_noise=False,
            persistant_data=True
        )
        hist_ideal = backend_ideal.get_histogram_data(shots=shots)
        
        print(f"  {hist_ideal}")
        print(f"  Expected: ~50% '000', ~50% '111'")
        
        # Noisy histogram
        print("\n[Noisy Distribution]")
        backend_noisy = QtorchBackend(
            circuit=annotated_circuit,
            simulate_with_noise=True,
            persistant_data=True
        )
        hist_noisy = backend_noisy.get_histogram_data(shots=shots)
        
        print(f"  {hist_noisy}")
        
        # Compute Hellinger fidelity
        fidelity = hellinger_fidelity(hist_ideal, hist_noisy)
        
        print(f"\n[Hellinger Fidelity]")
        print(f"  Fidelity: {fidelity:.6f}")
        print(f"  Infidelity: {(1 - fidelity)*100:.2f}%")
        
        # Analyze error leakage
        error_outcomes = {k: v for k, v in hist_noisy.items() 
                         if k not in ['000', '111']}
        total_error = sum(error_outcomes.values())
        error_rate = total_error / shots
        
        print(f"\n[Error Analysis]")
        print(f"  Leaked to error states: {error_rate*100:.2f}%")
        print(f"  Top error states:")
        for outcome, count in sorted(error_outcomes.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            print(f"    |{outcome}⟩: {count/shots*100:.2f}%")
        
        self.results['measurement'] = {
            'hist_ideal': hist_ideal,
            'hist_noisy': hist_noisy,
            'fidelity': fidelity,
            'error_rate': error_rate
        }
    
    def benchmark_gate_error_contribution(self):
        """
        Benchmark: Which gates contribute most to fidelity loss?
        """
        print("\n" + "="*70)
        print("BENCHMARK 4: PER-GATE ERROR CONTRIBUTION")
        print("="*70)
        
        gate_tests = [
            ("Single H", [Gate("H", [0])]),
            ("Single RX", [Gate("RX", [0], [np.pi/4])]),
            ("CNOT", [Gate("CNOT", [0, 1])]),
            ("Toffoli", [Gate("TOFFOLI", [0, 1, 2])]),
            ("10× H", [Gate("H", [0]) for _ in range(10)]),
            ("10× CNOT", [Gate("CNOT", [0, 1]) for _ in range(10)]),
        ]
        
        print("\n{:20} | {:>12} | {:>15}".format(
            "Gate Sequence", "Fidelity", "Error Rate"
        ))
        print("-" * 52)
        
        gate_fidelities = {}
        
        for name, gates in gate_tests:
            # Build circuit
            circuit = Circuit(num_qubits=3)
            for gate in gates:
                circuit.add(gate)
            
            # Annotate
            extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
            annotated_circuit = extractor.annotate_circuit()
            
            # Ideal
            backend_ideal = QtorchBackend(
                circuit=circuit,
                simulate_with_noise=False,
                persistant_data=True
            )
            for gate in circuit.gates:
                backend_ideal.apply_gate(gate)
            ideal_state = backend_ideal.get_final_statevector()
            
            # Noisy
            fidelities = []
            for _ in range(50):
                backend_noisy = QtorchBackend(
                    circuit=annotated_circuit,
                    simulate_with_noise=True,
                    persistant_data=True
                )
                for gate in annotated_circuit.gates:
                    backend_noisy.apply_gate(gate)
                noisy_state = backend_noisy.get_final_statevector()
                
                fidelity = state_fidelity(ideal_state, noisy_state)
                fidelities.append(fidelity)
            
            mean_fid = np.mean(fidelities)
            error_rate = (1 - mean_fid) * 100
            
            gate_fidelities[name] = mean_fid
            
            print(f"{name:20} | {mean_fid:12.6f} | {error_rate:12.2f}%")
        
        self.results['gate_errors'] = gate_fidelities
    
    def plot_results(self):
        """Plot all fidelity results"""
        print("\n" + "="*70)
        print("GENERATING FIDELITY PLOTS")
        print("="*70)
        
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Bell state fidelity distribution
        if 'bell_state' in self.results:
            ax1 = plt.subplot(2, 3, 1)
            data = self.results['bell_state']
            
            ax1.hist(data['fidelities'], bins=30, color='#3498db', 
                    alpha=0.7, edgecolor='black')
            ax1.axvline(data['mean'], color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {data['mean']:.4f}")
            ax1.set_xlabel('Fidelity', fontsize=11)
            ax1.set_ylabel('Count', fontsize=11)
            ax1.set_title('Bell State Fidelity Distribution', 
                         fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # Plot 2: Fidelity vs depth
        if 'depth' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            data = self.results['depth']
            
            ax2.errorbar(data['depths'], data['mean_fidelities'],
                        yerr=data['std_fidelities'],
                        marker='o', linewidth=2, markersize=8,
                        capsize=5, color='#e74c3c', label='Noisy')
            ax2.axhline(1.0, color='green', linestyle='--', 
                       linewidth=2, label='Ideal')
            ax2.set_xlabel('Circuit Depth (# layers)', fontsize=11)
            ax2.set_ylabel('Fidelity', fontsize=11)
            ax2.set_title('Fidelity Degradation vs Depth', 
                         fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            ax2.set_ylim([0.5, 1.05])
        
        # Plot 3: Measurement histogram comparison
        if 'measurement' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            data = self.results['measurement']
            
            # Normalize histograms
            total_ideal = sum(data['hist_ideal'].values())
            total_noisy = sum(data['hist_noisy'].values())
            
            outcomes = sorted(set(data['hist_ideal'].keys()) | 
                            set(data['hist_noisy'].keys()))
            
            ideal_probs = [data['hist_ideal'].get(o, 0)/total_ideal for o in outcomes]
            noisy_probs = [data['hist_noisy'].get(o, 0)/total_noisy for o in outcomes]
            
            x = np.arange(len(outcomes))
            width = 0.35
            
            ax3.bar(x - width/2, ideal_probs, width, label='Ideal', 
                   color='#2ecc71', alpha=0.8)
            ax3.bar(x + width/2, noisy_probs, width, label='Noisy', 
                   color='#f39c12', alpha=0.8)
            
            ax3.set_xlabel('Outcome', fontsize=11)
            ax3.set_ylabel('Probability', fontsize=11)
            ax3.set_title(f'GHZ State: Fidelity = {data["fidelity"]:.4f}', 
                         fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(outcomes, rotation=45, ha='right', fontsize=9)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Gate error contribution
        if 'gate_errors' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            data = self.results['gate_errors']
            
            names = list(data.keys())
            fidelities = list(data.values())
            error_rates = [(1 - f) * 100 for f in fidelities]
            
            colors = ['#2ecc71' if f > 0.99 else '#f39c12' if f > 0.95 
                     else '#e74c3c' for f in fidelities]
            
            bars = ax4.barh(names, error_rates, color=colors, alpha=0.8)
            ax4.set_xlabel('Error Rate (%)', fontsize=11)
            ax4.set_title('Per-Gate Error Contribution', 
                         fontsize=12, fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            
            # Add values
            for i, (bar, err) in enumerate(zip(bars, error_rates)):
                ax4.text(err + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{err:.2f}%', va='center', fontsize=9)
        
        # Plot 5: Fidelity decay formula fit
        if 'depth' in self.results:
            ax5 = plt.subplot(2, 3, 5)
            data = self.results['depth']
            
            # Fit exponential decay: F(d) = exp(-λd)
            depths = np.array(data['depths'])
            fidelities = np.array(data['mean_fidelities'])
            
            # Log-linear fit
            log_fid = np.log(fidelities)
            lambda_fit = -np.polyfit(depths, log_fid, 1)[0]
            
            # Plot
            ax5.semilogy(depths, fidelities, 'o', markersize=10, 
                        label='Measured', color='#3498db')
            
            # Fit curve
            depths_fit = np.linspace(depths[0], depths[-1], 100)
            fid_fit = np.exp(-lambda_fit * depths_fit)
            ax5.semilogy(depths_fit, fid_fit, '--', linewidth=2,
                        label=f'Fit: F(d) = exp(-{lambda_fit:.4f}×d)', 
                        color='red')
            
            ax5.set_xlabel('Circuit Depth', fontsize=11)
            ax5.set_ylabel('Fidelity (log scale)', fontsize=11)
            ax5.set_title('Exponential Fidelity Decay', 
                         fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        
        # Plot 6: Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "FIDELITY SUMMARY\n" + "="*40 + "\n\n"
        
        if 'bell_state' in self.results:
            data = self.results['bell_state']
            summary_text += f"Bell State (2q):\n"
            summary_text += f"  Mean Fidelity: {data['mean']:.6f}\n"
            summary_text += f"  Error Rate: {(1-data['mean'])*100:.2f}%\n\n"
        
        if 'depth' in self.results:
            data = self.results['depth']
            summary_text += f"Depth Scaling:\n"
            summary_text += f"  Depth 1:  F = {data['mean_fidelities'][0]:.6f}\n"
            summary_text += f"  Depth 50: F = {data['mean_fidelities'][-1]:.6f}\n"
            summary_text += f"  Decay rate: λ = {lambda_fit:.4f}\n\n"
        
        if 'measurement' in self.results:
            data = self.results['measurement']
            summary_text += f"GHZ State (3q):\n"
            summary_text += f"  Histogram Fidelity: {data['fidelity']:.6f}\n"
            summary_text += f"  Error leakage: {data['error_rate']*100:.2f}%\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('fidelity_analysis.png', dpi=150, bbox_inches='tight')
        print("  Saved plot to: fidelity_analysis.png")
        plt.show()
    
    def run_all(self):
        """Run all fidelity benchmarks"""
        print("\n" + "="*70)
        print("FIDELITY ANALYSIS SUITE")
        print("="*70)
        
        self.benchmark_bell_state_fidelity()
        self.benchmark_circuit_depth_fidelity()
        self.benchmark_measurement_fidelity()
        self.benchmark_gate_error_contribution()
        
        print("\n" + "="*70)
        print("FIDELITY ANALYSIS COMPLETE!")
        print("="*70)
        
        self.plot_results()


# ============================================================================
# RUN BENCHMARK
# ============================================================================

if __name__ == "__main__":
    benchmark = FidelityBenchmark()
    benchmark.run_all()
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from entry import *
W = torch.tensor([
    # Memory, SpatDiff, Disturb, Nonlocal, Nonlin, Stochastic
    [0.15,    0.08,     0.25,    0.03,     0.08,    0.38],  # X-errors (stochastic-heavy)
    [0.12,    0.12,     0.20,    0.05,     0.10,    0.28],  # Y-errors (balanced)
    [0.45,    0.25,     0.50,    0.02,     0.06,    0.18]   # Z-errors (memory+disturb heavy) ✅
], dtype=torch.float32)

# Baseline should ALSO be different per Pauli type!
B = torch.tensor([-4.0, -4.5, -3.0], dtype=torch.float32)
# ============================================================================
# OUTPUT DIFFERENCE ANALYSIS
# ============================================================================

def total_variation_distance(hist1: Dict[str, int], hist2: Dict[str, int]) -> float:
    """
    Compute Total Variation Distance between two distributions.
    
    TVD = 0.5 * Σ|p_i - q_i|
    
    Returns:
        Distance in [0, 1], where 0 = identical
    """
    all_outcomes = set(hist1.keys()) | set(hist2.keys())
    
    total1 = sum(hist1.values())
    total2 = sum(hist2.values())
    
    tvd = 0.5 * sum(
        abs(hist1.get(outcome, 0)/total1 - hist2.get(outcome, 0)/total2)
        for outcome in all_outcomes
    )
    
    return tvd


def kl_divergence(hist1: Dict[str, int], hist2: Dict[str, int]) -> float:
    """
    Compute KL divergence: D_KL(P||Q)
    
    Returns:
        Divergence (can be > 1)
    """
    all_outcomes = set(hist1.keys()) | set(hist2.keys())
    
    total1 = sum(hist1.values())
    total2 = sum(hist2.values())
    
    kl = 0.0
    for outcome in all_outcomes:
        p = hist1.get(outcome, 0) / total1
        q = hist2.get(outcome, 0) / total2
        
        if p > 0:
            # Add small epsilon to avoid log(0)
            q = max(q, 1e-10)
            kl += p * np.log(p / q)
    
    return kl


class OutputDifferenceAnalysis:
    """Analyze differences between ideal and noisy circuit outputs"""
    
    def __init__(self):
        self.results = {}
    
    def test_circuit_1_bell_state(self, shots=10000):
        """
        Circuit 1: Bell State
        Simple 2-qubit entanglement
        """
        print("\n" + "="*70)
        print("CIRCUIT 1: BELL STATE (|00⟩ + |11⟩)/√2")
        print("="*70)
        
        circuit = Circuit(num_qubits=2)
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        
        print(f"\nCircuit: {circuit.size} gates, depth {circuit.depth}")
        print(circuit.visualize())
        
        # Annotate
        extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
        annotated = extractor.annotate_circuit()
        
        # Ideal
        backend_ideal = QtorchBackend(
            circuit=circuit,
            simulate_with_noise=False,
            persistant_data=True
        )
        hist_ideal = backend_ideal.get_histogram_data(shots=shots)
        
        # Noisy
        backend_noisy = QtorchBackend(
            circuit=annotated,
            simulate_with_noise=True,
            persistant_data=True
        )
        hist_noisy = backend_noisy.get_histogram_data(shots=shots)
        
        # Analyze
        self._analyze_difference(hist_ideal, hist_noisy, "Bell State")
        
        self.results['bell'] = {
            'circuit': circuit,
            'hist_ideal': hist_ideal,
            'hist_noisy': hist_noisy
        }
    
    def test_circuit_2_qft(self, shots=10000):
        """
        Circuit 2: 3-Qubit QFT
        Complex multi-gate circuit with many rotations
        """
        print("\n" + "="*70)
        print("CIRCUIT 2: 3-QUBIT QUANTUM FOURIER TRANSFORM")
        print("="*70)
        
        circuit = Circuit(num_qubits=3)
        
        # QFT implementation
        # Stage 1
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CRZ", [1, 0], [np.pi/2]))
        circuit.add(Gate("CRZ", [2, 0], [np.pi/4]))
        
        # Stage 2
        circuit.add(Gate("H", [1]))
        circuit.add(Gate("CRZ", [2, 1], [np.pi/2]))
        
        # Stage 3
        circuit.add(Gate("H", [2]))
        
        # Swaps
        circuit.add(Gate("SWAP", [0, 2]))
        
        print(f"\nCircuit: {circuit.size} gates, depth {circuit.depth}")
        print(circuit.visualize())
        
        # Annotate
        extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
        annotated = extractor.annotate_circuit()
        
        # Ideal
        backend_ideal = QtorchBackend(
            circuit=circuit,
            simulate_with_noise=False,
            persistant_data=True
        )
        hist_ideal = backend_ideal.get_histogram_data(shots=shots)
        
        # Noisy
        backend_noisy = QtorchBackend(
            circuit=annotated,
            simulate_with_noise=True,
            persistant_data=True
        )
        hist_noisy = backend_noisy.get_histogram_data(shots=shots)
        
        # Analyze
        self._analyze_difference(hist_ideal, hist_noisy, "QFT")
        
        self.results['qft'] = {
            'circuit': circuit,
            'hist_ideal': hist_ideal,
            'hist_noisy': hist_noisy
        }
    
    def test_circuit_3_random_deep(self, shots=100):
        """
        Circuit 3: Deep Random Circuit
        High depth to show noise accumulation
        """
        print("\n" + "="*70)
        print("CIRCUIT 3: DEEP RANDOM CIRCUIT (10 layers)")
        print("="*70)
        
        circuit = Circuit(num_qubits=4)
        
        # 10 layers of random gates
        np.random.seed(42)
        for layer in range(10):
            # Random single-qubit gates
            for q in range(4):
                gate_choice = np.random.choice(['H', 'RX', 'RY', 'RZ', 'S', 'T'])
                if gate_choice in ['RX', 'RY', 'RZ']:
                    angle = np.random.uniform(0, 2*np.pi)
                    circuit.add(Gate(gate_choice, [q], [angle]))
                else:
                    circuit.add(Gate(gate_choice, [q]))
            
            # Random two-qubit gates
            pairs = [(0,1), (2,3), (1,2)]
            for q1, q2 in pairs:
                circuit.add(Gate("CNOT", [q1, q2]))
        
        print(f"\nCircuit: {circuit.size} gates, depth {circuit.depth}")
        print(f"(Too large to visualize, showing stats)")
        
        # Annotate
        extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
        annotated = extractor.annotate_circuit()
        
        # Ideal
        backend_ideal = QtorchBackend(
            circuit=circuit,
            simulate_with_noise=False,
            persistant_data=True
        )
        hist_ideal = backend_ideal.get_histogram_data(shots=shots)
        
        # Noisy
        backend_noisy = QtorchBackend(
            circuit=annotated,
            simulate_with_noise=True,
            persistant_data=True
        )
        hist_noisy = backend_noisy.get_histogram_data(shots=shots)
        
        # Analyze
        self._analyze_difference(hist_ideal, hist_noisy, "Deep Random")
        
        self.results['deep'] = {
            'circuit': circuit,
            'hist_ideal': hist_ideal,
            'hist_noisy': hist_noisy
        }
    
    def _analyze_difference(self, hist_ideal: Dict, hist_noisy: Dict, name: str):
        """Analyze and print differences between histograms"""
        
        total_ideal = sum(hist_ideal.values())
        total_noisy = sum(hist_noisy.values())
        
        print(f"\n[Ideal Distribution]")
        print(f"  Unique outcomes: {len(hist_ideal)}")
        print(f"  Top 5 outcomes:")
        for outcome, count in sorted(hist_ideal.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            print(f"    |{outcome}⟩: {count/total_ideal*100:5.2f}%")
        
        print(f"\n[Noisy Distribution]")
        print(f"  Unique outcomes: {len(hist_noisy)}")
        print(f"  Top 5 outcomes:")
        for outcome, count in sorted(hist_noisy.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            print(f"    |{outcome}⟩: {count/total_noisy*100:5.2f}%")
        
        # Compute metrics
        tvd = total_variation_distance(hist_ideal, hist_noisy)
        kl = kl_divergence(hist_ideal, hist_noisy)
        
        # Hellinger fidelity
        hellinger_fid = hellinger_fidelity(hist_ideal, hist_noisy)
        
        print(f"\n[Distance Metrics]")
        print(f"  Total Variation Distance: {tvd:.6f}")
        print(f"  KL Divergence D_KL(ideal||noisy): {kl:.6f}")
        print(f"  Hellinger Fidelity: {hellinger_fid:.6f}")
        
        # Outcome spread analysis
        outcomes_only_ideal = set(hist_ideal.keys()) - set(hist_noisy.keys())
        outcomes_only_noisy = set(hist_noisy.keys()) - set(hist_ideal.keys())
        outcomes_both = set(hist_ideal.keys()) & set(hist_noisy.keys())
        
        print(f"\n[Outcome Spread]")
        print(f"  Outcomes in both: {len(outcomes_both)}")
        print(f"  Only in ideal: {len(outcomes_only_ideal)}")
        print(f"  Only in noisy: {len(outcomes_only_noisy)}")
        
        if outcomes_only_noisy:
            print(f"  New outcomes from noise (top 5):")
            new_outcomes = {o: hist_noisy[o] for o in outcomes_only_noisy}
            for outcome, count in sorted(new_outcomes.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]:
                print(f"    |{outcome}⟩: {count/total_noisy*100:5.2f}%")
        
        # Probability shift analysis
        print(f"\n[Probability Shifts]")
        max_increase = 0
        max_decrease = 0
        max_inc_outcome = None
        max_dec_outcome = None
        
        for outcome in outcomes_both:
            p_ideal = hist_ideal[outcome] / total_ideal
            p_noisy = hist_noisy[outcome] / total_noisy
            shift = p_noisy - p_ideal
            
            if shift > max_increase:
                max_increase = shift
                max_inc_outcome = outcome
            if shift < max_decrease:
                max_decrease = shift
                max_dec_outcome = outcome
        
        if max_inc_outcome:
            print(f"  Largest increase: |{max_inc_outcome}⟩ (+{max_increase*100:.2f}%)")
        if max_dec_outcome:
            print(f"  Largest decrease: |{max_dec_outcome}⟩ ({max_decrease*100:.2f}%)")
    
    def plot_comparison(self):
        """Plot side-by-side comparisons"""
        print("\n" + "="*70)
        print("GENERATING COMPARISON PLOTS")
        print("="*70)
        
        fig = plt.figure(figsize=(18, 12))
        
        circuit_names = ['bell', 'qft', 'deep']
        titles = ['Bell State', '3-Qubit QFT', 'Deep Random (10 layers)']
        
        for idx, (name, title) in enumerate(zip(circuit_names, titles)):
            if name not in self.results:
                continue
            
            data = self.results[name]
            hist_ideal = data['hist_ideal']
            hist_noisy = data['hist_noisy']
            
            # Normalize
            total_ideal = sum(hist_ideal.values())
            total_noisy = sum(hist_noisy.values())
            
            # Get all outcomes, sorted by ideal probability
            all_outcomes = sorted(
                set(hist_ideal.keys()) | set(hist_noisy.keys()),
                key=lambda x: hist_ideal.get(x, 0),
                reverse=True
            )
            
            # Limit to top 16 outcomes for readability
            if len(all_outcomes) > 16:
                all_outcomes = all_outcomes[:16]
            
            ideal_probs = [hist_ideal.get(o, 0)/total_ideal for o in all_outcomes]
            noisy_probs = [hist_noisy.get(o, 0)/total_noisy for o in all_outcomes]
            
            # Plot 1: Bar chart comparison
            ax1 = plt.subplot(3, 3, idx*3 + 1)
            x = np.arange(len(all_outcomes))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, ideal_probs, width, 
                           label='Ideal', color='#2ecc71', alpha=0.8)
            bars2 = ax1.bar(x + width/2, noisy_probs, width, 
                           label='Noisy', color='#e74c3c', alpha=0.8)
            
            ax1.set_xlabel('Outcome', fontsize=10)
            ax1.set_ylabel('Probability', fontsize=10)
            ax1.set_title(f'{title}: Probability Distribution', 
                         fontsize=11, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(all_outcomes, rotation=45, ha='right', fontsize=8)
            ax1.legend(fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            
            # Plot 2: Difference plot
            ax2 = plt.subplot(3, 3, idx*3 + 2)
            differences = [noisy_probs[i] - ideal_probs[i] 
                          for i in range(len(all_outcomes))]
            colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in differences]
            
            bars = ax2.bar(x, differences, color=colors, alpha=0.8)
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax2.set_xlabel('Outcome', fontsize=10)
            ax2.set_ylabel('Probability Shift', fontsize=10)
            ax2.set_title(f'{title}: Noisy - Ideal', 
                         fontsize=11, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(all_outcomes, rotation=45, ha='right', fontsize=8)
            ax2.grid(axis='y', alpha=0.3)
            
            # Plot 3: Scatter plot (ideal vs noisy)
            ax3 = plt.subplot(3, 3, idx*3 + 3)
            ax3.scatter(ideal_probs, noisy_probs, s=100, alpha=0.6, color='#3498db')
            
            # Perfect correlation line
            max_prob = max(max(ideal_probs), max(noisy_probs))
            ax3.plot([0, max_prob], [0, max_prob], 'r--', 
                    linewidth=2, label='Perfect match')
            
            ax3.set_xlabel('Ideal Probability', fontsize=10)
            ax3.set_ylabel('Noisy Probability', fontsize=10)
            ax3.set_title(f'{title}: Correlation', 
                         fontsize=11, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(alpha=0.3)
            ax3.set_aspect('equal')
            
            # Compute R²
            from scipy.stats import pearsonr
            if len(ideal_probs) > 1:
                r, _ = pearsonr(ideal_probs, noisy_probs)
                ax3.text(0.05, 0.95, f'R² = {r**2:.4f}',
                        transform=ax3.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('output_difference_analysis.png', dpi=150, bbox_inches='tight')
        print("  Saved plot to: output_difference_analysis.png")
        plt.show()
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        for name, title in [('bell', 'Bell State'), 
                           ('qft', 'QFT'), 
                           ('deep', 'Deep Random')]:
            if name not in self.results:
                continue
            
            data = self.results[name]
            hist_ideal = data['hist_ideal']
            hist_noisy = data['hist_noisy']
            
            tvd = total_variation_distance(hist_ideal, hist_noisy)
            hellinger_fid = hellinger_fidelity(hist_ideal, hist_noisy)
            
            print(f"\n{title}:")
            print(f"  Gates: {data['circuit'].size}, Depth: {data['circuit'].depth}")
            print(f"  TVD: {tvd:.6f}")
            print(f"  Hellinger Fidelity: {hellinger_fid:.6f}")
            print(f"  Unique outcomes (ideal): {len(hist_ideal)}")
            print(f"  Unique outcomes (noisy): {len(hist_noisy)}")
    
    def run_all(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("OUTPUT DIFFERENCE ANALYSIS")
        print("="*70)
        
        self.test_circuit_1_bell_state()
        self.test_circuit_2_qft()
        self.test_circuit_3_random_deep()
        
        self.print_summary()
        self.plot_comparison()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)


# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    analysis = OutputDifferenceAnalysis()
    analysis.run_all()
