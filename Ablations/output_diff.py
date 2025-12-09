import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from entry import *

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
        extractor = PhiManifoldExtractor(circuit)
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
        extractor = PhiManifoldExtractor(circuit)
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
    
    def test_circuit_3_random_deep(self, shots=10000):
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
        extractor = PhiManifoldExtractor(circuit)
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
