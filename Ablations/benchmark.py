import torch
import numpy as np
import time
from typing import Dict, List
import matplotlib.pyplot as plt
from entry import *

# ============================================================================
# BENCHMARK SUITE FOR QtorchBackend
# ============================================================================
W = torch.tensor([
    # Memory, SpatDiff, Disturb, Nonlocal, Nonlin, Stochastic
    [0.15,    0.08,     0.25,    0.03,     0.08,    0.38],  # X-errors (stochastic-heavy)
    [0.12,    0.12,     0.20,    0.05,     0.10,    0.28],  # Y-errors (balanced)
    [0.45,    0.25,     0.50,    0.02,     0.06,    0.18]   # Z-errors (memory+disturb heavy) ✅
], dtype=torch.float32)

# Baseline should ALSO be different per Pauli type!
B = torch.tensor([-4.0, -4.5, -3.0], dtype=torch.float32)
class QtorchBenchmark:
    """Comprehensive benchmark suite for QtorchBackend"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_caching(self):
        """Benchmark caching performance"""
        print("\n" + "="*70)
        print("BENCHMARK 1: CACHING PERFORMANCE")
        print("="*70)
        
        # Build circuit with mixed gates
        circuit = Circuit(num_qubits=3)
        circuit.add(Gate("H", [0]))           # Static
        circuit.add(Gate("RX", [0], [np.pi/4]))  # Parametric
        circuit.add(Gate("CNOT", [0, 1]))     # Static 2-qubit
        circuit.add(Gate("RY", [1], [np.pi/3]))  # Parametric
        circuit.add(Gate("RX", [0], [np.pi/4]))  # Duplicate (cache hit!)
        circuit.add(Gate("S", [2]))           # Static
        circuit.add(Gate("RZ", [2], [np.pi/6]))  # Parametric
        circuit.add(Gate("CNOT", [1, 2]))     # Static 2-qubit
        
        shots = 1000
        
        # Test WITH caching
        print("\n[With Caching] persistent_data=True")
        backend_cached = QtorchBackend(
            circuit=circuit,
            persistant_data=True,
            verbose=True
        )
        
        start = time.time()
        results_cached = backend_cached.execute_circuit(shots=shots)
        time_cached = time.time() - start
        
        stats = backend_cached.get_cache_stats()
        print(f"\n  Execution time: {time_cached*1000:.2f} ms")
        print(f"  Fixed cache: {stats['fixed_cache_size']} gates")
        if 'lru_cache' in stats:
            print(f"  LRU hits: {stats['lru_cache']['hits']}")
            print(f"  LRU misses: {stats['lru_cache']['misses']}")
            print(f"  LRU hit rate: {stats['lru_cache']['hit_rate']:.1f}%")
        
        # Test WITHOUT caching
        print("\n[Without Caching] persistent_data=False")
        backend_no_cache = QtorchBackend(
            circuit=circuit,
            persistant_data=False
        )
        
        start = time.time()
        results_no_cache = backend_no_cache.execute_circuit(shots=shots)
        time_no_cache = time.time() - start
        
        print(f"  Execution time: {time_no_cache*1000:.2f} ms")
        
        # Compare
        speedup = time_no_cache / time_cached
        print(f"\n[Result] Speedup from caching: {speedup:.2f}x")
        print(f"[Result] Time saved: {(time_no_cache - time_cached)*1000:.2f} ms")
        
        self.results['caching'] = {
            'time_cached': time_cached,
            'time_no_cache': time_no_cache,
            'speedup': speedup
        }
    
    def benchmark_noise_overhead(self):
        """Benchmark noise simulation overhead"""
        print("\n" + "="*70)
        print("BENCHMARK 2: NOISE SIMULATION OVERHEAD")
        print("="*70)
        
        # Build and annotate circuit
        circuit = Circuit(num_qubits=3)
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        circuit.add(Gate("RX", [1], [np.pi/4]))
        circuit.add(Gate("CNOT", [1, 2]))
        circuit.add(Gate("RY", [2], [np.pi/3]))
        
        # Annotate with noise
        extractor = PhiManifoldExtractor(circuit, DecoherenceProjectionMatrix=W, BaselinePauliOffset=B)
        annotated_circuit = extractor.annotate_circuit()
        
        shots = 500
        
        # Test WITHOUT noise
        print("\n[Ideal Simulation] simulate_with_noise=False")
        backend_ideal = QtorchBackend(
            circuit=annotated_circuit,
            simulate_with_noise=False,
            persistant_data=True
        )
        
        start = time.time()
        results_ideal = backend_ideal.execute_circuit(shots=shots)
        time_ideal = time.time() - start
        
        print(f"  Execution time: {time_ideal*1000:.2f} ms")
        print(f"  Time per shot: {time_ideal/shots*1000:.2f} ms")
        
        # Test WITH noise
        print("\n[Noisy Simulation] simulate_with_noise=True")
        backend_noisy = QtorchBackend(
            circuit=annotated_circuit,
            simulate_with_noise=True,
            persistant_data=True
        )
        
        start = time.time()
        results_noisy = backend_noisy.execute_circuit(shots=shots)
        time_noisy = time.time() - start
        
        print(f"  Execution time: {time_noisy*1000:.2f} ms")
        print(f"  Time per shot: {time_noisy/shots*1000:.2f} ms")
        
        # Compare
        overhead = (time_noisy - time_ideal) / time_ideal * 100
        print(f"\n[Result] Noise overhead: {overhead:.1f}%")
        print(f"[Result] Slowdown factor: {time_noisy/time_ideal:.2f}x")
        
        self.results['noise'] = {
            'time_ideal': time_ideal,
            'time_noisy': time_noisy,
            'overhead_percent': overhead
        }
    
    def benchmark_scaling(self):
        """Benchmark scaling with number of qubits"""
        print("\n" + "="*70)
        print("BENCHMARK 3: QUBIT SCALING")
        print("="*70)
        
        qubit_counts = [2, 4, 6, 8, 10, 12]
        times = []
        
        for n_qubits in qubit_counts:
            # Build simple circuit
            circuit = Circuit(num_qubits=n_qubits)
            
            # Add H to all qubits
            for i in range(n_qubits):
                circuit.add(Gate("H", [i]))
            
            # Add CNOT ladder
            for i in range(n_qubits - 1):
                circuit.add(Gate("CNOT", [i, i+1]))
            
            # Benchmark
            backend = QtorchBackend(
                circuit=circuit,
                persistant_data=True
            )
            
            shots = 100
            start = time.time()
            backend.execute_circuit(shots=shots)
            elapsed = time.time() - start
            
            times.append(elapsed)
            
            print(f"  n={n_qubits:2d} qubits: {elapsed*1000:6.2f} ms " +
                  f"({elapsed/shots*1000:5.2f} ms/shot)")
        
        self.results['scaling'] = {
            'qubit_counts': qubit_counts,
            'times': times
        }
    
    def benchmark_gate_types(self):
        """Benchmark different gate types"""
        print("\n" + "="*70)
        print("BENCHMARK 4: GATE TYPE PERFORMANCE")
        print("="*70)
        
        shots = 1000
        gate_times = {}
        
        gate_tests = [
            ("Single-qubit (H)", lambda: Gate("H", [0])),
            ("Single-qubit (RX)", lambda: Gate("RX", [0], [np.pi/4])),
            ("Two-qubit (CNOT)", lambda: Gate("CNOT", [0, 1])),
            ("Two-qubit (CRX)", lambda: Gate("CRX", [0, 1], [np.pi/4])),
            ("Three-qubit (Toffoli)", lambda: Gate("TOFFOLI", [0, 1, 2])),
        ]
        
        for name, gate_fn in gate_tests:
            circuit = Circuit(num_qubits=3)
            
            # Add gate 10 times
            for _ in range(10):
                circuit.add(gate_fn())
            
            backend = QtorchBackend(
                circuit=circuit,
                persistant_data=True
            )
            
            start = time.time()
            backend.execute_circuit(shots=shots)
            elapsed = time.time() - start
            
            gate_times[name] = elapsed
            
            print(f"  {name:25s}: {elapsed*1000:6.2f} ms")
        
        self.results['gate_types'] = gate_times
    
    def benchmark_histogram_generation(self):
        """Benchmark histogram data generation"""
        print("\n" + "="*70)
        print("BENCHMARK 5: HISTOGRAM GENERATION")
        print("="*70)
        
        # Bell state circuit
        circuit = Circuit(num_qubits=2)
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        
        backend = QtorchBackend(
            circuit=circuit,
            persistant_data=True
        )
        
        shot_counts = [100, 1000, 10000]
        
        for shots in shot_counts:
            start = time.time()
            hist = backend.get_histogram_data(shots=shots)
            elapsed = time.time() - start
            
            print(f"\n  Shots: {shots:5d}")
            print(f"    Time: {elapsed*1000:.2f} ms")
            print(f"    Time per shot: {elapsed/shots*1000:.3f} ms")
            print(f"    Results: {hist}")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n" + "="*70)
        print("BENCHMARK 6: MEMORY USAGE")
        print("="*70)
        
        if not torch.cuda.is_available():
            print("  CUDA not available, skipping GPU memory test")
            return
        
        qubit_counts = [2, 4, 6, 8, 10, 12, 14]
        
        print("\n  Qubits | Statevector Size | GPU Memory")
        print("  " + "-"*50)
        
        for n_qubits in qubit_counts:
            circuit = Circuit(num_qubits=n_qubits)
            circuit.add(Gate("H", [0]))
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            backend = QtorchBackend(
                circuit=circuit,
                persistant_data=True,
                device='cuda'
            )
            
            backend.execute_circuit(shots=10)
            
            # Memory in MB
            mem_used = torch.cuda.max_memory_allocated() / 1024**2
            sv_size = 2**n_qubits * 8 * 2 / 1024**2  # complex64 = 8 bytes
            
            print(f"  {n_qubits:6d} | {sv_size:15.2f} MB | {mem_used:10.2f} MB")
    
    def plot_results(self):
        """Plot benchmark results"""
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Caching speedup
        if 'caching' in self.results:
            ax = axes[0, 0]
            data = self.results['caching']
            categories = ['With Cache', 'Without Cache']
            times = [data['time_cached'], data['time_no_cache']]
            
            bars = ax.bar(categories, times, color=['#2ecc71', '#e74c3c'])
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'Caching Performance (Speedup: {data["speedup"]:.2f}x)')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height*1000:.1f} ms',
                       ha='center', va='bottom')
        
        # Plot 2: Noise overhead
        if 'noise' in self.results:
            ax = axes[0, 1]
            data = self.results['noise']
            categories = ['Ideal', 'Noisy']
            times = [data['time_ideal'], data['time_noisy']]
            
            bars = ax.bar(categories, times, color=['#3498db', '#f39c12'])
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'Noise Overhead ({data["overhead_percent"]:.1f}%)')
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height*1000:.1f} ms',
                       ha='center', va='bottom')
        
        # Plot 3: Qubit scaling
        if 'scaling' in self.results:
            ax = axes[1, 0]
            data = self.results['scaling']
            
            ax.plot(data['qubit_counts'], data['times'], 'o-', 
                   linewidth=2, markersize=8, color='#9b59b6')
            ax.set_xlabel('Number of Qubits')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Scaling with Qubit Count')
            ax.grid(alpha=0.3)
            ax.set_yscale('log')
        
        # Plot 4: Gate type comparison
        if 'gate_types' in self.results:
            ax = axes[1, 1]
            data = self.results['gate_types']
            
            names = list(data.keys())
            times = [data[name] * 1000 for name in names]  # Convert to ms
            
            bars = ax.barh(names, times, color='#1abc9c')
            ax.set_xlabel('Time (ms)')
            ax.set_title('Gate Type Performance')
            ax.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.1f} ms',
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('qtorch_benchmark_results.png', dpi=150, bbox_inches='tight')
        print("  Saved plot to: qtorch_benchmark_results.png")
        plt.show()
    
    def run_all(self):
        """Run all benchmarks"""
        print("\n" + "="*70)
        print("QtorchBackend COMPREHENSIVE BENCHMARK SUITE")
        print("="*70)
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"PyTorch version: {torch.__version__}")
        print("="*70)
        
        self.benchmark_caching()
        self.benchmark_noise_overhead()
        self.benchmark_scaling()
        self.benchmark_gate_types()
        self.benchmark_histogram_generation()
        self.benchmark_memory_usage()
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE!")
        print("="*70)
        
        # Print summary
        print("\n[SUMMARY]")
        if 'caching' in self.results:
            print(f"  Caching speedup: {self.results['caching']['speedup']:.2f}x")
        if 'noise' in self.results:
            print(f"  Noise overhead: {self.results['noise']['overhead_percent']:.1f}%")
        
        # Generate plots
        self.plot_results()


# ============================================================================
# RUN BENCHMARKS
# ============================================================================

if __name__ == "__main__":
    benchmark = QtorchBenchmark()
    benchmark.run_all()
