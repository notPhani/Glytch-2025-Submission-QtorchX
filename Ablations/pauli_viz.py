import torch
from entry import *

# Hardware-calibrated projection (CORRECT!)
W = torch.tensor([
    # Memory, SpatDiff, Disturb, Nonlocal, Nonlin, Stochastic
    [0.15,    0.08,     0.25,    0.03,     0.08,    0.38],  # X-errors (stochastic-heavy)
    [0.12,    0.12,     0.20,    0.05,     0.10,    0.28],  # Y-errors (balanced)
    [0.45,    0.25,     0.50,    0.02,     0.06,    0.18]   # Z-errors (memory+disturb heavy) ✅
], dtype=torch.float32)

# Baseline should ALSO be different per Pauli type!
B = torch.tensor([-4.0, -4.5, -3.0], dtype=torch.float32)
#                  X     Y     Z
# Z has higher baseline → more common (dephasing dominant!)

#============================================================================
# TEST 1: BELL STATE |Φ+⟩
#============================================================================
print("="*70)
print("TEST 1: BELL STATE |Φ+⟩")
print("="*70)

circuit_bell = Circuit(num_qubits=2)
circuit_bell.add(Gate("H", [0]))
circuit_bell.add(Gate("CNOT", [0, 1]))

print(f"\nCircuit: {circuit_bell}")
print(circuit_bell.visualize())

# Extract and annotate
extractor_bell = PhiManifoldExtractor(
    circuit_bell,
    alpha=0.92, beta=0.12, kappa=0.65,
    epsilon=0.003, gamma=1.2, rho=0.1,
    sigma=0.09, a=0.6, b=2.0, 
    DecoherenceProjectionMatrix=W,
    BaselinePauliOffset=B
)

manifold_bell = extractor_bell.GetManifold()
print(f"\n✓ Manifold shape: {manifold_bell.shape}")

annotated_bell = extractor_bell.annotate_circuit()

# Print stats
print("\n" + "-"*70)
print("GATE-BY-GATE NOISE:")
print("-"*70)

for gate in annotated_bell.gates:
    if 'noise_model' not in gate.metadata:
        continue
    
    noise = gate.metadata['noise_model']
    print(f"\n[t={gate.t}] {gate.name} on q{gate.qubits}")
    
    for q, probs in noise['pauli_probs'].items():
        p_i, p_x, p_y, p_z = probs
        total_err = (p_x + p_y + p_z) * 100
        print(f"  q{q}: Error={total_err:5.2f}% (X:{p_x*100:.2f}% Y:{p_y*100:.2f}% Z:{p_z*100:.2f}%)")

# Circuit-level stats
anno = annotated_bell.metadata['noise_annotation']
print(f"\n" + "-"*70)
print("CIRCUIT SUMMARY:")
print("-"*70)
print(f"  Gates annotated: {anno['gates_annotated']}")
print(f"  Max error prob:  {anno['max_error_probability']*100:.2f}%")
print(f"  Error distribution:")
print(f"    X: {anno['error_distribution']['X']:.1f}%")
print(f"    Y: {anno['error_distribution']['Y']:.1f}%")
print(f"    Z: {anno['error_distribution']['Z']:.1f}%")


#============================================================================
# TEST 2: QUANTUM TELEPORTATION
#============================================================================
print("\n\n" + "="*70)
print("TEST 2: QUANTUM TELEPORTATION")
print("="*70)

circuit_teleport = Circuit(num_qubits=3)

# Prepare message
circuit_teleport.add(Gate("H", [0]))

# Create Bell pair
circuit_teleport.add(Gate("H", [1]))
circuit_teleport.add(Gate("CNOT", [1, 2]))

# Alice's operations
circuit_teleport.add(Gate("CNOT", [0, 1]))
circuit_teleport.add(Gate("H", [0]))

# Measurements
circuit_teleport.add(Gate("M", [0]))
circuit_teleport.add(Gate("M", [1]))

# Bob's corrections
circuit_teleport.add(Gate("X", [2]))
circuit_teleport.add(Gate("Z", [2]))

print(f"\nCircuit: {circuit_teleport}")
print(circuit_teleport.visualize())

# Extract and annotate
extractor_teleport = PhiManifoldExtractor(
    circuit_teleport,
    alpha=0.92, beta=0.12, kappa=0.65,
    epsilon=0.003, gamma=1.2, rho=0.1,
    sigma=0.09, a=0.6, b=2.0,
    DecoherenceProjectionMatrix=W,
    BaselinePauliOffset=B
)

manifold_teleport = extractor_teleport.GetManifold()
print(f"\n✓ Manifold shape: {manifold_teleport.shape}")

annotated_teleport = extractor_teleport.annotate_circuit()

# Print stats
print("\n" + "-"*70)
print("GATE-BY-GATE NOISE:")
print("-"*70)

for gate in annotated_teleport.gates:
    if 'noise_model' not in gate.metadata:
        continue
    
    noise = gate.metadata['noise_model']
    
    # Compute average error for this gate
    avg_error = 0.0
    for q, probs in noise['pauli_probs'].items():
        p_i, p_x, p_y, p_z = probs
        avg_error += (p_x + p_y + p_z)
    avg_error = (avg_error / len(noise['pauli_probs'])) * 100
    
    print(f"[t={gate.t:2d}] {gate.name:5s} q{gate.qubits} → "
          f"Avg error: {avg_error:5.2f}% (dominant: {noise['dominant_error']})")

# Circuit-level stats
anno = annotated_teleport.metadata['noise_annotation']
print(f"\n" + "-"*70)
print("CIRCUIT SUMMARY:")
print("-"*70)
print(f"  Gates annotated: {anno['gates_annotated']}")
print(f"  Max error prob:  {anno['max_error_probability']*100:.2f}%")
print(f"  Error distribution:")
print(f"    X: {anno['error_distribution']['X']:.1f}%")
print(f"    Y: {anno['error_distribution']['Y']:.1f}%")
print(f"    Z: {anno['error_distribution']['Z']:.1f}%")


#============================================================================
# COMPARISON
#============================================================================
print("\n\n" + "="*70)
print("COMPARISON: BELL vs TELEPORTATION")
print("="*70)

bell_anno = annotated_bell.metadata['noise_annotation']
tele_anno = annotated_teleport.metadata['noise_annotation']

print(f"\n{'Metric':<30s} {'Bell':>12s} {'Teleport':>12s}")
print("-"*70)
print(f"{'Circuit depth':<30s} {circuit_bell.depth:>12d} {circuit_teleport.depth:>12d}")
print(f"{'Total gates':<30s} {circuit_bell.size:>12d} {circuit_teleport.size:>12d}")
print(f"{'Gates annotated':<30s} {bell_anno['gates_annotated']:>12d} {tele_anno['gates_annotated']:>12d}")
print(f"{'Max error (%)':<30s} {bell_anno['max_error_probability']*100:>12.2f} {tele_anno['max_error_probability']*100:>12.2f}")
print(f"{'X-error proportion (%)':<30s} {bell_anno['error_distribution']['X']:>12.1f} {tele_anno['error_distribution']['X']:>12.1f}")
print(f"{'Y-error proportion (%)':<30s} {bell_anno['error_distribution']['Y']:>12.1f} {tele_anno['error_distribution']['Y']:>12.1f}")
print(f"{'Z-error proportion (%)':<30s} {bell_anno['error_distribution']['Z']:>12.1f} {tele_anno['error_distribution']['Z']:>12.1f}")

print("\n" + "="*70)
print("✓ TESTS COMPLETE!")
print("="*70)
