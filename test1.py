import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from entry import *

# ============================================================================
# VISUALIZATION SUITE
# ============================================================================

def visualize_composite_heatmap(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None,
    title: str = "Phi Manifold - Composite (Absolute Heat)"
):
    """
    Visualize composite heatmap (absolute sum of all 6 features).
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
        title: Plot title
    """
    composite = extractor.get_composite_manifold().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(
        composite,
        aspect='auto',
        cmap='hot',  # Better for absolute heat
        interpolation='bilinear',
        origin='lower'
    )
    
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Qubit Index', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5, color='white')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='|Φ(t)| (Total Heat)')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_all_features(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    THE MONEY SHOT! 2x3 grid showing all 6 feature channels.
    EACH FEATURE GETS ITS OWN INDEPENDENT COLOR SCALE!
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    feature_names = [
        'Memory (Non-Markovian)',
        'Spatial Diffusion',
        'Disturbance Propagation',
        'Nonlocal Bleed',
        'Nonlinear Saturation',
        'Stochastic Kicks'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx in range(6):
        channel = extractor.get_feature_channel(idx).cpu().numpy()
        
        # INDEPENDENT SCALE: Each feature gets its own vmin/vmax
        vmin_local = channel.min()
        vmax_local = channel.max()
        
        # Handle edge case: all zeros or constant
        if vmax_local == vmin_local:
            vmax_local = vmin_local + 1e-6
        
        im = axes[idx].imshow(
            channel,
            aspect='auto',
            cmap='RdYlBu_r',
            interpolation='bilinear',
            origin='lower',
            vmin=vmin_local,  # INDEPENDENT MIN
            vmax=vmax_local   # INDEPENDENT MAX
        )
        
        axes[idx].set_title(f"[{idx}] {feature_names[idx]}", 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time Step', fontsize=10)
        axes[idx].set_ylabel('Qubit', fontsize=10)
        axes[idx].grid(alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Individual colorbar with LOCAL range
        cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        
        # Add range annotation to show actual scale
        range_text = f"[{vmin_local:.3f}, {vmax_local:.3f}]"
        axes[idx].text(
            0.02, 0.98, range_text,
            transform=axes[idx].transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.suptitle(
        'Phi Manifold: 6-Channel Feature Decomposition (Independent Scales)',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_feature_importance(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    Bar chart showing which features dominate the manifold dynamics.
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    importance = extractor.get_feature_importance()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(importance.keys())
    values = list(importance.values())
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
    
    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Contribution to Total Activity (%)', fontsize=14, fontweight='bold')
    ax.set_title('Feature Importance in Phi Manifold', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.5, 
            bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%',
            va='center',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_time_evolution(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    Line plot showing manifold evolution over time (per-qubit traces).
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    composite = extractor.get_composite_manifold().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time_steps = np.arange(composite.shape[1])
    
    # Plot each qubit's trace
    for q in range(composite.shape[0]):
        ax.plot(time_steps, composite[q, :], 
               label=f'Qubit {q}', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('|Φ(t)| (Absolute Heat)', fontsize=14, fontweight='bold')
    ax.set_title('Phi Manifold Time Evolution (Per-Qubit)', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_activity_histogram(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    Histogram showing distribution of phi values across manifold.
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    composite = extractor.get_composite_manifold().cpu().numpy().flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(composite, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('|Φ| Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Phi Manifold Value Distribution', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add mean line
    mean_val = np.mean(composite)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.4f}')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_circuit_overlay(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    Composite heatmap with circuit structure overlay showing gate positions.
    ULTRA COOL for presentations! 🔥
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    composite = extractor.get_composite_manifold().cpu().numpy()
    circuit = extractor.circuit
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Plot heatmap
    im = ax.imshow(
        composite,
        aspect='auto',
        cmap='hot',
        interpolation='bilinear',
        origin='lower',
        alpha=0.8
    )
    
    # Overlay gate markers
    for gate in circuit.gates:
        t = gate.t
        qubits = gate.qubits
        
        # Determine marker style based on gate type
        if gate.name.upper() == 'M':
            marker = 'X'
            color = 'cyan'  # Changed to cyan for visibility on hot colormap
            size = 150
        elif len(qubits) == 1:
            marker = 'o'
            color = 'lime'
            size = 100
        else:
            marker = 's'
            color = 'yellow'
            size = 120
        
        # Plot marker for each qubit
        for q in qubits:
            ax.scatter(t, q, marker=marker, s=size, 
                      color=color, edgecolor='black', linewidth=2, zorder=10)
        
        # Draw connections for multi-qubit gates
        if len(qubits) > 1:
            q_min, q_max = min(qubits), max(qubits)
            ax.plot([t, t], [q_min, q_max], 
                   color='yellow', linewidth=3, linestyle='-', zorder=9)
    
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Qubit Index', fontsize=14, fontweight='bold')
    ax.set_title('Phi Manifold with Circuit Structure Overlay', 
                fontsize=16, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
               markersize=10, label='Single-Qubit Gate'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', 
               markersize=10, label='Multi-Qubit Gate'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='cyan', 
               markersize=10, label='Measurement')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='|Φ(t)| (Total Heat)')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


# ============================================================================
# COMPLETE DEMO SCRIPT
# ============================================================================

def generate_all_visualizations(extractor: PhiManifoldExtractor, prefix: str = "phi"):
    """
    Generate all visualizations for hackathon presentation.
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        prefix: Filename prefix for saved images
    """
    print("="*70)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*70)
    
    # 1. Composite heatmap
    print("\n[1/6] Composite heatmap...")
    visualize_composite_heatmap(extractor, save_path=f"{prefix}_composite.png")
    
    # 2. All features grid (THE MONEY SHOT!)
    print("\n[2/6] All features grid (INDEPENDENT SCALES)...")
    visualize_all_features(extractor, save_path=f"{prefix}_all_features.png")
    
    # 3. Feature importance
    print("\n[3/6] Feature importance...")
    visualize_feature_importance(extractor, save_path=f"{prefix}_importance.png")
    
    # 4. Time evolution
    print("\n[4/6] Time evolution...")
    visualize_time_evolution(extractor, save_path=f"{prefix}_evolution.png")
    
    # 5. Activity histogram
    print("\n[5/6] Activity histogram...")
    visualize_activity_histogram(extractor, save_path=f"{prefix}_histogram.png")
    
    # 6. Circuit overlay
    print("\n[6/6] Circuit overlay...")
    visualize_circuit_overlay(extractor, save_path=f"{prefix}_overlay.png")
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)


# ============================================================================
# TEST WITH TELEPORTATION CIRCUIT
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PHI MANIFOLD VISUALIZATION - TELEPORTATION CIRCUIT")
    print("="*70)
    
    # Build teleportation circuit
    circuit = Circuit(num_qubits=3)
    
    # Step 1: Prepare message state
    circuit.add(Gate("H", [0]))
    
    # Step 2: Create Bell pair
    circuit.add(Gate("H", [1]))
    circuit.add(Gate("CNOT", [1, 2]))
    
    # Step 3: Alice's operations
    circuit.add(Gate("CNOT", [0, 1]))
    circuit.add(Gate("H", [0]))
    
    # Step 4: Measurements
    circuit.add(Gate("M", [0]))
    circuit.add(Gate("M", [1]))
    
    # Step 5: Bob's corrections
    circuit.add(Gate("X", [2]))
    circuit.add(Gate("Z", [2]))
    
    print(f"\nCircuit: {circuit}")
    print("\n" + circuit.visualize())
    
    # Create extractor with OPTIMIZED parameters
    W = torch.randn(6, 3) * 0.1
    B = torch.tensor([0.001, 0.001, 0.001])
    
    extractor = PhiManifoldExtractor(
        circuit,
        DecoherenceProjectionMatrix=W,
        BaselinePauliOffset=B,
        alpha=0.85,
        beta=0.18,
        kappa=0.12,
        epsilon=0.003,
        gamma=1.2,
        rho=0.1,
        sigma=0.06,
        a=1.0,
        b=2.5
    )
    
    print(f"\n{extractor}")
    
    # Extract manifold
    print("\nExtracting manifold...")
    manifold = extractor.GetManifold()
    
    print(f"✓ Shape: {manifold.shape}")
    
    # Print stats
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    stats = extractor.get_stats()
    for key, val in stats.items():
        print(f"  {key:20s}: {val:10.6f}")
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    importance = extractor.get_feature_importance()
    for name, pct in importance.items():
        print(f"  {name:25s}: {pct:6.2f}%")
    
    # Generate all visualizations
    print("\n")
    generate_all_visualizations(extractor, prefix="teleportation")
    
    print("\n🔥 READY TO DEMOLISH THE HACKATHON! 🔥")



