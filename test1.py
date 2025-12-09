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


def visualize_pauli_channels(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    Visualize X, Y, Z error channels separately.
    Shows how 6-channel phi manifold projects into physical Pauli errors.
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    pauli_channel = extractor.get_pauli_channel().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pauli_names = ['X-Error (Bit Flip)', 'Y-Error (Bit+Phase)', 'Z-Error (Dephasing)']
    
    # Find global scale for consistency (or use independent scales)
    vmin_global = pauli_channel.min()
    vmax_global = pauli_channel.max()
    
    for i, (ax, name) in enumerate(zip(axes, pauli_names)):
        im = ax.imshow(
            pauli_channel[i],
            aspect='auto',
            cmap='Reds',
            interpolation='bilinear',
            origin='lower',
            vmin=vmin_global,
            vmax=vmax_global
        )
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Qubit', fontsize=12)
        ax.grid(alpha=0.2, linestyle='--', color='white', linewidth=0.5)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Error Strength', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    
    plt.suptitle('Pauli Error Channels from Phi Manifold', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_pauli_with_identity(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    Visualize ALL 4 Pauli channels: I (no error), X, Y, Z.
    Identity channel shows probability of NO error occurring.
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    pauli_channel = extractor.get_pauli_channel().cpu().numpy()
    
    # Compute identity (no-error) channel
    # P_I = 1 - (P_X + P_Y + P_Z)
    # Use sigmoid to convert error strengths to probabilities
    from scipy.special import expit  # sigmoid
    
    p_x = expit(pauli_channel[0])
    p_y = expit(pauli_channel[1])
    p_z = expit(pauli_channel[2])
    
    # Normalize
    p_total = p_x + p_y + p_z
    p_total = np.clip(p_total, 0, 1)  # Ensure <= 1
    
    p_i = 1.0 - p_total  # Identity (no error)
    
    # Stack all 4 channels
    all_channels = np.stack([p_i, p_x, p_y, p_z], axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    channel_names = [
        'I (No Error / Identity)',
        'X (Bit Flip)',
        'Y (Bit+Phase Flip)',
        'Z (Phase Flip / Dephasing)'
    ]
    
    colormaps = ['Greens', 'Reds', 'Purples', 'Blues']
    
    for i, (ax, name, cmap) in enumerate(zip(axes, channel_names, colormaps)):
        im = ax.imshow(
            all_channels[i],
            aspect='auto',
            cmap=cmap,
            interpolation='bilinear',
            origin='lower',
            vmin=0.0,
            vmax=1.0
        )
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Qubit', fontsize=12)
        ax.grid(alpha=0.2, linestyle='--', color='white', linewidth=0.5)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        # Add mean probability annotation
        mean_prob = all_channels[i].mean()
        ax.text(
            0.02, 0.98, f"Mean: {mean_prob:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.suptitle('Complete Pauli Channel: I, X, Y, Z Error Probabilities', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_pauli_importance(
    extractor: PhiManifoldExtractor,
    save_path: Optional[str] = None
):
    """
    Bar chart showing which Pauli error type dominates.
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        save_path: Path to save figure (optional)
    """
    pauli_channel = extractor.get_pauli_channel().cpu().numpy()
    
    # Compute total "activity" (absolute sum) for each Pauli type
    x_total = np.abs(pauli_channel[0]).sum()
    y_total = np.abs(pauli_channel[1]).sum()
    z_total = np.abs(pauli_channel[2]).sum()
    
    total = x_total + y_total + z_total
    
    # Convert to percentages
    x_pct = (x_total / total) * 100
    y_pct = (y_total / total) * 100
    z_pct = (z_total / total) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = ['X-Error\n(Bit Flip)', 'Y-Error\n(Bit+Phase)', 'Z-Error\n(Dephasing)']
    values = [x_pct, y_pct, z_pct]
    colors = ['#e74c3c', '#9b59b6', '#3498db']
    
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    ax.set_ylabel('Contribution to Total Error Activity (%)', fontsize=14, fontweight='bold')
    ax.set_title('Pauli Error Type Distribution', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 1,
            f'{val:.1f}%',
            ha='center',
            fontsize=13,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


def visualize_pauli_time_evolution(
    extractor: PhiManifoldExtractor,
    qubit: int = 0,
    save_path: Optional[str] = None
):
    """
    Line plot showing how X, Y, Z error probabilities evolve over time
    for a specific qubit.
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        qubit: Which qubit to plot (default: 0)
        save_path: Path to save figure (optional)
    """
    pauli_channel = extractor.get_pauli_channel().cpu().numpy()
    
    # Convert to probabilities using sigmoid
    from scipy.special import expit
    
    p_x = expit(pauli_channel[0, qubit, :])
    p_y = expit(pauli_channel[1, qubit, :])
    p_z = expit(pauli_channel[2, qubit, :])
    
    # Normalize
    p_total = p_x + p_y + p_z
    p_x_norm = p_x / p_total
    p_y_norm = p_y / p_total
    p_z_norm = p_z / p_total
    p_i = 1.0 - (p_x_norm + p_y_norm + p_z_norm)
    
    time_steps = np.arange(len(p_x))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(time_steps, p_i, label='I (No Error)', 
            linewidth=2.5, marker='o', markersize=6, color='green')
    ax.plot(time_steps, p_x_norm, label='X (Bit Flip)', 
            linewidth=2.5, marker='s', markersize=6, color='red')
    ax.plot(time_steps, p_y_norm, label='Y (Bit+Phase)', 
            linewidth=2.5, marker='^', markersize=6, color='purple')
    ax.plot(time_steps, p_z_norm, label='Z (Dephasing)', 
            linewidth=2.5, marker='d', markersize=6, color='blue')
    
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error Probability', fontsize=14, fontweight='bold')
    ax.set_title(f'Pauli Error Evolution (Qubit {qubit})', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


# ============================================================================
# UPDATE generate_all_visualizations
# ============================================================================

def generate_all_visualizations(extractor: PhiManifoldExtractor, prefix: str = "phi"):
    """
    Generate all visualizations for hackathon presentation.
    NOW WITH PAULI CHANNELS! 🔥
    
    Args:
        extractor: PhiManifoldExtractor instance (after GetManifold())
        prefix: Filename prefix for saved images
    """
    print("="*70)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*70)
    
    # 1. Composite heatmap
    print("\n[1/9] Composite heatmap...")
    visualize_composite_heatmap(extractor, save_path=f"{prefix}_composite.png")
    
    # 2. All features grid (THE MONEY SHOT!)
    print("\n[2/9] All features grid (INDEPENDENT SCALES)...")
    visualize_all_features(extractor, save_path=f"{prefix}_all_features.png")
    
    # 3. Feature importance
    print("\n[3/9] Feature importance...")
    visualize_feature_importance(extractor, save_path=f"{prefix}_importance.png")
    
    # 4. Time evolution
    print("\n[4/9] Time evolution...")
    visualize_time_evolution(extractor, save_path=f"{prefix}_evolution.png")
    
    # 5. Activity histogram
    print("\n[5/9] Activity histogram...")
    visualize_activity_histogram(extractor, save_path=f"{prefix}_histogram.png")
    
    # 6. Circuit overlay
    print("\n[6/9] Circuit overlay...")
    visualize_circuit_overlay(extractor, save_path=f"{prefix}_overlay.png")
    
    # 7. Pauli channels (X, Y, Z)
    print("\n[7/9] Pauli channels (X, Y, Z)...")
    visualize_pauli_channels(extractor, save_path=f"{prefix}_pauli_xyz.png")
    
    # 8. Pauli with identity (I, X, Y, Z)
    print("\n[8/9] Pauli with identity (I, X, Y, Z)...")
    visualize_pauli_with_identity(extractor, save_path=f"{prefix}_pauli_ixyz.png")
    
    # 9. Pauli importance
    print("\n[9/9] Pauli error distribution...")
    visualize_pauli_importance(extractor, save_path=f"{prefix}_pauli_importance.png")
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)

# ============================================================================
# MAIN FUNCTION - COMPLETE WORKFLOW
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # ========================================================================
    # ARGUMENT PARSER (Optional - for flexibility)
    # ========================================================================
    parser = argparse.ArgumentParser(
        description='QtorchX: Quantum Circuit Simulator with Phi Manifold Visualization'
    )
    parser.add_argument('--circuit', type=str, default='bell_state',
                       choices=['bell_state', 'teleportation', 'ghz', 'qft'],
                       help='Circuit to simulate')
    parser.add_argument('--qubits', type=int, default=2,
                       help='Number of qubits (for custom circuits)')
    parser.add_argument('--alpha', type=float, default=0.92,
                       help='Memory persistence (0-1)')
    parser.add_argument('--kappa', type=float, default=0.65,
                       help='Disturbance coupling (gate bursts)')
    parser.add_argument('--a', type=float, default=0.6,
                       help='Gate amplification factor')
    parser.add_argument('--b', type=float, default=2.0,
                       help='Measurement amplification factor')
    parser.add_argument('--prefix', type=str, default='phi_manifold',
                       help='Output filename prefix')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # ========================================================================
    # PRINT HEADER
    # ========================================================================
    print("="*70)
    print("  ___  _____                 _     __   __")
    print(" / _ \\_   _|__  _ __ ___| |__  \\ \\ / /")
    print("| | | || |/ _ \\| '__/ __| '_ \\  \\ V / ")
    print("| |_| || | (_) | | | (__| | | |  | |  ")
    print(" \\__\\_\\|_|\\___/|_|  \\___|_| |_|  |_|  ")
    print()
    print("Quantum Circuit Simulator with Phi Manifold Visualization")
    print("Hardware-Calibrated Noise • Non-Markovian Dynamics • 6-Channel Features")
    print("="*70)
    
    # ========================================================================
    # CIRCUIT SELECTION
    # ========================================================================
    if args.circuit == 'bell_state':
        print("\n[Circuit] Building 4 Bell States: Φ+, Φ-, Ψ+, Ψ-")
        circuit = Circuit(num_qubits=2)
        
        # Φ+ = (|00⟩ + |11⟩)/√2
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        
        # Φ- = (|00⟩ - |11⟩)/√2
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        circuit.add(Gate("Z", [0]))
        
        # Ψ+ = (|01⟩ + |10⟩)/√2
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        circuit.add(Gate("X", [0]))
        
        # Ψ- = (|01⟩ - |10⟩)/√2
        circuit.add(Gate("H", [0]))
        circuit.add(Gate("CNOT", [0, 1]))
        circuit.add(Gate("X", [0]))
        circuit.add(Gate("Z", [0]))
        
    elif args.circuit == 'teleportation':
        print("\n[Circuit] Building Quantum Teleportation")
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
        
    elif args.circuit == 'ghz':
        print(f"\n[Circuit] Building {args.qubits}-qubit GHZ State")
        circuit = Circuit(num_qubits=args.qubits)
        
        # GHZ: |000...0⟩ + |111...1⟩
        circuit.add(Gate("H", [0]))
        for i in range(args.qubits - 1):
            circuit.add(Gate("CNOT", [i, i+1]))
            
    elif args.circuit == 'qft':
        print(f"\n[Circuit] Building {args.qubits}-qubit QFT")
        circuit = Circuit(num_qubits=args.qubits)
        
        # Simplified QFT (just Hadamards and phase gates for demo)
        for i in range(args.qubits):
            circuit.add(Gate("H", [i]))
            for j in range(i+1, args.qubits):
                angle = np.pi / (2 ** (j - i))
                circuit.add(Gate("P", [j], params=[angle]))
    
    # ========================================================================
    # DISPLAY CIRCUIT
    # ========================================================================
    print(f"\n[Info] Circuit: {circuit}")
    print("\n[Diagram]")
    print(circuit.visualize())
    
    # ========================================================================
    # INITIALIZE PHI MANIFOLD EXTRACTOR
    # ========================================================================
    print("\n[Setup] Initializing Phi Manifold Extractor...")
    
    # Initialize projection matrices
    W = torch.tensor([
    # X-errors (bit flips): Dominated by stochastic + disturbance
    [0.15, 0.08, 0.25, 0.03, 0.08, 0.35],
    
    # Y-errors (bit+phase): Balanced mix (Y = iXZ)
    [0.12, 0.12, 0.20, 0.05, 0.10, 0.25],
    
    # Z-errors (dephasing): Dominated by memory + disturbance
    [0.40, 0.25, 0.45, 0.02, 0.08, 0.15]
], dtype=torch.float32)

# Baseline Pauli offset (Z-errors have higher baseline)
    B = torch.tensor([0.0005, 0.0003, 0.0025], dtype=torch.float32)
#                    X       Y       Z (3x higher!)
    
    extractor = PhiManifoldExtractor(
        circuit,
        DecoherenceProjectionMatrix=W,
        BaselinePauliOffset=B,
        alpha=args.alpha,     # Memory persistence
        beta=0.12,            # Spatial diffusion
        kappa=args.kappa,     # Disturbance coupling
        epsilon=0.003,        # Nonlocal bleed
        gamma=1.2,            # Distance decay
        rho=0.1,              # Nonlinear saturation
        sigma=0.09,           # Stochastic noise
        a=args.a,             # Gate amplification
        b=args.b              # Measurement amplification
    )
    
    print(f"\n{extractor}")
    
    # ========================================================================
    # EXTRACT PHI MANIFOLD
    # ========================================================================
    print("\n[Processing] Extracting Phi Manifold...")
    print("  - Building graph Laplacian from circuit topology")
    print("  - Computing all-pairs shortest paths (Floyd-Warshall)")
    print("  - Simulating 6-channel coupled dynamics")
    print("  - Applying hardware-calibrated burst weights")
    
    import time
    start_time = time.time()
    
    manifold = extractor.GetManifold()
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Manifold extraction complete in {elapsed:.3f}s")
    print(f"  Shape: {manifold.shape}")
    print(f"  Memory: {manifold.element_size() * manifold.nelement() / 1024:.2f} KB")
    
    # ========================================================================
    # COMPUTE STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("PHI MANIFOLD STATISTICS")
    print("="*70)
    
    stats = extractor.get_stats()
    for key, val in stats.items():
        print(f"  {key:20s}: {val:10.6f}")
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (6 Physical Mechanisms)")
    print("="*70)
    
    importance = extractor.get_feature_importance()
    for name, pct in importance.items():
        bar_length = int(pct / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        print(f"  {name:25s}: {bar:50s} {pct:5.1f}%")
    
    # ========================================================================
    # PROJECT TO PAULI CHANNEL
    # ========================================================================
    print("\n" + "="*70)
    print("PAULI ERROR CHANNEL PROJECTION")
    print("="*70)
    
    pauli_channel = extractor.get_pauli_channel()
    print(f"  PauliChannel shape: {pauli_channel.shape}")
    
    # Compute Pauli error distribution
    x_total = torch.abs(pauli_channel[0]).sum().item()
    y_total = torch.abs(pauli_channel[1]).sum().item()
    z_total = torch.abs(pauli_channel[2]).sum().item()
    total = x_total + y_total + z_total
    
    print(f"\n  X-Error (Bit Flip):     {(x_total/total)*100:5.1f}%")
    print(f"  Y-Error (Bit+Phase):    {(y_total/total)*100:5.1f}%")
    print(f"  Z-Error (Dephasing):    {(z_total/total)*100:5.1f}%")
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    if not args.no_viz:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        generate_all_visualizations(extractor, prefix=args.prefix)
        
        print("\n✓ All visualizations saved to disk!")
        print(f"  Prefix: {args.prefix}_*.png")
    else:
        print("\n[Info] Skipping visualizations (--no-viz flag)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"  Circuit: {args.circuit}")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Depth: {circuit.depth}")
    print(f"  Gates: {circuit.size}")
    print(f"  Processing time: {elapsed:.3f}s")
    print(f"  Hyperparameters: α={args.alpha}, κ={args.kappa}, a={args.a}, b={args.b}")
    
    if not args.no_viz:
        print(f"\n  Generated files:")
        print(f"    - {args.prefix}_composite.png")
        print(f"    - {args.prefix}_all_features.png")
        print(f"    - {args.prefix}_importance.png")
        print(f"    - {args.prefix}_evolution.png")
        print(f"    - {args.prefix}_histogram.png")
        print(f"    - {args.prefix}_overlay.png")
        print(f"    - {args.prefix}_pauli_xyz.png")
        print(f"    - {args.prefix}_pauli_ixyz.png")
        print(f"    - {args.prefix}_pauli_importance.png")
    
    print("\n🔥 QtorchX: Ready to demolish the hackathon! 🔥\n")
