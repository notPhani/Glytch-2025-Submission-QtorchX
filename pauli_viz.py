
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
