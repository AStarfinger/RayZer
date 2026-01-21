"""
Plot PSNR results from temporal index ablation study using matplotlib.

This script reads the PSNR values exported from the ablation study
and creates publication-ready plots.

Usage:
    python plot_psnr_results.py --psnr_file ablation_results_temporal/psnr_values.csv
    
    or 
    
    python plot_psnr_results.py --summary_file ablation_results_temporal/psnr_summary.json
"""

import argparse
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def load_psnr_csv(csv_file):
    """Load PSNR values from CSV file."""
    temporal_indices = []
    psnr_values = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            temporal_idx = int(row['temporal_index'])
            sample_idx = int(row['sample_index'])
            psnr = float(row['psnr'])
            
            if temporal_idx not in psnr_values:
                psnr_values[temporal_idx] = []
                temporal_indices.append(temporal_idx)
            
            psnr_values[temporal_idx].append(psnr)
    
    temporal_indices = sorted(temporal_indices)
    return temporal_indices, psnr_values


def load_psnr_summary(json_file):
    """Load PSNR summary from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    temporal_indices = sorted([int(k) for k in data.keys()])
    
    # Convert back to the same format as CSV
    psnr_values = {}
    for temporal_idx in temporal_indices:
        psnr_values[temporal_idx] = data[str(temporal_idx)]['samples']
    
    return temporal_indices, psnr_values


def plot_psnr_line(temporal_indices, psnr_values, output_file=None):
    """Plot PSNR as a line graph with error bars."""
    means = []
    stds = []
    
    for temporal_idx in temporal_indices:
        values = psnr_values[temporal_idx]
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(temporal_indices, means, marker='o', linewidth=2.5, markersize=10, 
            color='#1f77b4', label='Mean PSNR', zorder=3)
    
    # Add light error bars (less visually dominant)
    ax.fill_between(temporal_indices, 
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     alpha=0.65, color='#1f77b4', zorder=1)
    
    ax.set_xlabel('Temporal Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('PSNR vs Temporal Index', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(temporal_indices)
    
    # Set y-axis limits to reduce visual starkness of height differences
    y_min = min(means) - 3
    y_max = max(means) + 3
    ax.set_ylim([y_min, y_max])
    
    # Add value labels on points
    for x, y, std in zip(temporal_indices, means, stds):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
    else:
        plt.show()


def plot_psnr_boxplot(temporal_indices, psnr_values, output_file=None):
    """Plot PSNR as box plots for each temporal index."""
    data_to_plot = [psnr_values[idx] for idx in temporal_indices]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data_to_plot, labels=temporal_indices, patch_artist=True)
    
    # Style the boxes with consistent color
    for patch in bp['boxes']:
        patch.set_facecolor('#1f77b4')
        patch.set_alpha(0.7)
    
    # Style whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5)
    for cap in bp['caps']:
        cap.set(linewidth=1.5)
    
    ax.set_xlabel('Temporal Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title('PSNR Distribution by Temporal Index', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Box plot saved to: {output_file}")
    else:
        plt.show()


def plot_psnr_heatmap(temporal_indices, psnr_values, output_file=None):
    """Plot PSNR as a heatmap (temporal index vs sample)."""
    # Get max number of samples
    max_samples = max(len(psnr_values[idx]) for idx in temporal_indices)
    
    # Create matrix
    data_matrix = np.full((len(temporal_indices), max_samples), np.nan)
    
    for i, temporal_idx in enumerate(temporal_indices):
        for j, psnr in enumerate(psnr_values[temporal_idx]):
            data_matrix[i, j] = psnr
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temporal Index', fontsize=12, fontweight='bold')
    ax.set_title('PSNR Heatmap: Temporal Index vs Sample', fontsize=13, fontweight='bold')
    
    ax.set_xticks(range(max_samples))
    ax.set_yticks(range(len(temporal_indices)))
    ax.set_yticklabels(temporal_indices)
    
    cbar = plt.colorbar(im, ax=ax, label='PSNR (dB)')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved to: {output_file}")
    else:
        plt.show()


def print_statistics(temporal_indices, psnr_values):
    """Print PSNR statistics."""
    print("\n" + "="*80)
    print("TEMPORAL INDEX ABLATION STUDY - PSNR STATISTICS")
    print("="*80)
    print(f"\n{'Temporal Index':<15} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print("-"*80)
    
    all_psnr_values = []
    for temporal_idx in temporal_indices:
        values = psnr_values[temporal_idx]
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        all_psnr_values.extend(values)
        
        print(f"{temporal_idx:<15} {mean:<15.4f} {std:<15.4f} {min_val:<15.4f} {max_val:<15.4f}")
    
    print("="*80)
    print("\nOVERALL STATISTICS")
    print("-"*80)
    print(f"Overall Mean PSNR: {np.mean(all_psnr_values):.4f} dB")
    print(f"Overall Std Dev:   {np.std(all_psnr_values):.4f}")
    print(f"Overall Min:       {np.min(all_psnr_values):.4f} dB")
    print(f"Overall Max:       {np.max(all_psnr_values):.4f} dB")
    print("="*80 + "\n")
    
    # Find best and worst temporal indices
    temporal_means = {idx: np.mean(psnr_values[idx]) for idx in temporal_indices}
    best_idx = max(temporal_means, key=temporal_means.get)
    worst_idx = min(temporal_means, key=temporal_means.get)
    
    print("KEY OBSERVATIONS")
    print("-"*80)
    print(f"Best temporal index: {best_idx} with mean PSNR {temporal_means[best_idx]:.4f} dB")
    print(f"Worst temporal index: {worst_idx} with mean PSNR {temporal_means[worst_idx]:.4f} dB")
    
    psnr_drop = temporal_means[worst_idx] - temporal_means[best_idx]
    print(f"PSNR variation: {abs(psnr_drop):.4f} dB (range across temporal indices)")
    
    if abs(psnr_drop) < 1.0:
        print("\n✓ Model shows robustness to temporal index changes (low variation)")
    else:
        print(f"\n✓ Model is sensitive to temporal index assignment (variation > 1 dB)")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot PSNR results from ablation study")
    parser.add_argument('--psnr_file', type=str, default=None,
                       help='Path to psnr_values.csv file')
    parser.add_argument('--summary_file', type=str, default=None,
                       help='Path to psnr_summary.json file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as input file)')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['all', 'line', 'box', 'heatmap'],
                       help='Type of plot to generate (default: all)')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.psnr_file:
        csv_file = args.psnr_file
        print(f"Loading PSNR values from: {csv_file}")
        temporal_indices, psnr_values = load_psnr_csv(csv_file)
        default_output_dir = str(Path(csv_file).parent)
    elif args.summary_file:
        json_file = args.summary_file
        print(f"Loading PSNR summary from: {json_file}")
        temporal_indices, psnr_values = load_psnr_summary(json_file)
        default_output_dir = str(Path(json_file).parent)
    else:
        print("ERROR: Please provide either --psnr_file or --summary_file")
        return
    
    output_dir = args.output_dir or default_output_dir
    
    # Print statistics
    print_statistics(temporal_indices, psnr_values)
    
    # Generate plots
    if args.plot_type in ['all', 'line']:
        output_file = f"{output_dir}/psnr_line_plot.png"
        plot_psnr_line(temporal_indices, psnr_values, output_file=output_file)
    
    if args.plot_type in ['all', 'box']:
        output_file = f"{output_dir}/psnr_boxplot.png"
        plot_psnr_boxplot(temporal_indices, psnr_values, output_file=output_file)
    
    if args.plot_type in ['all', 'heatmap']:
        output_file = f"{output_dir}/psnr_heatmap.png"
        plot_psnr_heatmap(temporal_indices, psnr_values, output_file=output_file)
    
    print("\n✓ Plotting complete!")


if __name__ == '__main__':
    main()
