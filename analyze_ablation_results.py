"""
Visualization and analysis tools for temporal index ablation studies.

This script provides utilities to visualize and analyze the results of
temporal index ablation experiments.

Usage:
    python analyze_ablation_results.py --results_dir ./ablation_results_temporal
"""

import json
import argparse
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_ablation_results(results_file):
    """Load ablation results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def aggregate_metrics_by_temporal_idx(results):
    """
    Aggregate metrics across samples for each temporal index.
    
    Returns:
        dict mapping temporal_idx -> list of metric dicts
    """
    temporal_metrics = defaultdict(list)
    
    for sample in results['samples']:
        for temporal_idx_str, ablation_data in sample['temporal_ablations'].items():
            if 'metrics' in ablation_data:
                temporal_metrics[int(temporal_idx_str)].append(ablation_data['metrics'])
    
    return temporal_metrics


def compute_statistics(temporal_metrics):
    """
    Compute mean and std for each metric across temporal indices.
    
    Returns:
        dict of statistics
    """
    stats = {}
    
    for temporal_idx in sorted(temporal_metrics.keys()):
        metrics_list = temporal_metrics[temporal_idx]
        if not metrics_list:
            continue
        
        stats[temporal_idx] = {}
        
        # Average metrics across samples
        metric_names = metrics_list[0].keys()
        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            stats[temporal_idx][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }
    
    return stats


def print_statistics(stats):
    """Print human-readable statistics."""
    print("\n" + "="*80)
    print("TEMPORAL INDEX ABLATION RESULTS - DETAILED STATISTICS")
    print("="*80 + "\n")
    
    for temporal_idx in sorted(stats.keys()):
        print(f"\nTemporal Index: {temporal_idx}")
        print("-" * 80)
        
        metrics = stats[temporal_idx]
        
        # Print each metric
        for metric_name in sorted(metrics.keys()):
            stat = metrics[metric_name]
            print(f"  {metric_name:30s}: {stat['mean']:10.6f} ± {stat['std']:10.6f} "
                  f"[{stat['min']:10.6f}, {stat['max']:10.6f}]")
    
    print("\n" + "="*80 + "\n")


def plot_metric_vs_temporal_idx(stats, metric_name, output_dir):
    """
    Plot a specific metric as a function of temporal index.
    
    Args:
        stats: Statistics dictionary
        metric_name: Name of metric to plot
        output_dir: Directory to save plot
    """
    temporal_indices = sorted(stats.keys())
    means = []
    stds = []
    
    for idx in temporal_indices:
        if metric_name in stats[idx]:
            means.append(stats[idx][metric_name]['mean'])
            stds.append(stats[idx][metric_name]['std'])
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(temporal_indices, means, yerr=stds, fmt='o-', linewidth=2, 
                 markersize=8, capsize=5, capthick=2, label=metric_name)
    
    plt.xlabel('Temporal Index', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} vs Temporal Index\n(Error bars show ±1 std dev)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    output_path = Path(output_dir) / f'metric_{metric_name.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close()


def plot_all_metrics(stats, output_dir):
    """Generate plots for all metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all metric names
    all_metrics = set()
    for temporal_idx in stats.values():
        all_metrics.update(temporal_idx.keys())
    
    print(f"\nGenerating plots for {len(all_metrics)} metrics...")
    for metric_name in sorted(all_metrics):
        plot_metric_vs_temporal_idx(stats, metric_name, output_dir)


def generate_summary_report(results_file, output_dir=None):
    """
    Generate a comprehensive summary report of the ablation study.
    """
    if output_dir is None:
        output_dir = Path(results_file).parent
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_file}...")
    results = load_ablation_results(results_file)
    
    # Aggregate metrics
    print("Aggregating metrics...")
    temporal_metrics = aggregate_metrics_by_temporal_idx(results)
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(temporal_metrics)
    
    # Print statistics
    print_statistics(stats)
    
    # Generate plots
    print("Generating plots...")
    plot_all_metrics(stats, output_dir)
    
    # Save statistics to JSON
    # Convert numpy types to Python types for JSON serialization
    stats_serializable = {}
    for temporal_idx, metrics in stats.items():
        stats_serializable[temporal_idx] = {}
        for metric_name, stat in metrics.items():
            stats_serializable[temporal_idx][metric_name] = {
                'mean': float(stat['mean']),
                'std': float(stat['std']),
                'min': float(stat['min']),
                'max': float(stat['max']),
            }
    
    stats_file = Path(output_dir) / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"\nSaved statistics: {stats_file}")
    
    return stats


def main():
    import os
    
    parser = argparse.ArgumentParser(
        description="Analyze temporal index ablation study results"
    )
    parser.add_argument(
        '--results_dir', type=str, required=True,
        help='Directory containing ablation_results.json'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for plots and analysis (defaults to results_dir)'
    )
    
    args = parser.parse_args()
    
    results_file = Path(args.results_dir) / 'ablation_results.json'
    
    if not results_file.exists():
        print(f"ERROR: Results file not found at {results_file}")
        return
    
    # Generate report
    generate_summary_report(str(results_file), args.output_dir or args.results_dir)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
