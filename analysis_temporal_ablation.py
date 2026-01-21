"""
Interactive Jupyter notebook for analyzing temporal index ablation results.

This notebook provides interactive exploration and visualization of ablation study results.
"""

# Installation requirements: pip install pandas matplotlib seaborn numpy scipy

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

# Modify this path to your results directory
RESULTS_DIR = './ablation_results_temporal'
RESULTS_FILE = Path(RESULTS_DIR) / 'ablation_results.json'

def load_results(results_file):
    """Load ablation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

# Load results
print("Loading ablation results...")
results = load_results(RESULTS_FILE)
print(f"✓ Loaded results for {len(results['samples'])} samples")

# ============================================================================
# 2. PREPARE DATA FOR ANALYSIS
# ============================================================================

def prepare_dataframe(results):
    """Convert results to pandas DataFrame for easier analysis."""
    rows = []
    
    for sample_idx, sample in enumerate(results['samples']):
        for temporal_idx_str, ablation_data in sample['temporal_ablations'].items():
            if 'metrics' in ablation_data:
                row = {
                    'sample_idx': sample_idx,
                    'temporal_idx': int(temporal_idx_str),
                }
                row.update(ablation_data['metrics'])
                rows.append(row)
    
    return pd.DataFrame(rows)

df = prepare_dataframe(results)
print(f"\n✓ Created DataFrame with {len(df)} rows")
print(f"  Samples: {df['sample_idx'].max() + 1}")
print(f"  Temporal indices: {sorted(df['temporal_idx'].unique())}")
print(f"  Metrics: {[col for col in df.columns if col not in ['sample_idx', 'temporal_idx']]}")

# ============================================================================
# 3. SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS BY TEMPORAL INDEX")
print("="*80)

# Group by temporal index and compute statistics
summary = df.groupby('temporal_idx').agg(['mean', 'std', 'min', 'max'])
print(summary)

# ============================================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_metric_by_temporal_index(df, metric_name, title=None):
    """Plot how a metric varies with temporal index."""
    if title is None:
        title = f'{metric_name} vs Temporal Index'
    
    grouped = df.groupby('temporal_idx')[metric_name].agg(['mean', 'std'])
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                 fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2)
    plt.xlabel('Temporal Index', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

def plot_metric_distribution(df, metric_name):
    """Plot distribution of metric across samples for each temporal index."""
    plt.figure(figsize=(12, 6))
    
    temporal_indices = sorted(df['temporal_idx'].unique())
    data_to_plot = [df[df['temporal_idx'] == idx][metric_name].values 
                    for idx in temporal_indices]
    
    bp = plt.boxplot(data_to_plot, labels=temporal_indices, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(temporal_indices)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Temporal Index', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Distribution of {metric_name} by Temporal Index', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return plt

def plot_heatmap(df, metric_name):
    """Plot metric values as heatmap (samples x temporal_indices)."""
    pivot_table = df.pivot_table(
        index='sample_idx', 
        columns='temporal_idx', 
        values=metric_name
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='RdYlGn', center=pivot_table.mean().mean(),
                cbar_kws={'label': metric_name})
    plt.xlabel('Temporal Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.title(f'{metric_name} Heatmap: Samples vs Temporal Indices', fontsize=14)
    plt.tight_layout()
    return plt

# ============================================================================
# 5. EXAMPLE ANALYSES (Uncomment to run)
# ============================================================================

# Get available metrics
metric_names = [col for col in df.columns if col not in ['sample_idx', 'temporal_idx']]

print(f"\n{'='*80}")
print("AVAILABLE METRICS FOR VISUALIZATION:")
print(f"{'='*80}")
for i, metric in enumerate(metric_names, 1):
    print(f"{i}. {metric}")

print(f"\n{'='*80}")
print("EXAMPLE ANALYSES:")
print(f"{'='*80}\n")

# Example 1: Plot mean_fx vs temporal index
if 'mean_fx' in metric_names:
    print("1. Focal length (fx) vs Temporal Index")
    p = plot_metric_by_temporal_index(df, 'mean_fx', 'Mean Focal Length (fx) vs Temporal Index')
    plt.savefig('metric_mean_fx.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: metric_mean_fx.png\n")
    plt.show()

# Example 2: Distribution plot
if 'mean_fy' in metric_names:
    print("2. Distribution of Focal Length (fy) across temporal indices")
    p = plot_metric_distribution(df, 'mean_fy')
    plt.savefig('distribution_mean_fy.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: distribution_mean_fy.png\n")
    plt.show()

# Example 3: Heatmap
if 'mean_cx' in metric_names:
    print("3. Heatmap of Principal Point X (cx)")
    p = plot_heatmap(df, 'mean_cx')
    plt.savefig('heatmap_mean_cx.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: heatmap_mean_cx.png\n")
    plt.show()

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================

print(f"{'='*80}")
print("CORRELATION ANALYSIS:")
print(f"{'='*80}\n")

# Correlation between temporal index and each metric
correlations = {}
for metric in metric_names:
    corr = df['temporal_idx'].corr(df[metric])
    correlations[metric] = corr
    print(f"{metric:30s}: {corr:+.4f}")

print(f"\n{'='*80}")
print("INTERPRETATION:")
print(f"{'='*80}")
print("""
Correlation closer to +1.0: Metric INCREASES with temporal index
Correlation closer to -1.0: Metric DECREASES with temporal index  
Correlation close to  0.0: Metric is INDEPENDENT of temporal index

A correlation near 0 suggests the model is robust to temporal position changes.
A strong correlation indicates temporal position affects predictions.
""")

# ============================================================================
# 7. RECOMMENDATIONS
# ============================================================================

print(f"{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")

print("Next steps:")
print("1. Examine the generated plots for patterns")
print("2. Look for metrics with high correlation to temporal index")
print("3. Compare behavior at different temporal positions (start/middle/end)")
print("4. Check if effects are consistent across samples or sample-dependent")
print("5. Investigate outliers or unexpected behaviors\n")

# ============================================================================
# 8. CUSTOM QUERIES (Modify as needed)
# ============================================================================

print("CUSTOM ANALYSIS EXAMPLES:")
print("-" * 80)

# Find the temporal index with best performance on a metric
if 'mean_fx' in metric_names:
    best_idx = df.groupby('temporal_idx')['mean_fx'].mean().idxmin()
    best_val = df.groupby('temporal_idx')['mean_fx'].mean().min()
    print(f"\nBest mean_fx at temporal index: {best_idx} (value: {best_val:.6f})")

# Find samples with high variance
if 'mean_fy' in metric_names:
    sample_vars = df.groupby('sample_idx')['mean_fy'].std()
    most_variable = sample_vars.idxmax()
    print(f"Most variable sample: {most_variable} (std: {sample_vars[most_variable]:.6f})")

print("\n" + "="*80)
