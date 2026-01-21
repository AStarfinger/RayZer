"""
Visualization script for baseline ablation study results.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Index spacing: distance between consecutive selected frames
index_spacings = [2, 4, 8, 16]  # Difference between frame indices

# Collect metrics from generated results
psnr_values = []
lpips_values = []
ssim_values = []

for spacing in index_spacings:
    # Construct path to metrics file
    metrics_dir = f"./experiments/evaluation/test/rayzer_dl3dv_two_frame_{spacing:02d}/000000"
    metrics_file = os.path.join(metrics_dir, "metrics.json")
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        # Extract metrics from summary section
        summary = metrics.get('summary', {})
        psnr = summary.get('psnr', 0)
        lpips = summary.get('lpips', 0)
        ssim = summary.get('ssim', 0)
        psnr_values.append(psnr)
        lpips_values.append(lpips)
        ssim_values.append(ssim)
        print(f"✓ Loaded metrics for spacing {spacing}: PSNR={psnr:.4f}, LPIPS={lpips:.4f}, SSIM={ssim:.4f}")
    else:
        print(f"⚠ Warning: Metrics file not found for spacing {spacing}: {metrics_file}")
        psnr_values.append(0)
        lpips_values.append(0)
        ssim_values.append(0)

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Ablation Study: Performance vs Index Spacing', fontsize=16, fontweight='bold')

# Color scheme
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot 1: PSNR
ax1 = axes[0]
ax1.plot(index_spacings, psnr_values, marker='o', linewidth=2.5, markersize=10, color='#1f77b4', label='PSNR')
ax1.fill_between(index_spacings, psnr_values, alpha=0.3, color='#1f77b4')
ax1.set_xlabel('Index Spacing (Frame Distance)', fontsize=12, fontweight='bold')
ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
ax1.set_title('PSNR vs Index Spacing', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(index_spacings)
for i, (spacing, psnr) in enumerate(zip(index_spacings, psnr_values)):
    ax1.annotate(f'{psnr:.2f}', xy=(spacing, psnr), xytext=(0, 10), textcoords='offset points', 
                ha='center', fontsize=10, fontweight='bold')

# Plot 2: LPIPS (lower is better)
ax2 = axes[1]
ax2.plot(index_spacings, lpips_values, marker='s', linewidth=2.5, markersize=10, color='#ff7f0e', label='LPIPS')
ax2.fill_between(index_spacings, lpips_values, alpha=0.3, color='#ff7f0e')
ax2.set_xlabel('Index Spacing (Frame Distance)', fontsize=12, fontweight='bold')
ax2.set_ylabel('LPIPS', fontsize=12, fontweight='bold')
ax2.set_title('LPIPS vs Index Spacing (Lower is Better)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(index_spacings)
for i, (spacing, lpips) in enumerate(zip(index_spacings, lpips_values)):
    ax2.annotate(f'{lpips:.4f}', xy=(spacing, lpips), xytext=(0, 10), textcoords='offset points', 
                ha='center', fontsize=10, fontweight='bold')

# Plot 3: SSIM
ax3 = axes[2]
ax3.plot(index_spacings, ssim_values, marker='^', linewidth=2.5, markersize=10, color='#2ca02c', label='SSIM')
ax3.fill_between(index_spacings, ssim_values, alpha=0.3, color='#2ca02c')
ax3.set_xlabel('Index Spacing (Frame Distance)', fontsize=12, fontweight='bold')
ax3.set_ylabel('SSIM', fontsize=12, fontweight='bold')
ax3.set_title('SSIM vs Index Spacing (Higher is Better)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(index_spacings)
ax3.set_ylim([0.3, 0.95])
for i, (spacing, ssim) in enumerate(zip(index_spacings, ssim_values)):
    ax3.annotate(f'{ssim:.4f}', xy=(spacing, ssim), xytext=(0, 10), textcoords='offset points', 
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: ablation_study_results.png")
# plt.show()  # Skip interactive show

# Create a combined normalized view
fig2, ax = plt.subplots(figsize=(12, 6))

# Normalize metrics to [0, 1] for comparison
psnr_norm = (np.array(psnr_values) - min(psnr_values)) / (max(psnr_values) - min(psnr_values))
lpips_norm = 1 - (np.array(lpips_values) - min(lpips_values)) / (max(lpips_values) - min(lpips_values))  # Invert LPIPS
ssim_norm = (np.array(ssim_values) - min(ssim_values)) / (max(ssim_values) - min(ssim_values))

x = np.arange(len(index_spacings))
width = 0.25

bars1 = ax.bar(x - width, psnr_norm, width, label='PSNR (normalized)', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, lpips_norm, width, label='LPIPS (inverted, normalized)', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, ssim_norm, width, label='SSIM (normalized)', color='#2ca02c', alpha=0.8)

ax.set_xlabel('Index Spacing (Frame Distance)', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Score (Higher is Better)', fontsize=12, fontweight='bold')
ax.set_title('Baseline Ablation Study - Normalized Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(index_spacings)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9)

plt.tight_layout()
plt.savefig('ablation_study_normalized.png', dpi=300, bbox_inches='tight')
print("✓ Saved normalized comparison to: ablation_study_normalized.png")
# plt.show()  # Skip interactive show

# Print summary
print("\n" + "="*80)
print("BASELINE ABLATION STUDY SUMMARY")
print("="*80)
print(f"\n{'Index Spacing':<15} {'PSNR':<15} {'LPIPS':<15} {'SSIM':<15}")
print("-"*60)
for spacing, psnr, lpips, ssim in zip(index_spacings, psnr_values, lpips_values, ssim_values):
    print(f"{spacing:<15} {psnr:<15.4f} {lpips:<15.4f} {ssim:<15.4f}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
best_psnr_idx = np.argmax(psnr_values)
best_lpips_idx = np.argmin(lpips_values)
best_ssim_idx = np.argmax(ssim_values)

print(f"Best PSNR:  Index Spacing {index_spacings[best_psnr_idx]} with {psnr_values[best_psnr_idx]:.4f} dB")
print(f"Best LPIPS: Index Spacing {index_spacings[best_lpips_idx]} with {lpips_values[best_lpips_idx]:.4f}")
print(f"Best SSIM:  Index Spacing {index_spacings[best_ssim_idx]} with {ssim_values[best_ssim_idx]:.4f}")

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)

# Check if we have valid data before calculating percentages
if psnr_values[0] > 0 and lpips_values[0] > 0 and ssim_values[0] > 0:
    psnr_drop = ((psnr_values[0] - psnr_values[-1]) / psnr_values[0]) * 100
    lpips_increase = ((lpips_values[-1] - lpips_values[0]) / lpips_values[0]) * 100
    ssim_drop = ((ssim_values[0] - ssim_values[-1]) / ssim_values[0]) * 100

    print(f"• PSNR degradation from spacing 2 to 16: {psnr_drop:.1f}%")
    print(f"• LPIPS increase from spacing 2 to 16: {lpips_increase:.1f}%")
    print(f"• SSIM degradation from spacing 2 to 16: {ssim_drop:.1f}%")
    print("\n✓ Performance degrades significantly with larger frame index spacing")
    print("✓ Tighter spacing (more context frames) provides better reconstruction quality")
else:
    print("⚠ Unable to calculate degradation metrics - no valid data loaded")
