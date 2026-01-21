"""
Simple Ablation Study Example

This is a minimal working example showing how to run temporal index ablation.
Adapt this to your specific data loading and config setup.
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Adjust imports based on your actual project structure
# from model.rayzer import RayZer
# from utils.data_utils import load_data_batch


def compute_psnr(rendered, target, max_val=1.0):
    """Compute PSNR between rendered and target images."""
    rendered = torch.clamp(rendered, 0, max_val)
    target = torch.clamp(target, 0, max_val)
    mse = torch.mean((rendered - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse + 1e-10)
    return psnr.item()


def run_simple_ablation(
    model,
    data_batch,
    target_image_idx,
    temporal_indices,
    device='cuda'
):
    """
    Run ablation study with specified temporal indices.
    
    Args:
        model: RayZer model in eval mode
        data_batch: Input batch with 'image' and other required fields
        target_image_idx: Which target image to ablate (0-indexed in target set)
        temporal_indices: List or array of temporal indices to test
        device: Device to run on
        
    Returns:
        Dictionary with results for each index
    """
    model.eval()
    results = {}
    
    for temporal_idx in temporal_indices:
        # Set ablation configuration
        model.set_ablation_config(target_image_idx, int(temporal_idx))
        
        try:
            with torch.no_grad():
                output = model(data_batch)
            
            # Extract results
            rendered = output['render']  # [batch_size, num_target_views, 3, h, w]
            target_images = output['target'].image  # [batch_size, num_target_views, 3, h, w]
            cam_predictions = output['c2w']  # [batch_size, num_all_views, 4, 4]
            
            # Compute metrics on the ablated target image
            # Note: You may need to adjust indexing based on your output format
            rendered_target = rendered[:, target_image_idx]  # [batch, 3, h, w]
            gt_target = target_images[:, target_image_idx]   # [batch, 3, h, w]
            
            psnr = compute_psnr(rendered_target, gt_target)
            
            results[int(temporal_idx)] = {
                'psnr': psnr,
                'rendered': rendered.detach().cpu(),
                'camera_predictions': cam_predictions.detach().cpu(),
            }
            
            print(f"✓ Temporal index {temporal_idx:2d}: PSNR = {psnr:6.2f} dB")
            
        except Exception as e:
            print(f"✗ Temporal index {temporal_idx:2d}: Error - {e}")
            results[int(temporal_idx)] = {'error': str(e)}
        
        finally:
            # Always clear ablation config
            model.clear_ablation_config()
    
    return results


def plot_ablation_results(results, output_path=None):
    """
    Plot PSNR vs temporal index.
    
    Args:
        results: Dictionary from run_simple_ablation
        output_path: Path to save plot (optional)
    """
    indices = []
    psnrs = []
    
    for idx in sorted(results.keys()):
        if 'psnr' in results[idx]:
            indices.append(idx)
            psnrs.append(results[idx]['psnr'])
    
    if not indices:
        print("No valid results to plot")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(indices, psnrs, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Temporal Index', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Ablation Study: Impact of Temporal Index on Reconstruction Quality', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Highlight best index
    best_idx = indices[np.argmax(psnrs)]
    best_psnr = max(psnrs)
    plt.plot(best_idx, best_psnr, 'r*', markersize=15, label=f'Best: idx={best_idx}, PSNR={best_psnr:.2f}')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def save_results(results, output_dir):
    """
    Save ablation results to JSON and visualizations.
    
    Args:
        results: Dictionary from run_simple_ablation
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    summary = {}
    for idx in sorted(results.keys()):
        if 'psnr' in results[idx]:
            summary[str(idx)] = {
                'psnr': results[idx]['psnr'],
            }
        else:
            summary[str(idx)] = {'error': results[idx].get('error', 'Unknown error')}
    
    summary_path = output_dir / 'ablation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    # Plot results
    plot_path = output_dir / 'ablation_psnr_plot.png'
    plot_ablation_results(results, plot_path)
    
    # Save best result info
    best_idx = max(
        ((idx, results[idx]['psnr']) for idx in results if 'psnr' in results[idx]),
        key=lambda x: x[1],
        default=(None, 0)
    )[0]
    
    best_info = {
        'best_temporal_index': best_idx,
        'best_psnr': results[best_idx]['psnr'] if best_idx is not None else None,
        'interpretation': "This temporal index produces the best reconstruction quality.",
    }
    
    best_path = output_dir / 'best_result.json'
    with open(best_path, 'w') as f:
        json.dump(best_info, f, indent=2)
    print(f"Best result info saved to {best_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # This is a template. You need to fill in the actual data loading
    
    print("=" * 70)
    print("Temporal Index Ablation Study")
    print("=" * 70)
    
    # Step 1: Load model
    print("\n[1/4] Loading model...")
    # model = RayZer(config)
    # model.load_ckpt('path/to/checkpoint.pt')
    # model = model.cuda().eval()
    print("  ✓ Model loaded (placeholder)")
    
    # Step 2: Load data
    print("\n[2/4] Loading data...")
    # data_batch = load_data_batch('path/to/data')
    # Move to GPU
    # data_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
    #               for k, v in data_batch.items()}
    print("  ✓ Data loaded (placeholder)")
    
    # Step 3: Run ablation
    print("\n[3/4] Running ablation study...")
    print("  Testing temporal indices: 0, 1, 2, ..., 23 (24 total)")
    
    temporal_indices = np.arange(0, 24)  # Test all 24 possible indices
    # results = run_simple_ablation(
    #     model,
    #     data_batch,
    #     target_image_idx=0,  # Ablate first target image
    #     temporal_indices=temporal_indices,
    # )
    
    # Placeholder results
    results = {
        i: {'psnr': 25 + np.sin(i/5) * 3} for i in range(24)
    }
    print("  ✓ Ablation study completed")
    
    # Step 4: Save results
    print("\n[4/4] Saving results...")
    save_results(results, './ablation_results')
    print("  ✓ Results saved")
    
    print("\n" + "=" * 70)
    print("Ablation study complete! Check ./ablation_results/ for outputs.")
    print("=" * 70)


# ============================================================================
# INTERPRETATION GUIDE
# ============================================================================

"""
After running ablation, interpret results as follows:

SCENARIO 1: Sharp peak at original index
- Indicates strong dependence on temporal information
- Model learned that temporal order matters
- Original index was correct → high PSNR

SCENARIO 2: Broad peak across multiple indices
- Model is robust to temporal index variation
- Less sensitive to "when" the image was taken
- May indicate the camera predictor is well-constrained by geometry

SCENARIO 3: Low and flat PSNR across all indices
- Model doesn't strongly use temporal information for this image
- Temporal PE has minimal impact
- Other signals (geometry, appearance) dominate

SCENARIO 4: Gradually improving/degrading PSNR
- Smooth dependence on temporal index
- May indicate temporal information encodes some continuous property
- Could be related to scene dynamics or view consistency

TIPS:
- Run ablation for different target images to compare
- Check if the pattern is consistent across batches
- Compare with baseline (no ablation) PSNR
- Visualize rendered images for top/bottom indices
"""
