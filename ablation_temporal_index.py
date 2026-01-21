"""
Ablation Study: Temporal Index Manipulation for Target Images

This script runs ablation studies on RayZer by changing the temporal index (positional embedding)
of a specific target image and observing how it affects camera prediction and reconstruction quality.

Usage:
    python ablation_temporal_index.py --model_ckpt path/to/checkpoint --data_config path/to/data_config \
        --target_image_idx 0 --num_index_values 20 --output_dir ./ablation_results
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
import json
from tqdm import tqdm

# You'll need to adjust imports based on your actual structure
from model.rayzer import RayZer
from utils.data_utils import get_data_loader  # Adjust based on your data loading


def load_config_and_model(config_path, checkpoint_path, device='cuda'):
    """Load configuration and initialize model with checkpoint."""
    # Assuming you have a config loading utility
    # This is a placeholder - adjust based on your actual config loading
    from configs import load_config  # Adjust import path
    
    config = load_config(config_path)
    model = RayZer(config)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
    
    model = model.to(device)
    model.eval()
    return model, config


def run_ablation_inference(model, data, target_image_idx, temporal_idx, device='cuda'):
    """
    Run inference with a specific temporal index for the target image.
    
    Args:
        model: RayZer model
        data: Input batch data
        target_image_idx: Which target image to ablate (0-indexed)
        temporal_idx: Temporal index to assign to that image
        device: Device to run on
        
    Returns:
        Dictionary with results including rendered images and predictions
    """
    # Set ablation configuration
    model.set_ablation_config(target_image_idx, temporal_idx)
    
    with torch.no_grad():
        results = model(data, create_visual=False, render_video=False)
    
    # Clear ablation after inference
    model.clear_ablation_config()
    
    return results


def compute_reconstruction_metrics(rendered, target, mask=None):
    """
    Compute reconstruction quality metrics.
    
    Args:
        rendered: Rendered images [b, v, c, h, w]
        target: Target images [b, v, c, h, w]
        mask: Optional mask for region of interest
        
    Returns:
        Dictionary of metrics (PSNR, SSIM, MAE, etc.)
    """
    # Ensure images are in [0, 1] range
    rendered = torch.clamp(rendered, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    if mask is not None:
        rendered = rendered * mask
        target = target * mask
    
    # MSE and PSNR
    mse = torch.mean((rendered - target) ** 2)
    psnr = -10 * torch.log10(mse + 1e-10)
    
    # MAE
    mae = torch.mean(torch.abs(rendered - target))
    
    # Simple SSIM approximation (mean correlation)
    # For full SSIM, use torchmetrics or skimage
    
    metrics = {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'mae': mae.item(),
    }
    
    return metrics


def run_ablation_study(model, data_loader, config, args, device='cuda'):
    """
    Run complete ablation study over multiple temporal indices.
    
    Args:
        model: RayZer model
        data_loader: DataLoader for batches
        config: Model configuration
        args: Command line arguments
        device: Device to run on
        
    Returns:
        Dictionary with all results
    """
    results_all = {
        'config': vars(args),
        'results_per_index': {}
    }
    
    # Generate temporal indices to test
    # You can use: linspace, random, or specific values
    num_images = 24  # You mentioned 24 images total, adjust if needed
    if args.index_strategy == 'linspace':
        temporal_indices = np.linspace(0, num_images - 1, args.num_index_values).astype(int)
    elif args.index_strategy == 'random':
        temporal_indices = np.random.randint(0, num_images, args.num_index_values)
    else:
        temporal_indices = np.array(args.temporal_indices)
    
    print(f"Testing temporal indices: {temporal_indices}")
    
    # For now, use first batch from data loader
    model.eval()
    
    for batch_data in data_loader:
        # Move batch to device
        batch_data_device = {}
        for key, val in batch_data.items():
            if isinstance(val, torch.Tensor):
                batch_data_device[key] = val.to(device)
            else:
                batch_data_device[key] = val
        
        # Run inference for each temporal index
        for temporal_idx in tqdm(temporal_indices, desc=f"Testing temporal indices"):
            try:
                results = run_ablation_inference(
                    model, 
                    batch_data_device, 
                    args.target_image_idx, 
                    int(temporal_idx),
                    device
                )
                
                # Compute metrics for the target image
                rendered_image = results['render']  # [b, v, c, h, w]
                target_image = results['target'].image  # [b, v, c, h, w]
                
                # Get only the ablated target image
                # Note: This assumes results use the same indexing as the model
                metrics = compute_reconstruction_metrics(rendered_image, target_image)
                
                # Store results
                results_all['results_per_index'][int(temporal_idx)] = {
                    'metrics': metrics,
                    'cam_info': results['c2w'].detach().cpu().numpy().tolist() if 'c2w' in results else None,
                }
                
                print(f"  Temporal idx {temporal_idx}: PSNR={metrics['psnr']:.2f}, MAE={metrics['mae']:.4f}")
                
            except Exception as e:
                print(f"  Error with temporal idx {temporal_idx}: {e}")
                results_all['results_per_index'][int(temporal_idx)] = {'error': str(e)}
        
        break  # Only process first batch for initial study
    
    return results_all


def main():
    parser = argparse.ArgumentParser(description='Temporal Index Ablation Study')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_config', type=str, required=True, help='Path to data config')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--target_image_idx', type=int, default=0, help='Which target image to ablate (0-indexed)')
    parser.add_argument('--num_index_values', type=int, default=20, help='Number of temporal indices to test')
    parser.add_argument('--index_strategy', type=str, choices=['linspace', 'random', 'custom'], default='linspace',
                        help='Strategy for selecting temporal indices')
    parser.add_argument('--temporal_indices', type=int, nargs='*', default=None, help='Custom temporal indices')
    parser.add_argument('--output_dir', type=str, default='./ablation_results', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and config
    print("Loading model and config...")
    model, config = load_config_and_model(args.data_config, args.model_ckpt, device)
    print(f"Model loaded successfully")
    
    # Load data
    print("Loading data...")
    # You'll need to adjust this based on your actual data loading function
    # data_loader = get_data_loader(args.data_path, config, args.batch_size, args.num_workers, split='val')
    # For now, this is a placeholder - adjust to your actual data loading
    
    # Run ablation study
    print(f"\nRunning ablation study...")
    print(f"  Target image index: {args.target_image_idx}")
    print(f"  Number of temporal indices: {args.num_index_values}")
    print(f"  Output directory: {args.output_dir}")
    
    # results = run_ablation_study(model, data_loader, config, args, device)
    
    # Save results
    # output_path = os.path.join(args.output_dir, 'ablation_results.json')
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=2)
    # print(f"\nResults saved to {output_path}")
    
    print("\n=== Note ===")
    print("You need to implement data loading. Replace the data_loader placeholder with your actual data loading.")
    print("Key modifications needed:")
    print("1. Import your actual data loading function")
    print("2. Load configuration using your config loader")
    print("3. Adjust metrics computation based on your output format")


if __name__ == '__main__':
    main()
