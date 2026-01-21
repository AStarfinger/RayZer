"""
Ablation Study: Temporal Index Manipulation for Target Images

This script runs ablation studies on RayZer by changing the temporal index (positional embedding)
of a specific target image and observing how it affects camera prediction and reconstruction quality.

The temporal index controls the positional embedding of an image in the sequence. By changing it,
we can test whether the model learns to predict camera poses independently or relies on the temporal
sequence structure.

Usage:
    python ablation_temporal_index_study.py \
        --config configs/rayzer_dl3dv.yaml \
        --checkpoint model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \
        --target_image_idx 0 \
        --test_indices 0,5,10,15,20 \
        --output_dir ./ablation_results_temporal \
        --num_samples 10
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
import json
from tqdm import tqdm
import importlib
import yaml

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def init_distributed_single_process():
    """
    Initialize distributed backend for single-process inference.
    This is needed when the model expects distributed to be initialized.
    """
    if not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(
            backend=backend,
            init_method='tcp://127.0.0.1:29500',
            rank=0,
            world_size=1
        )


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return edict(config_dict)


def load_model(config, checkpoint_path, device):
    """Load RayZer model from checkpoint."""
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    try:
        init_distributed_single_process()
    except Exception as e:
        print(f"Warning: Could not initialize distributed: {e}")
    
    module, class_name = config.model.class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module).__dict__[class_name]
    
    model = ModelClass(config).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    
    return model


def get_data_loader(config, dataset_split='val', start_idx=0, end_idx=None):
    """
    Load data for ablation study.
    
    Args:
        config: Configuration object
        dataset_split: Dataset split to use (default: 'val')
        start_idx: Start index in dataset (default: 0)
        end_idx: End index in dataset (default: None = to end)
    """
    dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    
    dataset = Dataset(config)
    
    # Create a subset if indices are specified
    if end_idx is not None or start_idx > 0:
        end_idx = end_idx if end_idx is not None else len(dataset)
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(start_idx, min(end_idx, len(dataset))))
        print(f"Using dataset samples from index {start_idx} to {end_idx-1}")
    
    # Use sequential sampler for reproducible results in ablation
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Batch size 1 for ablation studies
        shuffle=False,
        num_workers=0,  # Single worker for reproducibility
        pin_memory=True,
        drop_last=False,
    )
    
    return dataloader


def compute_psnr(predicted, target, data_range=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    Normalizes both images to [0, 1] range for consistent comparison.
    
    Args:
        predicted: Predicted images [b, v, c, h, w] or [b, c, h, w]
        target: Target images [b, v, c, h, w] or [b, c, h, w]
        data_range: Range of image values (default 1.0)
    
    Returns:
        dict with PSNR metrics
    """
    # Ensure tensors are on CPU
    predicted = predicted.detach().cpu()
    target = target.detach().cpu()
    
    # Normalize both to [0, 1] range
    pred_min, pred_max = predicted.min(), predicted.max()
    target_min, target_max = target.min(), target.max()
    
    # Normalize predicted
    if pred_min < -0.5:  # It's in [-1, 1] range
        predicted = (predicted + 1) / 2.0
    elif pred_max > 1.0:  # It might be in a different range
        predicted = (predicted - pred_min) / (pred_max - pred_min + 1e-6)
    
    # Normalize target
    if target_min < -0.5:  # It's in [-1, 1] range
        target = (target + 1) / 2.0
    elif target_max > 1.0:  # It might be in a different range
        target = (target - target_min) / (target_max - target_min + 1e-6)
    
    # Clamp to [0, 1]
    predicted = torch.clamp(predicted, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    # Compute MSE
    mse = torch.mean((predicted - target) ** 2, dim=tuple(range(1, predicted.ndim)))
    
    # Avoid log(0) by clamping
    mse = torch.clamp(mse, min=1e-10)
    
    # Compute PSNR: 20 * log10(1.0 / sqrt(MSE)) for [0, 1] range
    psnr_values = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
    
    return {
        'mean_psnr': psnr_values.mean().item(),
        'std_psnr': psnr_values.std().item(),
        'min_psnr': psnr_values.min().item(),
        'max_psnr': psnr_values.max().item(),
    }


def compute_camera_metrics(predicted_c2w, predicted_fxfycxcy, gt_c2w=None, gt_fxfycxcy=None):
    """
    Compute camera pose metrics.
    
    Args:
        predicted_c2w: Predicted camera-to-world matrices [b*v, 4, 4]
        predicted_fxfycxcy: Predicted camera intrinsics [b*v, 4]
        gt_c2w: Ground truth c2w (if available)
        gt_fxfycxcy: Ground truth intrinsics (if available)
    
    Returns:
        dict with metrics
    """
    metrics = {}
    
    # Compute rotation matrix frobenius norm (how close to identity)
    R = predicted_c2w[:, :3, :3]
    metrics['rotation_matrix_norm'] = torch.norm(R, p='fro', dim=(1, 2)).mean().item()
    
    # Compute translation magnitude
    t = predicted_c2w[:, :3, 3]
    metrics['translation_magnitude'] = torch.norm(t, dim=1).mean().item()
    
    # Compute focal length stats
    metrics['mean_fx'] = predicted_fxfycxcy[:, 0].mean().item()
    metrics['mean_fy'] = predicted_fxfycxcy[:, 1].mean().item()
    metrics['mean_cx'] = predicted_fxfycxcy[:, 2].mean().item()
    metrics['mean_cy'] = predicted_fxfycxcy[:, 3].mean().item()
    
    return metrics


def save_rendered_images(rendered_images, output_dir, batch_idx, temporal_idx, target_image_idx):
    """
    Save the ablated target image to disk.
    
    Args:
        rendered_images: Tensor of rendered images [b, v_target, c, h, w]
        output_dir: Base output directory
        batch_idx: Batch index
        temporal_idx: Temporal index being tested
        target_image_idx: Which target image was ablated
    
    Returns:
        Path to saved image
    """
    import torchvision.transforms.functional as F
    
    # Create flat directory for all ablated images
    ablated_dir = os.path.join(output_dir, 'ablated_images')
    os.makedirs(ablated_dir, exist_ok=True)
    
    # Extract only the ablated target image
    if len(rendered_images.shape) == 5:  # [b, v_target, c, h, w]
        img_tensor = rendered_images[0, target_image_idx]  # [c, h, w]
    elif len(rendered_images.shape) == 4:  # [b, c, h, w] - shouldn't happen but handle it
        img_tensor = rendered_images[0]
    else:
        img_tensor = rendered_images
    
    # Create filename: batch_001_temporal_idx_005.png
    filename = f'batch_{batch_idx:03d}_temporal_idx_{temporal_idx:03d}.png'
    img_path = os.path.join(ablated_dir, filename)
    
    # Save image
    save_image_tensor(img_tensor, img_path)
    
    return img_path


def save_image_tensor(img_tensor, save_path):
    """
    Save a single image tensor to disk.
    
    Args:
        img_tensor: Image tensor [c, h, w] with values in [0, 1] or [-1, 1]
        save_path: Path to save the image
    """
    import torchvision.transforms.functional as F
    
    # Ensure tensor is on CPU
    img_tensor = img_tensor.detach().cpu()
    
    # Handle different value ranges
    if img_tensor.min() < 0:
        # Range [-1, 1], convert to [0, 1]
        img_tensor = (img_tensor + 1) / 2
    
    # Clamp to valid range
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to PIL Image and save
    if img_tensor.shape[0] == 1:
        # Grayscale
        img_tensor = img_tensor.squeeze(0)
        pil_img = F.to_pil_image(img_tensor)
    else:
        # RGB or RGBA
        pil_img = F.to_pil_image(img_tensor)
    
    pil_img.save(save_path)


def run_ablation_on_batch(model, batch, target_image_idx, temporal_idx, device, use_amp=True, amp_dtype='float16'):
    """
    Run model inference with a specific temporal index for a target image.
    
    Args:
        model: RayZer model
        batch: Input batch data
        target_image_idx: Which target image to ablate (0-indexed within target set)
        temporal_idx: What temporal index to assign to this image
        device: torch device (string or torch.device)
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP ('float16' or 'bfloat16')
    
    Returns:
        result dict with camera predictions
    """
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = torch.device(device)
    
    # Set ablation configuration
    model.set_ablation_config(target_image_idx, temporal_idx)
    
    # Move batch to device
    batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
    
    # Map dtype string to torch dtype
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    amp_dtype_torch = dtype_map.get(amp_dtype, torch.float16)
    
    # Determine device type for autocast
    device_type = device.type if hasattr(device, 'type') else ('cuda' if 'cuda' in str(device) else 'cpu')
    
    # Run inference with rendering enabled
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type=device_type, 
                              dtype=amp_dtype_torch, enabled=True):
                result = model(batch_device, create_visual=False, render_video=False)
        else:
            result = model(batch_device, create_visual=False, render_video=False)
    
    # Clear ablation config
    model.clear_ablation_config()
    
    return result


def run_ablation_study(config, checkpoint_path, target_image_idx, test_temporal_indices, 
                       output_dir, num_samples=None, device='cuda', use_amp=True, amp_dtype='float16',
                       start_scene_idx=0, end_scene_idx=None):
    """
    Run complete ablation study.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to model checkpoint
        target_image_idx: Which target image to ablate
        test_temporal_indices: List of temporal indices to test
        output_dir: Directory to save results
        num_samples: Number of samples to process (None = all)
        device: torch device
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP ('float16' or 'bfloat16')
        start_scene_idx: Start index for scenes (0-indexed)
        end_scene_idx: End index for scenes (None = to end)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(config, checkpoint_path, device)
    model.eval()
    
    # Load data with optional scene range
    dataloader = get_data_loader(config, start_idx=start_scene_idx, end_idx=end_scene_idx)
    
    # Results storage
    ablation_results = {
        'config': {
            'target_image_idx': target_image_idx,
            'test_temporal_indices': test_temporal_indices,
            'use_amp': use_amp,
            'amp_dtype': amp_dtype,
        },
        'samples': []
    }
    
    print(f"\nRunning ablation study:")
    print(f"  Target image index: {target_image_idx}")
    print(f"  Temporal indices to test: {test_temporal_indices}")
    print(f"  Output directory: {output_dir}")
    print(f"  AMP enabled: {use_amp} (dtype: {amp_dtype})\n")
    
    # Process samples
    num_processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing samples")):
        if num_samples is not None and num_processed >= num_samples:
            break
        
        sample_results = {
            'batch_idx': batch_idx,
            'temporal_ablations': {}
        }
        
        # Run ablation with each temporal index
        for temporal_idx in test_temporal_indices:
            try:
                result = run_ablation_on_batch(
                    model, batch, target_image_idx, temporal_idx, device, 
                    use_amp=use_amp, amp_dtype=amp_dtype
                )
                
                # Extract camera predictions
                c2w = result['c2w']  # [b, v_all, 4, 4]
                fxfycxcy = result['fxfycxcy']  # [b*v_all, 4]
                rendered_images = result['render']  # [b, v_target, c, h, w]
                
                # Compute metrics
                metrics = compute_camera_metrics(c2w.reshape(-1, 4, 4), fxfycxcy)
                
                # Compute PSNR if ground truth is available
                if 'image' in batch and 'target_idx' in result:
                    # Extract only the target images from the batch
                    all_images = batch['image'] * 2.0 - 1.0  # Convert to [-1, 1] range, [b, v_all, c, h, w]
                    target_idx = result['target_idx']
                    
                    # Move target_idx to same device as all_images if needed
                    if isinstance(target_idx, torch.Tensor):
                        target_idx = target_idx.to(all_images.device)
                    
                    # Extract target images using target_idx
                    b = all_images.shape[0]
                    batch_idx_tensor = torch.arange(b, device=all_images.device).unsqueeze(1)
                    target_images = all_images[batch_idx_tensor, target_idx]  # [b, v_target, c, h, w]
                    
                    psnr_metrics = compute_psnr(rendered_images, target_images)
                    metrics.update(psnr_metrics)
                
                # Save rendered images
                rendered_path = save_rendered_images(
                    rendered_images, output_dir, batch_idx, temporal_idx, target_image_idx
                )
                
                sample_results['temporal_ablations'][str(temporal_idx)] = {
                    'metrics': metrics,
                    'c2w_sample': c2w[0, 0].detach().cpu().numpy().tolist(),  # Sample one pose
                    'fxfycxcy_sample': fxfycxcy[0].detach().cpu().numpy().tolist(),
                    'ablated_image': rendered_path,  # Path to ablated image
                }
            except Exception as e:
                print(f"Error processing batch {batch_idx} with temporal_idx {temporal_idx}: {e}")
                sample_results['temporal_ablations'][str(temporal_idx)] = {'error': str(e)}
        
        ablation_results['samples'].append(sample_results)
        num_processed += 1
        torch.cuda.empty_cache()
    
    # Save results
    results_file = os.path.join(output_dir, 'ablation_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy arrays in metrics to lists for JSON serialization
        json.dump(ablation_results, f, indent=2, default=str)
    
    # Export PSNR values for easy plotting
    export_psnr_values(ablation_results, output_dir)
    
    print(f"\nAblation study complete!")
    print(f"Results saved to: {results_file}")
    
    # Print summary statistics
    print_summary_stats(ablation_results)
    
    return ablation_results


def print_summary_stats(ablation_results):
    """Print summary statistics of the ablation study."""
    # Call PSNR summary if available
    try:
        print_psnr_summary(ablation_results)
    except Exception as e:
        print(f"Note: Could not print PSNR summary: {e}")


def print_psnr_summary(ablation_results):
    """Print PSNR summary statistics if available."""
    temporal_indices = ablation_results['config']['test_temporal_indices']
    samples = ablation_results['samples']
    
    # Check if PSNR data exists
    has_psnr = False
    for sample in samples:
        for temporal_idx_str, ablation_data in sample['temporal_ablations'].items():
            if 'metrics' in ablation_data and 'mean_psnr' in ablation_data['metrics']:
                has_psnr = True
                break
        if has_psnr:
            break
    
    if not has_psnr:
        return
    
    # Aggregate PSNR metrics across samples
    psnr_by_temporal_idx = {str(idx): [] for idx in temporal_indices}
    
    for sample in samples:
        for temporal_idx_str, ablation_data in sample['temporal_ablations'].items():
            if 'metrics' in ablation_data and 'mean_psnr' in ablation_data['metrics']:
                psnr_by_temporal_idx[temporal_idx_str].append(ablation_data['metrics']['mean_psnr'])
    
    # Print results
    print("\n" + "="*80)
    print("PSNR RESULTS (Peak Signal-to-Noise Ratio)")
    print("="*80)
    print(f"\n{'Temporal Index':<15} {'Mean PSNR':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print("-" * 80)
    
    for temporal_idx_str in [str(idx) for idx in temporal_indices]:
        if psnr_by_temporal_idx[temporal_idx_str]:
            values = psnr_by_temporal_idx[temporal_idx_str]
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"{temporal_idx_str:<15} {mean:<15.4f} {std:<15.4f} {min_val:<15.4f} {max_val:<15.4f}")
    
    print("="*80 + "\n")


def export_psnr_values(ablation_results, output_dir):
    """
    Export PSNR values to CSV and JSON for easy plotting with matplotlib.
    
    Args:
        ablation_results: Results dictionary from ablation study
        output_dir: Output directory for exported files
    """
    temporal_indices = ablation_results['config']['test_temporal_indices']
    samples = ablation_results['samples']
    
    # Check if PSNR data exists
    has_psnr = False
    for sample in samples:
        for temporal_idx_str, ablation_data in sample['temporal_ablations'].items():
            if 'metrics' in ablation_data and 'mean_psnr' in ablation_data['metrics']:
                has_psnr = True
                break
        if has_psnr:
            break
    
    if not has_psnr:
        print("Note: No PSNR data found in results")
        return
    
    # Aggregate PSNR metrics
    psnr_data = {}
    
    for sample_idx, sample in enumerate(samples):
        for temporal_idx in temporal_indices:
            temporal_idx_str = str(temporal_idx)
            
            if temporal_idx_str not in psnr_data:
                psnr_data[temporal_idx_str] = []
            
            if temporal_idx_str in sample['temporal_ablations']:
                ablation_data = sample['temporal_ablations'][temporal_idx_str]
                if 'metrics' in ablation_data and 'mean_psnr' in ablation_data['metrics']:
                    psnr = ablation_data['metrics']['mean_psnr']
                    psnr_data[temporal_idx_str].append({
                        'sample_idx': sample_idx,
                        'temporal_idx': temporal_idx,
                        'psnr': psnr
                    })
    
    # Export to CSV
    csv_file = os.path.join(output_dir, 'psnr_values.csv')
    with open(csv_file, 'w') as f:
        f.write('temporal_index,sample_index,psnr\n')
        for temporal_idx_str in sorted(psnr_data.keys(), key=lambda x: int(x)):
            for entry in psnr_data[temporal_idx_str]:
                f.write(f"{entry['temporal_idx']},{entry['sample_idx']},{entry['psnr']:.6f}\n")
    
    print(f"✓ PSNR values exported to: {csv_file}")
    
    # Export to JSON for reference
    json_file = os.path.join(output_dir, 'psnr_summary.json')
    summary = {}
    for temporal_idx_str in sorted(psnr_data.keys(), key=lambda x: int(x)):
        values = [entry['psnr'] for entry in psnr_data[temporal_idx_str]]
        if values:
            summary[int(temporal_idx_str)] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'samples': values
            }
    
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ PSNR summary exported to: {json_file}")


def print_summary_stats_old(ablation_results):
    """Print summary statistics of the ablation study."""
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    temporal_indices = ablation_results['config']['test_temporal_indices']
    samples = ablation_results['samples']
    
    # Aggregate metrics across samples
    aggregated = {str(idx): [] for idx in temporal_indices}
    
    for sample in samples:
        for temporal_idx_str, ablation_data in sample['temporal_ablations'].items():
            if 'metrics' in ablation_data:
                aggregated[temporal_idx_str].append(ablation_data['metrics'])
    
    # Print results
    print(f"\nTesting temporal indices: {temporal_indices}")
    print(f"Number of samples: {len(samples)}")
    print("\nCamera prediction metrics (averaged across samples):\n")
    
    print(f"{'Temporal Index':<15} {'Mean FX':<12} {'Mean FY':<12} {'Mean CX':<12} {'Mean CY':<12}")
    print("-" * 60)
    
    for temporal_idx_str in [str(idx) for idx in temporal_indices]:
        if aggregated[temporal_idx_str]:
            metrics_list = aggregated[temporal_idx_str]
            mean_fx = np.mean([m['mean_fx'] for m in metrics_list])
            mean_fy = np.mean([m['mean_fy'] for m in metrics_list])
            mean_cx = np.mean([m['mean_cx'] for m in metrics_list])
            mean_cy = np.mean([m['mean_cy'] for m in metrics_list])
            
            print(f"{temporal_idx_str:<15} {mean_fx:<12.4f} {mean_fy:<12.4f} {mean_cx:<12.4f} {mean_cy:<12.4f}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on temporal indices for RayZer"
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--target_image_idx', type=int, default=0,
        help='Which target image to ablate (0-indexed within target set)'
    )
    parser.add_argument(
        '--test_indices', type=str, default='0,5,10,15,20',
        help='Comma-separated temporal indices to test'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Directory to save ablation results'
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Number of samples to process (None = all)'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--use_amp', action='store_true', default=True,
        help='Use automatic mixed precision (default: True)'
    )
    parser.add_argument(
        '--no_amp', action='store_true',
        help='Disable automatic mixed precision'
    )
    parser.add_argument(
        '--amp_dtype', type=str, default='float16',
        help='Data type for AMP: float16 or bfloat16 (default: float16)'
    )
    parser.add_argument(
        '--start_scene_idx', type=int, default=0,
        help='Start index of scenes/samples to process (0-indexed, default: 0)'
    )
    parser.add_argument(
        '--end_scene_idx', type=int, default=None,
        help='End index of scenes/samples to process (exclusive, default: None = all)'
    )
    
    args = parser.parse_args()
    
    # Handle AMP flag
    use_amp = not args.no_amp
    
    # Parse test indices
    test_indices = [int(idx.strip()) for idx in args.test_indices.split(',')]
    
    # Load config
    config = load_config(args.config)
    config.inference = edict(config.get('inference', {}))
    config.inference.render_video = False
    config.inference.if_inference = True
    
    # Run ablation study
    run_ablation_study(
        config=config,
        checkpoint_path=args.checkpoint,
        target_image_idx=args.target_image_idx,
        test_temporal_indices=test_indices,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
        use_amp=use_amp,
        amp_dtype=args.amp_dtype,
        start_scene_idx=args.start_scene_idx,
        end_scene_idx=args.end_scene_idx
    )


if __name__ == '__main__':
    main()
