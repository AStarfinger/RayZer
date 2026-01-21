"""
Quick Guide: Running Temporal Index Ablation Studies

This guide shows how to use the ablation functionality built into RayZer.
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

import torch
from model.rayzer import RayZer

# Load your model
model = RayZer(config)
model.load_ckpt('path/to/checkpoint')
model.eval()

# Prepare your data batch (24 images: 16 input + 8 target)
data_batch = {
    'image': images,  # [batch_size, 24, 3, height, width]
    # ... other required fields
}

# ============================================================================
# EXAMPLE 1: Single inference with modified temporal index
# ============================================================================

# Let's say you want to change the 0th target image to have temporal index 5
target_image_idx = 0  # First target image (within target set)
new_temporal_idx = 5

model.set_ablation_config(target_image_idx, new_temporal_idx)

with torch.no_grad():
    results = model(data_batch)

# results['render'] contains the rendered images
# results['c2w'] contains predicted camera poses
# You can compare results['render'] vs target to see the impact

model.clear_ablation_config()  # Clear ablation settings

# ============================================================================
# EXAMPLE 2: Loop over multiple temporal indices
# ============================================================================

import numpy as np

target_image_idx = 0  # Ablate first target image
temporal_indices = np.linspace(0, 23, 20).astype(int)  # 20 values from 0 to 23

results_dict = {}

for temporal_idx in temporal_indices:
    model.set_ablation_config(target_image_idx, temporal_idx)
    
    with torch.no_grad():
        results = model(data_batch)
    
    # Store results for comparison
    rendered = results['render']  # [batch, 8, 3, h, w]
    target_image = results['target'].image  # [batch, 8, 3, h, w]
    
    # Compute metrics on the ablated target image
    # (results index corresponds to target image ordering)
    psnr = compute_psnr(rendered[:, target_image_idx], target_image[:, target_image_idx])
    
    results_dict[temporal_idx] = {
        'psnr': psnr,
        'rendered': rendered.detach().cpu(),
        'camera_predictions': results['c2w'].detach().cpu(),
    }
    
    model.clear_ablation_config()
    print(f"Temporal index {temporal_idx}: PSNR = {psnr:.2f}")

# ============================================================================
# EXAMPLE 3: Compare camera predictions across indices
# ============================================================================

# The camera predictor in RayZer uses temporal PE to predict poses
# When you change the temporal index of a target image, its predicted pose changes
# This shows how the temporal embedding affects camera prediction

target_image_idx = 3  # Ablate 4th target image
temporal_indices = [0, 5, 10, 15, 20]

poses_dict = {}

for temporal_idx in temporal_indices:
    model.set_ablation_config(target_image_idx, temporal_idx)
    
    with torch.no_grad():
        results = model(data_batch)
    
    # Extract predicted camera for the 3rd target image
    # results['c2w'] shape: [batch, num_all_images, 4, 4]
    # You need to map from target_idx to the global image index
    # This depends on how your split_data works
    
    poses_dict[temporal_idx] = results['c2w'].detach().cpu().numpy()
    
    model.clear_ablation_config()

print("Poses predictions for each temporal index:")
for temporal_idx, poses in poses_dict.items():
    print(f"  Index {temporal_idx}: rotation={poses[0, 0, :3, :3]}, translation={poses[0, 0, :3, 3]}")

# ============================================================================
# KEY POINTS
# ============================================================================

"""
1. TEMPORAL INDEX RANGE:
   - You have 24 total images (16 input + 8 target)
   - Temporal indices should typically range from 0 to 23
   - Setting temporal_idx outside this range is valid but may produce unexpected results

2. WHAT CHANGES:
   - The positional embedding (PE) of the target image changes
   - The camera predictor sees a different temporal PE, so it predicts different pose
   - The scene representation stays the same (built from input images)
   - Only the rendering of that specific target image is affected

3. EXPECTED BEHAVIOR:
   - Changing temporal index from 0→12 is like telling the model "this image is from the middle"
   - Camera pose prediction will adjust accordingly
   - Reconstruction quality may improve or degrade depending on whether the new index matches actual content

4. COMPARISON METRICS:
   - PSNR: Peak Signal-to-Noise Ratio (higher is better)
   - SSIM: Structural Similarity Index (higher is better)
   - MAE: Mean Absolute Error (lower is better)
   - Camera error: Difference between predicted and ground truth poses

5. INTERPRETATION:
   - If PSNR increases with index X: the temporal embedding of index X helps reconstruction
   - If camera error decreases: the temporal index affects how well poses are predicted
   - This reveals how much the model depends on temporal information
"""

# ============================================================================
# HELPER FUNCTION: Compute PSNR
# ============================================================================

def compute_psnr(rendered, target, max_val=1.0):
    """Compute PSNR between rendered and target images."""
    mse = torch.mean((rendered - target) ** 2)
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse + 1e-10)
    return psnr.item()

# ============================================================================
# HELPER FUNCTION: Compute SSIM
# ============================================================================

def compute_ssim(rendered, target, window_size=11):
    """Compute SSIM between rendered and target images."""
    # This is a simplified version. For production, use skimage.metrics.structural_similarity
    # or torchmetrics.SSIM
    
    # Ensure correct shape: [batch, height, width, channels]
    if rendered.dim() == 4:
        rendered = rendered.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
    
    # Compute mean, variance
    mean_x = torch.nn.functional.avg_pool2d(rendered, window_size, stride=1)
    mean_y = torch.nn.functional.avg_pool2d(target, window_size, stride=1)
    
    # ... additional computation needed for full SSIM
    # For now, return placeholder
    return 0.0

# ============================================================================
# TIPS FOR INTERPRETATION
# ============================================================================

"""
When reviewing your ablation results, look for:

1. PATTERNS IN METRICS:
   - Does PSNR peak at a specific temporal index?
   - Is there a smooth curve or sharp jumps?

2. CAMERA PREDICTIONS:
   - How much do predicted poses vary with index?
   - Do they follow any pattern (e.g., linear interpolation)?

3. VISUAL INSPECTION:
   - Save rendered images for each index
   - Look for quality degradation with wrong indices
   - Check if certain indices produce mode collapse or artifacts

4. STATISTICS:
   - Mean PSNR across all indices
   - Standard deviation (how much variation?)
   - Min/max indices (best and worst)

Example output:
    Temporal index 0: PSNR = 25.43
    Temporal index 2: PSNR = 26.18
    Temporal index 5: PSNR = 28.92  <- Peak performance
    Temporal index 10: PSNR = 26.55
    Temporal index 15: PSNR = 25.21
    ...
    Conclusion: Model performs best when target image gets temporal index 5
                (which was its original index)
"""
