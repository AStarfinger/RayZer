# Temporal Index Ablation Study Guide

This guide explains how to use the ablation functionality in RayZer to study how temporal position encoding affects camera pose prediction.

## Overview

The temporal index directly affects the positional encoding (PE) of each image in the sequence. By changing the temporal index of a specific image and observing the resulting camera pose predictions, you can study:

1. **Whether the model relies on temporal position for camera prediction** - Does assigning a different temporal index change the predicted camera pose?
2. **Generalization to sequence positions** - How does the model perform if an image appears at an unexpected position in the sequence?
3. **Camera pose consistency** - Are predicted poses consistent or sensitive to temporal embedding changes?

## File Structure

### Core Implementation

**`model/rayzer.py`**
- `set_ablation_config(target_image_idx, ablation_temporal_idx)` - Configure which image to ablate and what temporal index to assign
- `clear_ablation_config()` - Clear ablation configuration
- `add_sptial_temporal_pe()` - Modified to apply temporal index changes before PE encoding

### Ablation Scripts

**`ablation_temporal_index_study.py`** - Main comprehensive ablation framework
- Loads model and data
- Runs inference with different temporal indices
- Computes camera metrics
- Saves detailed results to JSON

**`ablation_example.py`** - Simple example script
- Demonstrates basic usage
- Easy to modify for custom experiments

**`analyze_ablation_results.py`** - Analysis and visualization tools
- Aggregates metrics across samples
- Generates plots
- Produces summary statistics

## Quick Start

### 1. Simple Example (Recommended First Step)

```bash
# Edit the configuration in ablation_example.py:
# - Set CONFIG_PATH to your config file
# - Set CHECKPOINT_PATH to your model checkpoint
# - Adjust TARGET_IMAGE_IDX, TEST_TEMPORAL_INDICES, NUM_SAMPLES as needed

python ablation_example.py
```

### 2. Full Ablation Study

```bash
python ablation_temporal_index_study.py \
    --config configs/rayzer_dl3dv.yaml \
    --checkpoint model_checkpoints/best_model.pt \
    --target_image_idx 0 \
    --test_indices "0,5,10,15,20,25" \
    --output_dir ./ablation_results_temporal \
    --num_samples 20 \
    --device cuda
```

### 3. Analyze Results

```bash
python analyze_ablation_results.py \
    --results_dir ./ablation_results_temporal \
    --output_dir ./ablation_plots
```

## Parameter Explanation

### Ablation Configuration

**`target_image_idx`** (default: 0)
- Which target image to ablate within the target set
- 0-indexed, so 0 means the first target image
- The target set is created by `split_data()` in the forward pass

**`ablation_temporal_idx`** (range: 0 to num_views)
- What temporal index to assign to the target image
- Controls the positional embedding
- Can be any value from 0 to the total number of views
- Example: If you have 10 images, test values could be 0, 3, 6, 9

**`test_indices`**
- List of temporal indices to test for the same image
- Format: comma-separated integers (e.g., "0,5,10,15,20")
- Typically test from first position to last position of sequence

### Study Parameters

**`num_samples`** (default: None = all)
- How many validation samples to process
- Smaller numbers for quick tests, larger for comprehensive analysis
- Each sample is one scene

## Understanding the Implementation

### How Ablation Works

1. **Forward Pass**: 
   ```python
   result = model(batch, create_visual=False, render_video=False)
   ```

2. **Inside forward()**, the image tokens are processed:
   ```python
   img_tokens = self.image_tokenizer(image_all)  # Tokenize all images
   img_tokens = self.add_sptial_temporal_pe(img_tokens, b, v_all, h, w, target_idx)
   ```

3. **In add_sptial_temporal_pe()**, the ablation is applied:
   ```python
   if self.ablation_config is not None and target_idx is not None:
       # Find which target image to ablate
       global_img_pos = target_idx[batch_idx, ablation_target_idx].item()
       # Replace its temporal index
       img_indices[start_idx:end_idx] = ablation_temporal_idx
   ```

4. **Result**: The ablated image gets positional embedding corresponding to the specified temporal index, while the camera pose prediction still must predict the correct pose for that image's actual position

### Code Flow Diagram

```
Forward Pass Start
    ↓
Load Images (correct positions)
    ↓
Tokenize Images
    ↓
Add Spatial-Temporal PE
    ├─ Generate temporal indices [0, 1, 2, ..., v-1] for each view
    ├─ [ABLATION] Replace temporal index for target image
    └─ Generate sinusoidal PE based on (possibly modified) indices
    ↓
Encode Images
    ↓
Predict Camera Poses
    ↓
Render Images using ORIGINAL image positions
    ↓
Forward Pass End
```

## Expected Results

### What to Look For

1. **Temporal Index Sensitivity**
   - If metrics change dramatically with different temporal indices → model relies on temporal PE
   - If metrics remain stable → model is robust to temporal position changes

2. **Position-Dependent Effects**
   - First position (index=0) might have different behavior than middle or last
   - Could indicate the model has special handling for sequence boundaries

3. **Consistency Across Samples**
   - Are trends consistent across different scenes?
   - Or does behavior vary significantly by scene?

### Example Output

```
TEMPORAL INDEX ABLATION RESULTS - DETAILED STATISTICS
================================================================================

Temporal Index: 0
----------------
  mean_fx                      : 0.500000 ± 0.050000 [0.420000, 0.580000]
  mean_fy                      : 0.500000 ± 0.050000 [0.420000, 0.580000]
  mean_cx                      : 0.480000 ± 0.030000 [0.410000, 0.550000]
  mean_cy                      : 0.480000 ± 0.030000 [0.410000, 0.550000]

Temporal Index: 10
----------------
  mean_fx                      : 0.495000 ± 0.055000 [0.410000, 0.590000]
  mean_fy                      : 0.495000 ± 0.055000 [0.410000, 0.590000]
  mean_cx                      : 0.485000 ± 0.035000 [0.405000, 0.560000]
  mean_cy                      : 0.485000 ± 0.035000 [0.405000, 0.560000]
```

## Advanced Usage

### Custom Metric Computation

Modify `compute_camera_metrics()` in the ablation script to compute domain-specific metrics:

```python
def compute_camera_metrics(predicted_c2w, predicted_fxfycxcy, gt_c2w=None, gt_fxfycxcy=None):
    # Add your custom metrics here
    metrics['custom_metric'] = compute_something(predicted_c2w)
    return metrics
```

### Batch Ablation Multiple Images

To ablate multiple target images in a single run:

```python
for target_idx in range(num_target_images):
    ablation_results = run_ablation_study(
        config=config,
        checkpoint_path=checkpoint_path,
        target_image_idx=target_idx,
        test_temporal_indices=test_indices,
        output_dir=f'{output_dir}/target_{target_idx}',
        num_samples=10,
    )
```

### Different Temporal Index Ranges

Test different ranges depending on your hypothesis:

```bash
# Test sparse indices (large jumps)
--test_indices "0,10,20,30"

# Test consecutive indices (fine-grained analysis)
--test_indices "0,1,2,3,4,5"

# Test boundary positions (special handling at edges)
--test_indices "0,1,18,19,20"
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `num_samples`
- Use smaller batch size (already set to 1)
- Run on CPU if necessary: `--device cpu`

### Model Not Loading

- Check checkpoint path exists
- Ensure config file matches checkpoint architecture
- Try `strict=False` mode (already enabled)

### No Metrics Variation

- Verify ablation is actually being applied (check logs)
- Increase `num_samples` for better statistics
- Check if model is in eval mode (already done)

## Citation

If you use this ablation framework in your research, please cite:

```bibtex
@software{rayzer_ablation,
  title={RayZer: Temporal Index Ablation Framework},
  author={Jiang, Hanwen},
  year={2025}
}
```

## References

- Positional Encoding: "Attention is All You Need" (Vaswani et al., 2017)
- RayZer: [Your paper citation]
