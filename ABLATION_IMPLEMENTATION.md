# Temporal Index Ablation Study Implementation

This document summarizes the modifications made to enable temporal index ablation studies in RayZer.

## What Was Changed

### 1. **RayZer Model (`model/rayzer.py`)**

Added three new methods and one new attribute:

#### New Attribute
- `self.ablation_config`: Stores ablation configuration dictionary with keys:
  - `target_image_idx`: Which target image to ablate (0-indexed)
  - `ablation_temporal_idx`: What temporal index to assign

#### New Methods
- **`set_ablation_config(target_image_idx, ablation_temporal_idx)`**
  - Call before inference to enable ablation
  - Sets the ablation configuration
  
- **`clear_ablation_config()`**
  - Call after inference to disable ablation
  - Clears the configuration

#### Modified Method
- **`add_sptial_temporal_pe(img_tokens, b, v, h_origin, w_origin, target_idx=None)`**
  - Added `target_idx` parameter to receive target image indices
  - Added ablation logic that modifies temporal indices before PE encoding
  - When `ablation_config` is set, overrides the temporal index of the specified target image

## How It Works

```
Setup Phase:
  1. Model initialized with normal temporal indices (0 to 23 for 24 images)
  2. User calls: model.set_ablation_config(target_image_idx=0, ablation_temporal_idx=5)

Forward Pass:
  1. Images tokenized normally
  2. In add_sptial_temporal_pe():
     - Create normal temporal indices [0, 1, 2, ..., 23]
     - Check if ablation is active
     - If yes: replace temporal index of target image with ablation value
  3. Rest of forward pass continues with modified PE

Cleanup Phase:
  1. User calls: model.clear_ablation_config()
  2. Ablation is disabled for future inferences
```

## Key Design Decisions

1. **Non-invasive**: Changes only affect temporal PE encoding, nothing else
2. **Per-image**: Can ablate any single target image independently
3. **Safe**: Ablation config must be explicitly set and cleared
4. **Efficient**: No additional model parameters or computation overhead

## Usage Examples

### Basic Usage
```python
model.set_ablation_config(target_image_idx=0, ablation_temporal_idx=5)
results = model(data_batch)
model.clear_ablation_config()
```

### Loop Over Multiple Indices
```python
for temporal_idx in range(24):
    model.set_ablation_config(0, temporal_idx)
    results = model(data_batch)
    metrics[temporal_idx] = compute_metrics(results)
    model.clear_ablation_config()
```

## What Changes and What Doesn't

### Changes:
- ✅ Temporal positional embedding of the target image
- ✅ Camera pose prediction for that image (due to different input embedding)
- ✅ Rendered output for that image
- ✅ Reconstruction quality metrics for that image

### Stays the Same:
- ❌ Scene representation (built from input images, not affected)
- ❌ Camera intrinsics (K) prediction (separate from temporal index)
- ❌ Other target images
- ❌ Model weights
- ❌ Computational graph

## Interpretation Guide

After running ablation with 20 different temporal indices, look for:

1. **Peak PSNR**
   - Which index gives best reconstruction?
   - Likely the original/correct index for that image

2. **PSNR Curve Shape**
   - Sharp peak: strong temporal dependence
   - Broad peak: robust to temporal variation
   - Flat: weak temporal dependence

3. **Camera Predictions**
   - Do predicted poses vary smoothly with index?
   - Do they cluster around a few values?
   - Any discontinuities?

4. **Visual Quality**
   - Compare rendered images for best/worst indices
   - Look for artifacts, blur, color shifts
   - Check spatial consistency

## Files Added/Modified

### Modified Files:
- `model/rayzer.py`: Added ablation support to RayZer class

### New Example Scripts:
- `ablation_simple_example.py`: Simple working example with helper functions
- `ablation_temporal_index.py`: Full-featured ablation script template
- `ABLATION_GUIDE.md`: Detailed usage guide with theory

## Next Steps

1. **Adapt the example scripts** to your specific data loading pipeline
2. **Run ablation on a test batch** to verify functionality
3. **Visualize results** using the provided plotting functions
4. **Interpret findings** using the interpretation guide
5. **Extend to multiple images** by looping over `target_image_idx`

## Common Issues and Solutions

### Issue: IndexError in ablation code
- **Cause**: target_idx not being passed correctly
- **Solution**: Ensure `add_sptial_temporal_pe` is called with `target_idx` parameter

### Issue: Temporal index values seem ignored
- **Cause**: Ablation config not set or cleared unexpectedly
- **Solution**: Check that `set_ablation_config()` is called before forward pass

### Issue: Out of memory with ablation
- **Cause**: Accumulating rendered outputs
- **Solution**: Detach and move tensors to CPU immediately after inference

### Issue: Results look identical across indices
- **Cause**: Temporal PE might not significantly affect this image
- **Solution**: This is valid - indicates low temporal dependence

## Technical Notes

- Temporal PE is computed as sinusoidal positional encoding
- Indices are used to create different phase shifts in the sine waves
- Lower indices → different PE than higher indices
- PE is combined with spatial PE and passed through an MLP (`pe_embedder`)
- The combined PE is added to image tokens before transformer

## References

For understanding positional encodings:
- Original paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Sinusoidal PE: Different frequencies for different dimensions
- Combined PE: Spatial (2D grid position) + Temporal (image index)
