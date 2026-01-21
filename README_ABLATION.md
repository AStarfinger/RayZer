# Temporal Index Ablation - Complete Implementation

## Summary

You now have a **complete ablation study framework** for testing how temporal positional encodings affect camera pose prediction in RayZer. The implementation allows you to:

1. **Change an image's temporal index** before positional encoding
2. **Test multiple temporal indices** on the same image
3. **Measure camera prediction metrics** for each configuration
4. **Analyze and visualize results** across samples

## Implementation Details

### Core Change: `model/rayzer.py` (lines 505-520)

The ablation logic in `add_sptial_temporal_pe()` modifies image temporal indices **before** sinusoidal positional encoding:

```python
if self.ablation_config is not None and target_idx is not None:
    ablation_target_idx = self.ablation_config['target_image_idx']
    ablation_temporal_idx = self.ablation_config['ablation_temporal_idx']
    
    for batch_idx in range(b):
        if ablation_target_idx < target_idx.shape[1]:
            global_img_pos = target_idx[batch_idx, ablation_target_idx].item()
            start_idx = batch_idx * v * n + global_img_pos * n
            end_idx = start_idx + n
            img_indices[start_idx:end_idx] = ablation_temporal_idx
```

**Where the change happens:**
- After images load and are tokenized ✓
- Before positional encoding is computed ✓
- Selectively modifies only the target image ✓
- Preserves original image positions for rendering ✓

## New Files Created

### 1. **ablation_temporal_index_study.py** (300+ lines)
Full-featured ablation framework
- Load model and dataset
- Run inference with different temporal indices
- Compute camera metrics
- Save results to JSON
- Print summary statistics

### 2. **ablation_example.py** (90+ lines)
Simple entry point for quick experiments
- Edit config at top of file
- Run: `python ablation_example.py`
- Perfect for getting started

### 3. **analyze_ablation_results.py** (250+ lines)
Post-processing and visualization
- Aggregate metrics across samples
- Generate plots for each metric
- Compute statistics
- Export analysis results

### 4. **analysis_temporal_ablation.py** (200+ lines)
Interactive Python analysis script
- Load results into pandas DataFrame
- Compute correlations
- Create distribution plots and heatmaps
- Custom queries on results

### 5. **TEMPORAL_ABLATION_GUIDE.md** (350+ lines)
Comprehensive user documentation
- Overview and methodology
- Complete usage instructions
- Parameter explanations
- Code flow diagrams
- Expected results and interpretation
- Advanced usage examples
- Troubleshooting guide

### 6. **ABLATION_IMPLEMENTATION_SUMMARY.md**
Technical summary of implementation
- What was completed
- How it works
- Design decisions
- Integration with existing code

### 7. **ABLATION_QUICK_REFERENCE.md**
Quick start guide
- 2-minute setup
- Command examples
- Parameter quick reference
- Output interpretation
- Tips and tricks

## Quick Start

### 1. Configure (edit `ablation_example.py`)
```python
CONFIG_PATH = 'configs/rayzer_dl3dv.yaml'
CHECKPOINT_PATH = 'model_checkpoints/best_model.pt'
TARGET_IMAGE_IDX = 0
TEST_TEMPORAL_INDICES = [0, 5, 10, 15, 20]
NUM_SAMPLES = 10
```

### 2. Run
```bash
python ablation_example.py
```

### 3. Analyze
```bash
python analyze_ablation_results.py --results_dir ./ablation_results_temporal
```

## File Map

```
RayZer/
├── model/
│   └── rayzer.py ......................... [MODIFIED] Core implementation
├── ablation_temporal_index_study.py ...... [NEW] Full framework
├── ablation_example.py ................... [NEW] Simple entry point
├── analyze_ablation_results.py ........... [NEW] Analysis & visualization
├── analysis_temporal_ablation.py ......... [NEW] Interactive analysis
├── TEMPORAL_ABLATION_GUIDE.md ............ [NEW] Complete guide
├── ABLATION_IMPLEMENTATION_SUMMARY.md ... [NEW] Technical summary
└── ABLATION_QUICK_REFERENCE.md .......... [NEW] Quick start guide
```

## Usage Examples

### Quick Test (5 samples, 5 indices)
```bash
python ablation_example.py
```

### Full Study
```bash
python ablation_temporal_index_study.py \
    --config configs/rayzer_dl3dv.yaml \
    --checkpoint best_model.pt \
    --target_image_idx 0 \
    --test_indices "0,5,10,15,20,25,30,35,40,45,50" \
    --output_dir ./ablation_results \
    --num_samples 100
```

### Analyze Results
```bash
python analyze_ablation_results.py --results_dir ./ablation_results
```

## What You Can Measure

The framework measures camera prediction metrics:
- **mean_fx**: Mean focal length X
- **mean_fy**: Mean focal length Y
- **mean_cx**: Mean principal point X
- **mean_cy**: Mean principal point Y

You can extend to add custom metrics by modifying `compute_camera_metrics()`.

## Understanding the Ablation

**Normal Flow:**
```
Image loads → Tokenize → Add PE with original temporal index → Predict camera
```

**With Ablation:**
```
Image loads → Tokenize → Add PE with MODIFIED temporal index → Predict camera
```

The image stays in its original position, but the transformer sees it with positional encoding from a different temporal position.

## Key Research Questions Answered

1. **Does the model rely on temporal position for camera prediction?**
   - If metrics don't change → Temporal position is optional
   - If metrics change dramatically → Temporal position is critical

2. **Does the model handle unusual temporal positions?**
   - Test with very different temporal indices (e.g., first image at position 20)

3. **Are certain positions treated specially?**
   - Compare first position (0), middle, and last position effects

4. **Is the temporal encoding robust?**
   - Test multiple temporal indices for consistency

## Next Steps

1. **Read** `ABLATION_QUICK_REFERENCE.md` for quick overview
2. **Run** `ablation_example.py` with your model to verify it works
3. **Design** your ablation study based on research questions
4. **Execute** using `ablation_temporal_index_study.py` with custom parameters
5. **Analyze** using `analyze_ablation_results.py` and `analysis_temporal_ablation.py`

## Support for Complex Workflows

- **Multi-image ablation**: Loop through different target_image_idx values
- **Different index ranges**: Test sparse (0,10,20) or dense (0,1,2,...) ranges
- **Batch processing**: Process multiple models/checkpoints
- **Custom metrics**: Extend `compute_camera_metrics()` function
- **Statistical analysis**: Use `analysis_temporal_ablation.py` for advanced stats

## Architecture Integration

The ablation integrates cleanly with existing code:
- ✓ Uses existing `set_ablation_config()` framework
- ✓ Optional (doesn't affect normal operation when disabled)
- ✓ Works with DDP training
- ✓ Compatible with existing data loading
- ✓ No model architecture changes needed

## Output Example

```
TEMPORAL INDEX ABLATION RESULTS - DETAILED STATISTICS
================================================================================

Temporal Index: 0
  mean_fx : 0.500000 ± 0.050000 [0.420000, 0.580000]
  mean_fy : 0.500000 ± 0.050000 [0.420000, 0.580000]
  
Temporal Index: 10
  mean_fx : 0.495000 ± 0.055000 [0.410000, 0.590000]
  mean_fy : 0.495000 ± 0.055000 [0.410000, 0.590000]
```

Analysis plots generated:
- Metric vs Temporal Index (with error bars)
- Distribution plots (box plots)
- Heatmaps (samples × temporal indices)
- Correlation statistics

---

**You're all set!** Start with `ablation_example.py` and refer to the guides as needed.
