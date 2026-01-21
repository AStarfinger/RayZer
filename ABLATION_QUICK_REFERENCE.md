# Quick Reference: Temporal Index Ablation

## TL;DR - Get Started in 2 Minutes

### 1. Update Config (ablation_example.py)
```python
CONFIG_PATH = 'configs/rayzer_dl3dv.yaml'
CHECKPOINT_PATH = 'model_checkpoints/your_model.pt'
TEST_TEMPORAL_INDICES = list(range(0, 21, 5))  # Test [0, 5, 10, 15, 20]
```

### 2. Run Ablation
```bash
python ablation_example.py
```

### 3. Analyze Results
```bash
python analyze_ablation_results.py --results_dir ./ablation_results_temporal
```

---

## What It Does

Changes one image's **temporal position encoding** (not its actual position) to test if the model's camera prediction depends on temporal sequence position.

**Before Ablation:**
```
Images: [img_0, img_1, img_2, img_3, img_4]
Temporal indices: [0, 1, 2, 3, 4]
Model learns: camera poses based on temporal context
```

**During Ablation (ablate image 2 with temporal index 0):**
```
Images: [img_0, img_1, img_2, img_3, img_4]  # Same actual images
Temporal indices: [0, 1, 0, 3, 4]  # img_2 thinks it's at position 0
Model must predict: camera for img_2 with temporal PE of position 0
```

---

## Parameters Explained

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `target_image_idx` | Which target image to test (0-indexed) | `0` = first target image |
| `ablation_temporal_idx` | What temporal index to assign | `5` = pretend image is at position 5 |
| `test_indices` | Range of indices to test | `"0,5,10,15,20"` |
| `num_samples` | How many scenes to test | `10` = test on 10 scenes |

---

## Command Examples

### Minimal Test
```bash
python ablation_temporal_index_study.py \
    --config configs/rayzer_dl3dv.yaml \
    --checkpoint best_model.pt \
    --test_indices "0,10,20" \
    --output_dir ./ablation_quick \
    --num_samples 3
```

### Comprehensive Study
```bash
python ablation_temporal_index_study.py \
    --config configs/rayzer_dl3dv.yaml \
    --checkpoint best_model.pt \
    --test_indices "0,5,10,15,20,25,30,35,40,45,50" \
    --output_dir ./ablation_full \
    --num_samples 100 \
    --device cuda
```

### Multi-Image Ablation
```bash
for idx in 0 1 2; do
  python ablation_temporal_index_study.py \
    --config configs/rayzer_dl3dv.yaml \
    --checkpoint best_model.pt \
    --target_image_idx $idx \
    --test_indices "0,10,20" \
    --output_dir ./ablation_image_$idx \
    --num_samples 10
done
```

---

## Output Files

```
ablation_results_temporal/
├── ablation_results.json          # Raw results (all samples, all indices)
└── (after analyze script)
    ├── statistics.json             # Aggregated statistics
    ├── metric_mean_fx.png         # Focal length plot
    ├── metric_mean_fy.png         # Focal length plot
    ├── metric_mean_cx.png         # Principal point plot
    ├── metric_mean_cy.png         # Principal point plot
    └── ...
```

---

## Interpreting Results

### Example: How to Read the Output

```
Temporal Index: 0
  mean_fx: 0.500000 ± 0.050000
  
Temporal Index: 10
  mean_fx: 0.495000 ± 0.055000
```

**Interpretation:**
- Very small difference (0.500 → 0.495) = **Robust to temporal position**
- Large difference = **Depends heavily on temporal encoding**

### Statistics Breakdown

- **mean**: Average value across samples
- **std**: Variation between samples (higher = more inconsistent)
- **min/max**: Range of values observed

---

## Key Files

| File | Purpose |
|------|---------|
| `ablation_example.py` | **START HERE** - Simple example |
| `ablation_temporal_index_study.py` | Full framework for experiments |
| `analyze_ablation_results.py` | Generate plots and stats |
| `analysis_temporal_ablation.py` | Interactive exploration |
| `TEMPORAL_ABLATION_GUIDE.md` | Detailed documentation |
| `model/rayzer.py` | Actual implementation (lines 505-520) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **OOM Error** | Reduce `--num_samples` or use `--device cpu` |
| **Model not loading** | Check checkpoint path exists and matches config |
| **No metric variation** | Increase `--num_samples` for better statistics |
| **Slow running** | Use smaller `--num_samples` first, then scale up |

---

## Tips & Tricks

✓ **Start small**: Test with `--num_samples 3` to verify setup works  
✓ **Sparse indices**: Use `"0,10,20,30"` for quick rough analysis  
✓ **Dense indices**: Use `"0,1,2,3,4,5"` for fine-grained study  
✓ **Boundary testing**: Use `"0,1,4,8,9"` to test sequence edge effects  

---

## Understanding the Code

```python
# Before forward pass:
model.set_ablation_config(target_image_idx=0, ablation_temporal_idx=5)

# During forward pass, in add_sptial_temporal_pe():
if self.ablation_config is not None:
    # Find the target image in the full image set
    global_img_pos = target_idx[batch_idx, ablation_target_idx].item()
    # Replace its temporal index before PE computation
    img_indices[start_idx:end_idx] = ablation_temporal_idx
    # This affects only the positional encoding

# After forward pass:
model.clear_ablation_config()
```

**Key insight**: The ablation happens at the **positional encoding stage**, not the image loading stage. The actual image is in the correct position, but it "thinks" it's at a different temporal position.

---

## Expected Behavior

✓ Model should still predict roughly correct camera poses  
✓ Performance might degrade with very different temporal indices  
✓ First/last positions may be treated differently than middle  
✓ Consistent behavior indicates good generalization  

---

## Citation

```bibtex
@inproceedings{rayzer2025,
  title={RayZer: Temporal Index Ablation Study},
  author={Jiang, Hanwen},
  year={2025}
}
```

---

## Questions?

- **How ablation works**: See `TEMPORAL_ABLATION_GUIDE.md`
- **Detailed parameters**: See function docstrings in script files
- **Results interpretation**: See analysis examples in `analysis_temporal_ablation.py`
- **Code walkthrough**: See comments in `model/rayzer.py` lines 505-520
