# Baseline Ablation Study Guide

This guide explains how to run a baseline ablation study to evaluate model performance across different image baseline widths (distances between context and target images).

## Overview

The ablation study script generates evaluation indices with varying baseline widths and optionally runs inference to compare metrics (PSNR, LPIPS, SSIM) across different baseline settings.

**Key Features:**
- Generates evaluation index JSON files for each baseline width
- Counts total images from benchmark directory
- Creates interleaved context/target pairs (16 context, 8 target per baseline)
- Automatically runs inference for each baseline (optional)
- Collects and compares metrics across all baselines
- Generates text and CSV reports

## Prerequisites

1. A benchmark dataset directory with scene structure:
   ```
   dl3dv_benchmark/
   ├── scene_id_1/
   │   ├── images_undistort/
   │   │   ├── 000000.png
   │   │   ├── 000001.png
   │   │   └── ...
   │   └── opencv_cameras.json
   └── scene_id_2/
       └── ...
   ```

2. An evaluation index JSON file defining scenes to evaluate:
   ```json
   {
     "scene_id": {
       "context": [...],
       "target": [...]
     }
   }
   ```

3. A trained model checkpoint

4. A config YAML file (e.g., `configs/rayzer_dl3dv.yaml`)

## Usage Options

### Option 1: Generate Evaluation Indices Only

Generate baseline evaluation index files without running inference. Use this to manually run inference later.

```bash
cd /mnt/home/adrianstarfinger/RayZer

python my_src/baseline_ablation.py \
  --eval-index data/rayzer_evaluation_mine.json \
  --output-dir experiments/ablation_baseline \
  --benchmark-dir dl3dv_benchmark \
  --baseline-widths 2 4 8 16
```

**Output:**
- `experiments/ablation_baseline/eval_index_baseline_02.json`
- `experiments/ablation_baseline/eval_index_baseline_04.json`
- `experiments/ablation_baseline/eval_index_baseline_08.json`
- `experiments/ablation_baseline/eval_index_baseline_16.json`

### Option 2: Generate Indices AND Run Inference

Generate evaluation indices and automatically run inference for each baseline width.

```bash
cd /mnt/home/adrianstarfinger/RayZer

python my_src/baseline_ablation.py \
  --eval-index data/rayzer_evaluation_mine.json \
  --output-dir experiments/ablation_baseline \
  --benchmark-dir dl3dv_benchmark \
  --baseline-widths 2 4 8 16 \
  --run-inference \
  --config configs/rayzer_dl3dv.yaml \
  --model-path ./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \
  --dataset-path ./data/dl3dv10k_one_scene.txt \
  --inference-out-root ./experiments/evaluation/test
```

**Output:**
- Evaluation indices (as above)
- Inference results in:
  - `experiments/evaluation/test/rayzer_dl3dv_two_frame_02/000000/`
  - `experiments/evaluation/test/rayzer_dl3dv_two_frame_04/000001/`
  - `experiments/evaluation/test/rayzer_dl3dv_two_frame_08/000002/`
  - `experiments/evaluation/test/rayzer_dl3dv_two_frame_16/000003/`
- Metrics and reports in `experiments/ablation_baseline/`

### Option 3: Manual Inference + Collect Metrics

If you've already run inference with the generated indices, collect and compare results:

```bash
cd /mnt/home/adrianstarfinger/RayZer

python my_src/baseline_ablation.py \
  --eval-index data/rayzer_evaluation_mine.json \
  --output-dir experiments/ablation_baseline \
  --benchmark-dir dl3dv_benchmark \
  --baseline-widths 2 4 8 16 \
  --results-dir ./experiments/evaluation/test
```

## Parameters

### Required Arguments
- `--eval-index`: Path to evaluation index JSON file
- `--benchmark-dir`: Path to benchmark directory with scenes

### Optional Arguments
- `--output-dir` (default: `./experiments/ablation_baseline`): Directory for outputs
- `--baseline-widths` (default: `2 4 8 16`): Baseline widths to test
- `--results-dir`: Directory containing inference results (for metrics collection)

### Inference Arguments (when using `--run-inference`)
- `--config`: Path to config YAML file
- `--model-path`: Path to model checkpoint
- `--dataset-path`: Path to dataset list file
- `--inference-out-root`: Root directory for inference outputs
- `--run-inference`: Flag to enable automatic inference

## Output Files

After running the script, you'll find:

```
experiments/ablation_baseline/
├── eval_index_baseline_02.json    # Evaluation index for baseline 2
├── eval_index_baseline_04.json    # Evaluation index for baseline 4
├── eval_index_baseline_08.json    # Evaluation index for baseline 8
├── eval_index_baseline_16.json    # Evaluation index for baseline 16
├── ablation_report.txt             # Text report with metrics comparison
└── ablation_metrics.csv            # CSV with metrics for each baseline
```

### Sample Report Output

```
================================================================================
BASELINE ABLATION STUDY REPORT
================================================================================

SUMMARY METRICS (Averaged across views):
────────────────────────────────────────────────────────────────────────────────
Baseline Width       PSNR (mean)      LPIPS (mean)     SSIM (mean)         
────────────────────────────────────────────────────────────────────────────────
2                    20.1234          0.3421           0.6234              
4                    19.8765          0.3567           0.6012              
8                    19.2345          0.3891           0.5678              
16                   18.5432          0.4123           0.5234              

DETAILED STATISTICS:
────────────────────────────────────────────────────────────────────────────────

Baseline Width: 2
  Samples: 1
  PSNR:  20.1234 ± 0.0000
  LPIPS: 0.3421 ± 0.0000
  SSIM:  0.6234 ± 0.0000

[... more baselines ...]
```

## Understanding Baseline Widths

A **baseline width** is the spacing between selected frames:

- **Baseline 2**: Select every 2nd frame (tight spacing, more context-target diversity)
- **Baseline 4**: Select every 4th frame (medium spacing)
- **Baseline 8**: Select every 8th frame (wide spacing)
- **Baseline 16**: Select every 16th frame (very wide spacing, large gaps)

For a scene with 327 images and baseline width of 4:
- Selected frames: 0, 4, 8, 12, ..., 320
- These are split: 16 as context, 8 as target
- Context and target are interleaved for temporal continuity

## Troubleshooting

### "Metrics not found"
Make sure you've run inference and results are saved in the expected location:
```
experiments/evaluation/test/rayzer_dl3dv_two_frame_XX/000000/metrics.json
```

### "Scene directory not found"
Verify the benchmark directory structure:
```bash
ls /mnt/home/adrianstarfinger/RayZer/dl3dv_benchmark/scene_id/images_undistort/ | wc -l
```

### Inference fails with "torchrun not found"
The script automatically uses the correct Python environment. If still failing, ensure pytorch is installed:
```bash
pip install torch torchrun
```

## Examples

### Quick Test with Subset of Baselines
```bash
python my_src/baseline_ablation.py \
  --eval-index data/rayzer_evaluation_mine.json \
  --baseline-widths 2 4 \
  --run-inference \
  --config configs/rayzer_dl3dv.yaml \
  --model-path ./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \
  --dataset-path ./data/dl3dv10k_one_scene.txt
```

### Run Without Inference (Prepare Indices Only)
```bash
python my_src/baseline_ablation.py \
  --eval-index data/rayzer_evaluation_mine.json \
  --baseline-widths 2 4 8 16
```

### Analyze Previously Computed Results
```bash
python my_src/baseline_ablation.py \
  --eval-index data/rayzer_evaluation_mine.json \
  --baseline-widths 2 4 8 16 \
  --results-dir ./experiments/evaluation/test
```

## Notes

- The script runs inference sequentially for each baseline width
- Each baseline run uses a unique `rdzv_id` to avoid port conflicts
- Results are automatically organized with identifiers (000000, 000001, etc.)
- The script generates both human-readable text reports and CSV files for easy analysis
- All metrics are computed per-view and then averaged
