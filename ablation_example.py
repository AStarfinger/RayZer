"""
Simple example script for running temporal index ablation study.

This script demonstrates how to use the ablation functionality to test
how changing an image's temporal position affects camera prediction.

Setup:
    - Total images: 24 (16 input + 8 target)
    - Ablate: 1 target image (e.g., the first one)
    - Test temporal indices: Full range [0, 1, 2, ..., 23]
    
This shows how the model's predictions change when the target image
is perceived as being at different temporal positions in the sequence.

Example:
    python ablation_example.py
"""

import os
import torch
from pathlib import Path
from ablation_temporal_index_study import run_ablation_study, load_config, init_distributed_single_process

# Initialize distributed for single-process inference
try:
    init_distributed_single_process()
except Exception as e:
    print(f"Note: Distributed initialization (expected for single-process inference)")

# Configuration
CONFIG_PATH = 'configs/rayzer_dl3dv.yaml'
CHECKPOINT_PATH = 'model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt'  # Update with your checkpoint path
OUTPUT_DIR = './ablation_results_temporal'

# Ablation parameters
TARGET_IMAGE_IDX = 0  # Which target image to ablate (0-indexed)
NUM_TOTAL_VIEWS = 24  # Total number of images: 16 input + 8 target = 24 total
TEST_TEMPORAL_INDICES = list(range(NUM_TOTAL_VIEWS))  # Test ALL temporal positions: [0, 1, 2, ..., 23]
NUM_SAMPLES = 5  # Process 5 samples (set to None for all)
USE_AMP = True  # Use automatic mixed precision (recommended for flash attention)
AMP_DTYPE = 'float16'  # Data type for AMP: 'float16' or 'bfloat16'

def main():
    print("\n" + "="*70)
    print("TEMPORAL INDEX ABLATION STUDY FOR RAYZER")
    print("="*70)
    print(f"\nSetup:")
    print(f"  Total images in sequence: {NUM_TOTAL_VIEWS}")
    print(f"  Target image to ablate: {TARGET_IMAGE_IDX}")
    print(f"  Temporal indices to test: [0, 1, 2, ..., {NUM_TOTAL_VIEWS-1}] (FULL RANGE)")
    print(f"  Total combinations: {len(TEST_TEMPORAL_INDICES)} indices × {NUM_SAMPLES} samples")
    print(f"\nConfiguration:")
    print(f"  Config file: {CONFIG_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Number of samples: {NUM_SAMPLES}")
    print(f"  AMP enabled: {USE_AMP} (dtype: {AMP_DTYPE})\n")
    
    # Check if files exist
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file not found at {CONFIG_PATH}")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"Please update CHECKPOINT_PATH in this script or provide a valid path.")
        return
    
    # Load configuration
    try:
        config = load_config(CONFIG_PATH)
    except Exception as e:
        print(f"ERROR loading config: {e}")
        return
    
    # Ensure inference mode is set
    from easydict import EasyDict as edict
    if 'inference' not in config:
        config['inference'] = edict()
    config.inference.render_video = False
    config.inference.if_inference = True
    
    # Run ablation study
    try:
        ablation_results = run_ablation_study(
            config=config,
            checkpoint_path=CHECKPOINT_PATH,
            target_image_idx=TARGET_IMAGE_IDX,
            test_temporal_indices=TEST_TEMPORAL_INDICES,
            output_dir=OUTPUT_DIR,
            num_samples=NUM_SAMPLES,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE
        )
        
        print("\n✓ Ablation study completed successfully!")
        print(f"  Results saved to: {OUTPUT_DIR}/ablation_results.json")
        print(f"  Rendered images saved to: {OUTPUT_DIR}/rendered_images/")
        
    except Exception as e:
        print(f"\nERROR during ablation study: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
