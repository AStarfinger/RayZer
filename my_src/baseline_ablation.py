"""
Ablation study script for different baseline widths between input images.

This script creates evaluation indices with varying baseline widths (distances between 
image indices) and optionally runs inference and compares metrics like PSNR and LPIPS 
across different baseline settings.

WORKFLOW:

Option A: Generate indices only, run inference separately
1. python baseline_ablation.py --eval-index ... --baseline-widths 2 4 8 16
2. Run your own inference for each eval index, save to 000000/, 000001/, etc.
3. python baseline_ablation.py --eval-index ... --baseline-widths 2 4 8 16 --results-dir <results_path>

Option B: Generate indices AND run inference automatically
python baseline_ablation.py --eval-index ... \\
                            --benchmark-dir ... \\
                            --baseline-widths 2 4 8 16 \\
                            --run-inference \\
                            --config configs/rayzer_dl3dv.yaml \\
                            --model-path ./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \\
                            --dataset-path ./data/dl3dv10k_one_scene.txt \\
                            --inference-out-root ./experiments/evaluation/test

This script will:
1. Count total images in each scene from benchmark directory
2. Generate context/target index pairs with interleaved baselines
3. Optionally run inference for each baseline
4. Collect metrics from evaluation results
5. Generate comparison report (CSV + text)
"""

import json
import os
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


class BaselineAblationStudy:
    """Conducts ablation study on different baseline widths."""
    
    def __init__(self, eval_index_path: str, output_dir: str, benchmark_dir: str = None):
        """
        Initialize the ablation study.
        
        Args:
            eval_index_path: Path to the evaluation index JSON
            output_dir: Directory to save ablation results
            benchmark_dir: Path to benchmark directory containing scene folders
        """
        self.eval_index_path = eval_index_path
        self.output_dir = output_dir
        self.benchmark_dir = benchmark_dir
        self.eval_index = self._load_eval_index()
        self.scene_image_counts = self._count_scene_images() if benchmark_dir else {}
        
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_eval_index(self) -> Dict:
        """Load evaluation index from JSON file."""
        with open(self.eval_index_path, 'r') as f:
            return json.load(f)
    
    def _count_scene_images(self) -> Dict[str, int]:
        """
        Count the number of images in each scene directory.
        
        Returns:
            Dictionary mapping scene_id to image count
        """
        scene_counts = {}
        
        for scene_id in self.eval_index.keys():
            scene_path = os.path.join(self.benchmark_dir, scene_id, 'images_distort')
            
            if os.path.exists(scene_path):
                try:
                    image_files = [f for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    scene_counts[scene_id] = len(image_files)
                except Exception as e:
                    print(f"⚠ Error counting images in {scene_path}: {e}")
                    scene_counts[scene_id] = 0
            else:
                print(f"⚠ Scene directory not found: {scene_path}")
                scene_counts[scene_id] = 0
        
        return scene_counts
    
    def _generate_baseline_indices(self, all_indices: List[int], baseline_width: int) -> Tuple[List[int], List[int]]:
        """
        Generate context and target indices with specified baseline width.
        Interleaves them so targets are surrounded by context images.
        
        Args:
            all_indices: All available frame indices (sorted)
            baseline_width: Distance between consecutive selected frames
            
        Returns:
            Tuple of (context_indices, target_indices) where context=16, target=8
        """
        # Select frames with the given spacing
        selected_indices = all_indices[::baseline_width]
        
        if len(selected_indices) < 2:
            # If spacing is too large, just use first and last
            selected_indices = [all_indices[0], all_indices[-1]]
        
        # Target 8 target images and 16 context images (total 24)
        target_count = 8
        context_count = 16
        total_needed = target_count + context_count
        
        # If we don't have enough selected indices, use what we have
        if len(selected_indices) < total_needed:
            # Evenly distribute: pick every (total_selected / total_needed)-th frame
            step = max(1, len(selected_indices) // total_needed)
            selected_indices = selected_indices[::step][:total_needed]
        else:
            # Select the first 24 frames from our selection
            selected_indices = selected_indices[:total_needed]
        
        # Interleave: pick every 3rd frame as target (gives 8 targets from 24)
        # This ensures targets are surrounded by context
        target = []
        context = []
        
        for i, idx in enumerate(selected_indices):
            if len(target) < target_count and i % 3 == 2:
                # Every 3rd frame (positions 2, 5, 8, 11, 14, 17, 20, 23) is target
                target.append(idx)
            else:
                context.append(idx)
        
        # If we don't have exactly 8 targets due to rounding, adjust
        while len(target) < target_count and len(context) > context_count:
            context.pop()
            target.append(context.pop() if context else None)
        
        target = sorted([t for t in target if t is not None])
        context = sorted(context)
        
        # Ensure we have at least the desired counts
        if len(target) < target_count and len(context) > context_count:
            # Move some context to target
            excess = len(context) - context_count
            for _ in range(min(excess, target_count - len(target))):
                target.append(context.pop())
        elif len(context) < context_count and len(target) > target_count:
            # Move some target to context
            excess = len(target) - target_count
            for _ in range(min(excess, context_count - len(context))):
                context.append(target.pop())
        
        target = sorted(target)
        context = sorted(context)
        
        return context, target
    
    def create_ablation_indices(self, baseline_widths: List[int]) -> Dict[int, Dict]:
        """
        Create evaluation indices for different baseline widths.
        
        Args:
            baseline_widths: List of baseline widths to test (e.g., [2, 4, 8, 16])
            
        Returns:
            Dictionary mapping baseline_width to scene ablation data
        """
        ablation_data = {}
        
        for scene_id in self.eval_index.keys():
            # Get total number of images in this scene
            if scene_id in self.scene_image_counts:
                total_images = self.scene_image_counts[scene_id]
            else:
                # Fallback to original indices if benchmark_dir not available
                original_data = self.eval_index[scene_id]
                total_images = max(original_data['context'] + original_data['target']) + 1
            
            # Create full range of indices (0 to total_images - 1)
            all_indices = list(range(total_images))
            
            for baseline_width in baseline_widths:
                if baseline_width not in ablation_data:
                    ablation_data[baseline_width] = {}
                
                context, target = self._generate_baseline_indices(all_indices, baseline_width)
                
                ablation_data[baseline_width][scene_id] = {
                    'context': context,
                    'target': target,
                    'baseline_width': baseline_width,
                    'total_images': total_images
                }
        
        return ablation_data
    
    def save_ablation_jsons(self, ablation_data: Dict[int, Dict], baseline_widths: List[int]) -> Dict[int, str]:
        """
        Save ablation evaluation indices as JSON files with proper naming.
        
        Args:
            ablation_data: Dictionary of ablation data
            baseline_widths: List of baseline widths
            
        Returns:
            Dictionary mapping baseline_width to saved JSON path
        """
        saved_paths = {}
        
        for i, baseline_width in enumerate(baseline_widths):
            eval_json = {}
            
            for scene_id, data in ablation_data[baseline_width].items():
                eval_json[scene_id] = {
                    'context': data['context'],
                    'target': data['target']
                }
            
            output_path = os.path.join(
                self.output_dir,
                f'eval_index_baseline_{baseline_width:02d}.json'
            )
            
            with open(output_path, 'w') as f:
                json.dump(eval_json, f, indent=2)
            
            saved_paths[baseline_width] = output_path
            print(f"✓ Saved baseline {baseline_width} evaluation index to {output_path}")
        
        return saved_paths
    
    def run_inference(self, baseline_widths: List[int], saved_paths: Dict[int, str], 
                     inference_config: Dict) -> Dict[int, str]:
        """
        Run inference for each baseline width using the generated evaluation indices.
        
        Args:
            baseline_widths: List of baseline widths to test
            saved_paths: Dictionary mapping baseline_width to eval_index JSON path
            inference_config: Configuration dictionary with:
                - config: Path to base config YAML
                - model_path: Path to model checkpoint
                - dataset_path: Path to dataset list file
                - inference_out_root: Root directory for inference outputs
                - (optional) Additional config overrides
                
        Returns:
            Dictionary mapping baseline_width to output directory
        """
        results_dirs = {}
        
        for baseline_idx, baseline_width in enumerate(baseline_widths):
            # Map baseline index to identifier: baseline 0 -> 000000, baseline 1 -> 000001, etc.
            identifier = f"{baseline_idx:06d}"
            output_dir = os.path.join(inference_config['inference_out_root'], identifier)
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the evaluation index path for this baseline
            eval_index_path = saved_paths[baseline_width]
            
            print(f"\n{'='*80}")
            print(f"RUNNING INFERENCE: Baseline Width {baseline_width} -> {identifier}")
            print(f"{'='*80}")
            print(f"Eval index: {eval_index_path}")
            print(f"Output dir: {output_dir}")
            
            # Build the inference command
            cmd = [
                'torchrun',
                '--nproc_per_node', '1',
                '--nnodes', '1',
                '--rdzv_id', str(baseline_idx),
                '--rdzv_backend', 'c10d',
                '--rdzv_endpoint', 'localhost:29506',
                'inference.py',
                '--config', inference_config['config'],
                f'training.dataset_path={inference_config["dataset_path"]}',
                'training.batch_size_per_gpu=1',
                'training.target_has_input=false',
                'training.num_views=24',
                'training.num_input_views=16',
                'training.num_target_views=8',
                'inference.if_inference=true',
                'inference.compute_metrics=true',
                'inference.render_video=false',
                f'inference.view_idx_file_path={eval_index_path}',
                f'inference.model_path={inference_config["model_path"]}',
                f'inference_out_root={inference_config["inference_out_root"]}'
            ]
            
            # Add any additional config overrides
            for key, value in inference_config.items():
                if key not in ['config', 'model_path', 'dataset_path', 'inference_out_root']:
                    cmd.append(f'{key}={value}')
            
            try:
                print(f"\nRunning: {' '.join(cmd)}\n")
                result = subprocess.run(cmd, check=True, cwd='/mnt/home/adrianstarfinger/RayZer')
                print(f"✓ Inference completed for baseline {baseline_width}")
                results_dirs[baseline_width] = output_dir
            except subprocess.CalledProcessError as e:
                print(f"✗ Inference failed for baseline {baseline_width}: {e}")
                print(f"  Command: {' '.join(cmd)}")
        
        return results_dirs
        """
        Save ablation evaluation indices as JSON files with proper naming.
        
        Args:
            ablation_data: Dictionary of ablation data
            baseline_widths: List of baseline widths
            
        Returns:
            Dictionary mapping baseline_width to saved JSON path
        """
        saved_paths = {}
        
        for i, baseline_width in enumerate(baseline_widths):
            eval_json = {}
            
            for scene_id, data in ablation_data[baseline_width].items():
                eval_json[scene_id] = {
                    'context': data['context'],
                    'target': data['target']
                }
            
            output_path = os.path.join(
                self.output_dir,
                f'eval_index_baseline_{baseline_width:02d}.json'
            )
            
            with open(output_path, 'w') as f:
                json.dump(eval_json, f, indent=2)
            
            saved_paths[baseline_width] = output_path
            print(f"✓ Saved baseline {baseline_width} evaluation index to {output_path}")
        
        return saved_paths
    
    def collect_metrics(self, baseline_widths: List[int], results_base_dir: str) -> Dict[int, Dict]:
        """
        Collect metrics from evaluation results for different baselines.
        Searches recursively for metrics.json in subdirectories.
        
        Args:
            baseline_widths: List of baseline widths (in order)
            results_base_dir: Base directory containing evaluation results
                             Looks for patterns like:
                             - <results_base_dir>/<subdir>/000000/metrics.json
                             - <results_base_dir>/rayzer_dl3dv_two_frame_02/000000/metrics.json
            
        Returns:
            Dictionary mapping baseline_width to metrics
        """
        collected_metrics = {}
        
        for baseline_width in baseline_widths:
            collected_metrics[baseline_width] = {
                'psnr': [],
                'lpips': [],
                'ssim': [],
                'per_view': defaultdict(dict)
            }
            
            # Try multiple possible paths
            possible_paths = [
                # Path with baseline width in subdirectory name (most common)
                os.path.join(results_base_dir, f'rayzer_dl3dv_two_frame_{baseline_width:02d}', '000000', 'metrics.json'),
                # Also try with other potential naming patterns
                os.path.join(results_base_dir, f'baseline_{baseline_width}', '000000', 'metrics.json'),
                os.path.join(results_base_dir, f'baseline_width_{baseline_width}', '000000', 'metrics.json'),
            ]
            
            # Also search recursively for subdirectories containing baseline width
            if os.path.exists(results_base_dir):
                for root, dirs, files in os.walk(results_base_dir):
                    # Look for directories containing the baseline width
                    if (str(baseline_width) in os.path.basename(root) or 
                        f'_{baseline_width:02d}' in os.path.basename(root)) and 'metrics.json' in files:
                        metrics_path = os.path.join(root, 'metrics.json')
                        if metrics_path not in possible_paths:
                            possible_paths.append(metrics_path)
            
            # Try to find and load metrics
            metrics_found = False
            for metrics_path in possible_paths:
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                        
                        # Collect summary metrics
                        collected_metrics[baseline_width]['psnr'].append(
                            metrics['summary']['psnr']
                        )
                        collected_metrics[baseline_width]['lpips'].append(
                            metrics['summary']['lpips']
                        )
                        collected_metrics[baseline_width]['ssim'].append(
                            metrics['summary']['ssim']
                        )
                        
                        # Collect per-view metrics
                        for view_data in metrics.get('per_view', []):
                            view_idx = view_data['view']
                            collected_metrics[baseline_width]['per_view'][view_idx] = {
                                'psnr': view_data.get('psnr', 0),
                                'lpips': view_data.get('lpips', 0),
                                'ssim': view_data.get('ssim', 0)
                            }
                        
                        print(f"✓ Loaded metrics for baseline {baseline_width} from {metrics_path}")
                        metrics_found = True
                        break
                    except Exception as e:
                        print(f"⚠ Error reading metrics from {metrics_path}: {e}")
            
            if not metrics_found:
                print(f"⚠ Metrics not found for baseline {baseline_width}")
                print(f"  Searched in:")
                for path in possible_paths[:3]:  # Show first 3 paths
                    print(f"    - {path}")
        
        return collected_metrics
    
    def create_comparison_report(self, collected_metrics: Dict[int, Dict], 
                                 baseline_widths: List[int]) -> str:
        """
        Create a comparison report of metrics across different baselines.
        
        Args:
            collected_metrics: Collected metrics from results
            baseline_widths: List of baseline widths
            
        Returns:
            Formatted comparison report string
        """
        report = "\n" + "="*80 + "\n"
        report += "BASELINE ABLATION STUDY REPORT\n"
        report += "="*80 + "\n\n"
        
        # Summary statistics
        report += "SUMMARY METRICS (Averaged across views):\n"
        report += "-" * 80 + "\n"
        report += f"{'Baseline Width':<20} {'PSNR (mean)':<20} {'LPIPS (mean)':<20} {'SSIM (mean)':<20}\n"
        report += "-" * 80 + "\n"
        
        summary_data = []
        
        for baseline_width in baseline_widths:
            metrics = collected_metrics[baseline_width]
            
            psnr_mean = np.mean(metrics['psnr']) if metrics['psnr'] else 0
            lpips_mean = np.mean(metrics['lpips']) if metrics['lpips'] else 0
            ssim_mean = np.mean(metrics['ssim']) if metrics['ssim'] else 0
            
            report += f"{baseline_width:<20} {psnr_mean:<20.4f} {lpips_mean:<20.4f} {ssim_mean:<20.4f}\n"
            
            summary_data.append({
                'baseline_width': baseline_width,
                'psnr_mean': psnr_mean,
                'lpips_mean': lpips_mean,
                'ssim_mean': ssim_mean,
                'psnr_std': np.std(metrics['psnr']) if metrics['psnr'] else 0,
                'lpips_std': np.std(metrics['lpips']) if metrics['lpips'] else 0,
                'ssim_std': np.std(metrics['ssim']) if metrics['ssim'] else 0,
                'num_samples': len(metrics['psnr'])
            })
        
        report += "\n"
        
        # Detailed statistics
        report += "DETAILED STATISTICS:\n"
        report += "-" * 80 + "\n"
        
        for data in summary_data:
            report += f"\nBaseline Width: {data['baseline_width']}\n"
            report += f"  Samples: {data['num_samples']}\n"
            report += f"  PSNR:  {data['psnr_mean']:.4f} ± {data['psnr_std']:.4f}\n"
            report += f"  LPIPS: {data['lpips_mean']:.4f} ± {data['lpips_std']:.4f}\n"
            report += f"  SSIM:  {data['ssim_mean']:.4f} ± {data['ssim_std']:.4f}\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report
    
    def create_csv_report(self, collected_metrics: Dict[int, Dict], 
                         baseline_widths: List[int], output_csv: str):
        """
        Save metrics comparison as CSV file.
        
        Args:
            collected_metrics: Collected metrics from results
            baseline_widths: List of baseline widths
            output_csv: Output CSV file path
        """
        rows = []
        
        for baseline_width in baseline_widths:
            metrics = collected_metrics[baseline_width]
            
            rows.append({
                'baseline_width': baseline_width,
                'psnr_mean': np.mean(metrics['psnr']) if metrics['psnr'] else 0,
                'psnr_std': np.std(metrics['psnr']) if metrics['psnr'] else 0,
                'lpips_mean': np.mean(metrics['lpips']) if metrics['lpips'] else 0,
                'lpips_std': np.std(metrics['lpips']) if metrics['lpips'] else 0,
                'ssim_mean': np.mean(metrics['ssim']) if metrics['ssim'] else 0,
                'ssim_std': np.std(metrics['ssim']) if metrics['ssim'] else 0,
                'num_samples': len(metrics['psnr'])
            })
        
        # Write CSV manually
        with open(output_csv, 'w') as f:
            # Header
            f.write('baseline_width,psnr_mean,psnr_std,lpips_mean,lpips_std,ssim_mean,ssim_std,num_samples\n')
            # Rows
            for row in rows:
                f.write(f"{row['baseline_width']},{row['psnr_mean']:.4f},{row['psnr_std']:.4f},"
                       f"{row['lpips_mean']:.4f},{row['lpips_std']:.4f},"
                       f"{row['ssim_mean']:.4f},{row['ssim_std']:.4f},{row['num_samples']}\n")
        
        print(f"✓ Saved CSV report to {output_csv}")
    
    def run(self, baseline_widths: List[int], results_base_dir: str = None, 
            run_inference_flag: bool = False, inference_config: Dict = None) -> str:
        """
        Run the complete ablation study.
        
        Args:
            baseline_widths: List of baseline widths to test
            results_base_dir: Optional base directory containing evaluation results
            run_inference_flag: Whether to run inference for each baseline
            inference_config: Configuration for inference (required if run_inference_flag=True)
            
        Returns:
            Formatted comparison report
        """
        print(f"\n{'='*80}")
        print("BASELINE ABLATION STUDY")
        print(f"{'='*80}")
        
        # Step 0: Count images in each scene
        if self.benchmark_dir and self.scene_image_counts:
            print(f"\n0. Scene image counts:")
            for scene_id, count in self.scene_image_counts.items():
                print(f"   {scene_id}: {count} images")
        
        # Step 1: Create ablation indices
        print(f"\n1. Creating evaluation indices for baselines: {baseline_widths}")
        ablation_data = self.create_ablation_indices(baseline_widths)
        print(f"✓ Generated ablation data for {len(ablation_data)} baseline widths")
        
        # Step 2: Save ablation JSONs
        print(f"\n2. Saving evaluation index JSONs...")
        saved_paths = self.save_ablation_jsons(ablation_data, baseline_widths)
        
        # Step 3: Run inference if requested
        if run_inference_flag and inference_config:
            print(f"\n3. Running inference for each baseline...")
            self.run_inference(baseline_widths, saved_paths, inference_config)
            results_base_dir = inference_config['inference_out_root']
            print(f"\n✓ Inference completed. Results saved to {results_base_dir}")
        else:
            print(f"\n3. Skipping inference (use --run-inference to enable)")
        
        # Step 4: Collect metrics if results directory provided
        collected_metrics = None
        if results_base_dir and os.path.exists(results_base_dir):
            print(f"\n4. Collecting metrics from {results_base_dir}...")
            collected_metrics = self.collect_metrics(baseline_widths, results_base_dir)
            
            # Step 5: Create reports
            print(f"\n5. Generating comparison reports...")
            report = self.create_comparison_report(collected_metrics, baseline_widths)
            
            # Save report to file
            report_path = os.path.join(self.output_dir, 'ablation_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"✓ Saved text report to {report_path}")
            
            # Save CSV report
            csv_path = os.path.join(self.output_dir, 'ablation_metrics.csv')
            self.create_csv_report(collected_metrics, baseline_widths, csv_path)
        else:
            print(f"\n4. Results directory not provided or doesn't exist")
            print(f"   Skipping metrics collection step")
            report = "\nNo metrics collected. Please provide --results-dir with evaluation results or use --run-inference."
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for different baseline widths between images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate evaluation indices only
  python baseline_ablation.py --eval-index data/rayzer_evaluation_mine.json \\
                              --output-dir experiments/ablation_baseline \\
                              --benchmark-dir dl3dv_benchmark \\
                              --baseline-widths 2 4 8 16
  
  # Generate indices AND run inference
  python baseline_ablation.py --eval-index data/rayzer_evaluation_mine.json \\
                              --output-dir experiments/ablation_baseline \\
                              --benchmark-dir dl3dv_benchmark \\
                              --baseline-widths 2 4 8 16 \\
                              --run-inference \\
                              --config configs/rayzer_dl3dv.yaml \\
                              --model-path ./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \\
                              --dataset-path ./data/dl3dv10k_one_scene.txt \\
                              --inference-out-root ./experiments/evaluation/test
        """
    )
    parser.add_argument(
        '--eval-index',
        type=str,
        default='/mnt/home/adrianstarfinger/RayZer/data/rayzer_evaluation_mine.json',
        help='Path to evaluation index JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/mnt/home/adrianstarfinger/RayZer/experiments/ablation_baseline',
        help='Directory to save ablation results'
    )
    parser.add_argument(
        '--benchmark-dir',
        type=str,
        default='/mnt/home/adrianstarfinger/RayZer/dl3dv_benchmark',
        help='Path to benchmark directory containing scene folders'
    )
    parser.add_argument(
        '--baseline-widths',
        type=int,
        nargs='+',
        default=[2, 4, 8, 16],
        help='Baseline widths to test'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Base directory containing evaluation results. '
             'Expects results for baseline widths in order: 000000 for first width, 000001 for second, etc. '
             'Example: experiments/evaluation/test/rayzer_dl3dv_two_frame_mine/'
    )
    
    # Inference-related arguments
    parser.add_argument(
        '--run-inference',
        action='store_true',
        help='Run inference for each baseline width'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rayzer_dl3dv.yaml',
        help='Path to base config YAML file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./data/dl3dv10k_one_scene.txt',
        help='Path to dataset list file'
    )
    parser.add_argument(
        '--inference-out-root',
        type=str,
        default='./experiments/evaluation/test',
        help='Root directory for inference outputs'
    )
    
    args = parser.parse_args()
    
    # Initialize ablation study
    ablation = BaselineAblationStudy(args.eval_index, args.output_dir, args.benchmark_dir)
    
    # Prepare inference config if needed
    inference_config = None
    if args.run_inference:
        inference_config = {
            'config': args.config,
            'model_path': args.model_path,
            'dataset_path': args.dataset_path,
            'inference_out_root': args.inference_out_root
        }
    
    # Use inference_out_root as results_dir if running inference
    results_dir = args.results_dir
    if args.run_inference:
        results_dir = args.inference_out_root
    
    # Run the ablation study
    report = ablation.run(
        args.baseline_widths,
        results_base_dir=results_dir,
        run_inference_flag=args.run_inference,
        inference_config=inference_config
    )
    
    print(report)


if __name__ == '__main__':
    main()
