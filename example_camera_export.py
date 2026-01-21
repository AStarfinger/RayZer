"""
Example usage of camera export in your RayZer inference/training pipeline.

Add this to your inference.py or training script to export cameras.
"""

from utils.camera_export import export_cameras_to_json, export_cameras_to_npz


def export_predicted_cameras(result, output_dir, batch_idx=0, image_height=512, image_width=512):
    """
    Export predicted cameras from RayZer forward pass result.
    
    Args:
        result: dict - output from model.forward()
        output_dir: str - directory to save camera files
        batch_idx: int - which batch sample to export (default 0 for first batch)
        image_height: int - image height used in training
        image_width: int - image width used in training
    """
    c2w = result['c2w'][batch_idx]  # [v, 4, 4]
    fxfycxcy = result['fxfycxcy']   # [b*v, 4]
    
    # Extract intrinsics for this batch
    b_size = result['c2w'].shape[0]
    num_views = result['c2w'].shape[1]
    view_indices = torch.arange(batch_idx * num_views, (batch_idx + 1) * num_views)
    fxfycxcy_batch = fxfycxcy[view_indices]  # [v, 4]
    
    # Export to JSON (recommended for easy viewing)
    json_path = f"{output_dir}/cameras_batch_{batch_idx}.json"
    export_cameras_to_json(
        c2w.unsqueeze(0),  # Add batch dimension
        fxfycxcy_batch,
        json_path,
        image_height=image_height,
        image_width=image_width
    )
    
    # Optionally also export to NPZ for downstream processing
    npz_path = f"{output_dir}/cameras_batch_{batch_idx}.npz"
    export_cameras_to_npz(
        c2w.unsqueeze(0),
        fxfycxcy_batch,
        npz_path
    )
    
    print(f"Exported cameras for batch {batch_idx}")
    return json_path


# ============ EXAMPLE INTEGRATION ============
if __name__ == "__main__":
    import torch
    from model.rayzer import RayZer
    
    # Your existing setup
    config = ...  # Load your config
    model = RayZer(config)
    model.load_ckpt("path/to/checkpoint")
    model.eval()
    
    # Your data loading
    data = ...  # Load your data
    
    # Forward pass
    with torch.no_grad():
        result = model(data, create_visual=False, render_video=False)
    
    # Export cameras
    output_dir = "exported_cameras"
    json_path = export_predicted_cameras(
        result,
        output_dir,
        batch_idx=0,
        image_height=512,
        image_width=512
    )
    
    print(f"Cameras exported to: {json_path}")
    print(f"Now you can import this in Blender using the blender_import_cameras.py script")
