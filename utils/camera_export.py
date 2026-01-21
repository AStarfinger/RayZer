"""
Utility functions to export camera extrinsics and intrinsics to a format readable by Blender.
"""

import json
import numpy as np
import os
from pathlib import Path
from PIL import Image
import torch


def export_cameras_to_json(c2w, fxfycxcy, output_path, image_height=512, image_width=512):
    """
    Export camera extrinsics and intrinsics to a JSON file for Blender visualization.
    
    Args:
        c2w: torch tensor of shape [b, v, 4, 4] or [bv, 4, 4] - camera to world matrices
        fxfycxcy: torch tensor of shape [bv, 4] - [fx, fy, cx, cy] camera intrinsics (normalized or in pixels)
        output_path: str - path to save the JSON file
        image_height: int - image height in pixels
        image_width: int - image width in pixels
    """
    # Convert torch tensors to numpy - convert to float32 to handle bfloat16 and other unsupported dtypes
    if hasattr(c2w, 'cpu'):
        c2w = c2w.cpu().float().numpy()
    if hasattr(fxfycxcy, 'cpu'):
        fxfycxcy = fxfycxcy.cpu().float().numpy()
    
    # Reshape c2w if needed
    if c2w.ndim == 4:  # [b, v, 4, 4]
        b, v = c2w.shape[:2]
        c2w = c2w.reshape(b * v, 4, 4)
    
    # Reshape fxfycxcy if needed
    if fxfycxcy.ndim != 2:
        fxfycxcy = fxfycxcy.reshape(-1, 4)
    
    num_cameras = c2w.shape[0]
    
    cameras_data = {
        "num_cameras": num_cameras,
        "image_height": image_height,
        "image_width": image_width,
        "cameras": []
    }
    
    for i in range(num_cameras):
        fx, fy, cx, cy = fxfycxcy[i]
        
        # If intrinsics are normalized (values < 2), denormalize them
        if fx < 2:
            fx_pixel = float(fx) * image_width
            fy_pixel = float(fy) * image_height
            cx_pixel = float(cx) * image_width
            cy_pixel = float(cy) * image_height
        else:
            fx_pixel = float(fx)
            fy_pixel = float(fy)
            cx_pixel = float(cx)
            cy_pixel = float(cy)
        
        camera_dict = {
            "camera_id": i,
            "c2w": c2w[i].tolist(),  # 4x4 matrix as list
            "intrinsics": {
                "fx": fx_pixel,
                "fy": fy_pixel,
                "cx": cx_pixel,
                "cy": cy_pixel
            }
        }
        cameras_data["cameras"].append(camera_dict)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(cameras_data, f, indent=2)
    
    print(f"Exported {num_cameras} cameras to {output_path}")


def export_cameras_to_npz(c2w, fxfycxcy, output_path):
    """
    Export camera data to NPZ format (NumPy compressed format).
    
    Args:
        c2w: torch tensor of shape [b, v, 4, 4] or [bv, 4, 4]
        fxfycxcy: torch tensor of shape [bv, 4]
        output_path: str - path to save the NPZ file
    """
    # Convert torch tensors to numpy - convert to float32 to handle bfloat16 and other unsupported dtypes
    if hasattr(c2w, 'cpu'):
        c2w = c2w.cpu().float().numpy()
    if hasattr(fxfycxcy, 'cpu'):
        fxfycxcy = fxfycxcy.cpu().float().numpy()
    
    # Reshape if needed
    if c2w.ndim == 4:
        b, v = c2w.shape[:2]
        c2w = c2w.reshape(b * v, 4, 4)
    
    if fxfycxcy.ndim != 2:
        fxfycxcy = fxfycxcy.reshape(-1, 4)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    np.savez_compressed(output_path, c2w=c2w, intrinsics=fxfycxcy)
    print(f"Exported cameras to {output_path}")


def export_camera_images_with_indices(images, output_dir, camera_indices, prefix="camera"):
    """
    Export camera images with original camera indices as filenames.
    
    Args:
        images: torch tensor of shape [n, 3, h, w] - image tensor in range [0, 1] or [-1, 1]
        output_dir: str - directory to save the images
        camera_indices: list or array of global camera indices for each image
        prefix: str - prefix for image filenames
    """
    # Convert to numpy
    if hasattr(images, 'cpu'):
        images = images.cpu().float().numpy()
    
    # Reshape if needed
    if images.ndim == 5:  # [b, v, 3, h, w]
        b, v = images.shape[:2]
        images = images.reshape(b * v, 3, images.shape[3], images.shape[4])
    
    # Transpose from [n, 3, h, w] to [n, h, w, 3]
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    
    # Normalize to [0, 255]
    if images.max() <= 1.0:
        images = (images * 255).astype(np.uint8)
    elif images.min() < 0:  # [-1, 1] range
        images = ((images + 1) / 2 * 255).astype(np.uint8)
    else:
        images = images.astype(np.uint8)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each image with its original camera index
    image_paths = []
    for i in range(images.shape[0]):
        cam_idx = camera_indices[i]
        img_path = os.path.join(output_dir, f"{prefix}_{cam_idx:04d}.png")
        img = Image.fromarray(images[i])
        img.save(img_path)
        image_paths.append(img_path)
    
    print(f"Exported {len(image_paths)} camera images to {output_dir}")
    return image_paths


def export_camera_images(images, output_dir, prefix="camera"):
    """
    Export camera images to PNG files for use as background images in Blender.
    
    Args:
        images: torch tensor of shape [bv, 3, h, w] or [b, v, 3, h, w] - image tensor in range [0, 1] or [-1, 1]
        output_dir: str - directory to save the images
        prefix: str - prefix for image filenames
    """
    import os
    
    # Convert to numpy
    if hasattr(images, 'cpu'):
        images = images.cpu().float().numpy()
    
    # Reshape if needed
    if images.ndim == 5:  # [b, v, 3, h, w]
        b, v = images.shape[:2]
        images = images.reshape(b * v, 3, images.shape[3], images.shape[4])
    
    # Transpose from [bv, 3, h, w] to [bv, h, w, 3]
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    
    # Normalize to [0, 255]
    if images.max() <= 1.0:
        images = (images * 255).astype(np.uint8)
    elif images.min() < 0:  # [-1, 1] range
        images = ((images + 1) / 2 * 255).astype(np.uint8)
    else:
        images = images.astype(np.uint8)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each image
    image_paths = []
    for i in range(images.shape[0]):
        img_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        img = Image.fromarray(images[i])
        img.save(img_path)
        image_paths.append(img_path)
    
    print(f"Exported {len(image_paths)} camera images to {output_dir}")
    return image_paths


def export_cameras_with_images(c2w, fxfycxcy, output_dir, image_height=512, image_width=512, input_images=None, rendered_images=None, input_idx=None, target_idx=None):
    """
    Export all cameras with their corresponding images for Blender visualization.
    
    Args:
        c2w: torch tensor of shape [b, v, 4, 4] or [bv, 4, 4] - camera to world matrices (all cameras)
        fxfycxcy: torch tensor of shape [bv, 4] - [fx, fy, cx, cy] camera intrinsics (all cameras)
        output_dir: str - directory to save all files
        image_height: int - image height in pixels
        image_width: int - image width in pixels
        input_images: torch tensor of shape [b*v_input, 3, h, w] - input view images
        rendered_images: torch tensor of shape [b*v_target, 3, h, w] - rendered/target view images
        input_idx: torch tensor indicating which cameras are input views
        target_idx: torch tensor indicating which cameras are target views
    """
    # Create cameras subdirectory
    cameras_dir = os.path.join(output_dir, "cameras")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(cameras_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Export camera JSON (all cameras)
    cameras_json_path = os.path.join(cameras_dir, "cameras.json")
    export_cameras_to_json(c2w, fxfycxcy, cameras_json_path, image_height, image_width)
    
    # Convert input_idx and target_idx to numpy if needed
    if input_idx is not None and hasattr(input_idx, 'cpu'):
        input_idx = input_idx.cpu().numpy()
    if target_idx is not None and hasattr(target_idx, 'cpu'):
        target_idx = target_idx.cpu().numpy()
    
    # Flatten indices to get global positions
    input_idx_flat = None
    target_idx_flat = None
    
    if input_idx is not None:
        if input_idx.ndim == 2:  # [b, v_input]
            input_idx_flat = input_idx.flatten()
        else:
            input_idx_flat = input_idx
        input_idx_flat = [int(idx) for idx in input_idx_flat]
    
    if target_idx is not None:
        if target_idx.ndim == 2:  # [b, v_target]
            target_idx_flat = target_idx.flatten()
        else:
            target_idx_flat = target_idx
        target_idx_flat = [int(idx) for idx in target_idx_flat]
    
    # Export camera images with their original global indices
    input_image_paths = []
    rendered_image_paths = []
    
    if input_images is not None and input_idx_flat is not None:
        input_image_paths = export_camera_images_with_indices(
            input_images, images_dir, input_idx_flat, prefix="camera"
        )
    
    if rendered_images is not None and target_idx_flat is not None:
        rendered_image_paths = export_camera_images_with_indices(
            rendered_images, images_dir, target_idx_flat, prefix="camera"
        )
    
    # Update JSON to include image paths
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    num_cameras = cameras_data['num_cameras']
    
    # Create sets of indices for quick lookup
    input_set = set(input_idx_flat) if input_idx_flat else set()
    target_set = set(target_idx_flat) if target_idx_flat else set()
    
    # Add image paths to all cameras
    for i, camera in enumerate(cameras_data['cameras']):
        if i in input_set:
            # This is an input camera
            camera['image_path'] = f"images/camera_{i:04d}.png"
            camera['image_type'] = 'input'
        elif i in target_set:
            # This is a target camera with rendered image
            camera['image_path'] = f"images/camera_{i:04d}.png"
            camera['image_type'] = 'rendered'
        else:
            # This camera has no image
            camera['image_path'] = None
            camera['image_type'] = 'none'
    
    # Save updated JSON
    with open(cameras_json_path, 'w') as f:
        json.dump(cameras_data, f, indent=2)
    
    total_images = len(input_image_paths) + len(rendered_image_paths)
    print(f"Exported {num_cameras} cameras with {total_images} images ({len(input_image_paths)} input + {len(rendered_image_paths)} rendered) to {output_dir}")
