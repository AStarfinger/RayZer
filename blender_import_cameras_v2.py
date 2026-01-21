"""
Blender script to import cameras from JSON and set their background images.

Usage in Blender:
1. Go to Scripting tab
2. Open this file
3. Change the json_path and images_dir variables
4. Run the script

Or run from command line:
blender --python blender_import_cameras_v2.py -- --json_path cameras.json --images_dir ./images
"""

import bpy
import json
import os
import sys
import math
from pathlib import Path
import numpy as np


def load_json_cameras(json_path):
    """Load camera data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def matrix_to_blender(matrix_4x4):
    """Convert camera matrix to Blender format."""
    # matrix_4x4 is expected to be camera-to-world transformation
    mat = np.array(matrix_4x4)
    
    # Blender uses row-major, transpose if needed
    if mat.shape == (4, 4):
        return mat
    return mat.T


def import_cameras(json_path, images_dir=None, image_scale=1.0):
    """
    Import cameras from JSON file and optionally set background images.
    
    Args:
        json_path: Path to cameras.json file
        images_dir: Optional directory containing camera images
        image_scale: Scale factor for camera size visualization
    """
    # Load camera data
    camera_data = load_json_cameras(json_path)
    num_cameras = camera_data['num_cameras']
    image_height = camera_data.get('image_height', 512)
    image_width = camera_data.get('image_width', 512)
    
    print(f"Importing {num_cameras} cameras...")
    
    # Delete default camera if it exists
    if "Camera" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Camera"], do_unlink=True)
    
    # Import each camera
    for cam_idx, camera_info in enumerate(camera_data['cameras']):
        cam_id = camera_info['camera_id']
        c2w = np.array(camera_info['c2w'])
        intrinsics = camera_info['intrinsics']
        
        # Create camera object
        camera_data_obj = bpy.data.cameras.new(f"Camera_{cam_id}")
        camera_obj = bpy.data.objects.new(f"Camera_{cam_id}", camera_data_obj)
        bpy.context.collection.objects.link(camera_obj)
        
        # Set intrinsics
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        
        # Convert focal length to Blender format
        # Blender focal length is in mm, we have it in pixels
        # lens = (image_width / 2) / tan(fov/2)
        # For simplicity, use: focal_length_mm = (image_width / 2) / (fx / image_width)
        sensor_width = 36  # Standard sensor width in mm
        focal_length = (sensor_width * fx) / image_width
        
        camera_data_obj.lens = focal_length
        camera_data_obj.sensor_width = sensor_width
        camera_data_obj.lens_unit = 'MILLIMETERS'
        
        # Set camera position and rotation from c2w matrix
        # c2w is camera-to-world, Blender expects world-to-camera
        # Position is the last column of c2w (translation)
        position = c2w[:3, 3]
        rotation_matrix = c2w[:3, :3]
        
        # Convert to Blender coordinates if needed
        camera_obj.location = position
        
        # Convert rotation matrix to quaternion
        import mathutils
        mat = mathutils.Matrix(c2w)
        camera_obj.rotation_quaternion = mat.to_quaternion()
        
        # Set shift for principal point offset
        shift_x = (cx - image_width / 2) / image_width
        shift_y = (cy - image_height / 2) / image_height
        
        camera_data_obj.shift_x = shift_x
        camera_data_obj.shift_y = shift_y
        
        # Add image as background if available
        if images_dir and 'image_path' in camera_info:
            image_path = os.path.join(
                os.path.dirname(json_path), 
                camera_info['image_path']
            )
            
            if os.path.exists(image_path):
                try:
                    # Load image
                    img = bpy.data.images.load(image_path)
                    
                    # Add background image to camera
                    bg = camera_data_obj.background_images.new()
                    bg.image = img
                    bg.frame_method = 'FIT'
                    bg.scale = image_scale
                    bg.alpha = 0.5  # 50% transparency
                    
                    print(f"  Imported camera {cam_id} with background image")
                except Exception as e:
                    print(f"  Warning: Could not load image for camera {cam_id}: {e}")
            else:
                print(f"  Image not found: {image_path}")
        else:
            print(f"  Imported camera {cam_id}")
    
    print(f"Successfully imported {num_cameras} cameras!")
    print(f"Cameras are visible as background images in camera view (Numpad 0)")
    print(f"Adjust camera background image alpha and scale in camera properties")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import cameras into Blender')
    parser.add_argument('--json_path', type=str, default='cameras.json', 
                        help='Path to cameras.json file')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Directory containing camera images (optional)')
    parser.add_argument('--image_scale', type=float, default=1.0,
                        help='Scale factor for camera background images')
    
    # Handle both direct arguments and -- separated arguments
    args = sys.argv
    if '--' in args:
        args = args[args.index('--') + 1:]
    
    parsed_args = parser.parse_args(args)
    
    if os.path.exists(parsed_args.json_path):
        import_cameras(
            parsed_args.json_path, 
            parsed_args.images_dir,
            parsed_args.image_scale
        )
    else:
        print(f"Error: File not found: {parsed_args.json_path}")


if __name__ == '__main__':
    main()
