import bpy
import json
import numpy as np
from mathutils import Matrix, Vector
import os

# ============ CONFIGURATION ============
blend_dir = os.path.dirname(bpy.data.filepath) or os.getcwd()
JSON_PATH = os.path.join(blend_dir, "cameras.json")
IMAGES_DIR = os.path.join(blend_dir, "images")
PLANE_DISTANCE = 2.0  # How far in front of the camera the image plane sits
# ======================================

def load_cameras_from_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def c2w_to_blender_matrix(c2w_matrix):
    c2w = np.array(c2w_matrix, dtype=np.float32)
    # OpenCV (RDF) to Blender (RFU) conversion
    conversion = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ], dtype=np.float32)
    blender_mat = conversion @ c2w @ np.linalg.inv(conversion)
    return Matrix(blender_mat.tolist())

def create_image_plane(camera_obj, camera_id, intrinsics, img_path):
    """Creates a textured plane parented to the camera."""
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    w = intrinsics['image_width']
    h = intrinsics['image_height']
    
    # Calculate plane size at PLANE_DISTANCE to match FOV
    plane_w = (w / fx) * PLANE_DISTANCE
    plane_h = (h / fy) * PLANE_DISTANCE

    # Create mesh and object
    mesh = bpy.data.meshes.new(f"Plane_Mesh_{camera_id}")
    plane_obj = bpy.data.objects.new(f"ImagePlane_{camera_id}", mesh)
    
    half_w = plane_w / 2
    half_h = plane_h / 2
    
    # Vertices in local camera space (Z is forward in CV, but we apply to camera)
    # Note: In Blender camera space, -Z is the viewing direction
    verts = [(-half_w, -half_h, -PLANE_DISTANCE), (half_w, -half_h, -PLANE_DISTANCE),
             (half_w, half_h, -PLANE_DISTANCE), (-half_w, half_h, -PLANE_DISTANCE)]
    faces = [(0, 1, 2, 3)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]

    mesh.from_pydata(verts, [], faces)
    
    # Setup UVs
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for i, loop in enumerate(mesh.loops):
        uv_layer.data[loop.index].uv = uvs[loop.vertex_index]

    bpy.context.scene.collection.objects.link(plane_obj)
    plane_obj.parent = camera_obj

    # --- Material Setup ---
    mat = bpy.data.materials.new(name=f"Mat_Cam_{camera_id}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    for n in nodes: nodes.remove(n)
    
    # Create shader nodes
    node_tex = nodes.new('ShaderNodeTexImage')
    node_emit = nodes.new('ShaderNodeEmission') # Use Emission so it's bright regardless of lights
    node_out = nodes.new('ShaderNodeOutputMaterial')
    
    try:
        node_tex.image = bpy.data.images.load(img_path)
    except:
        print(f"Failed to load {img_path}")

    links.new(node_tex.outputs['Color'], node_emit.inputs['Color'])
    links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])
    
    plane_obj.data.materials.append(mat)
    return plane_obj

def main():
    print(f"Loading: {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        print("Error: JSON not found")
        return

    data = load_cameras_from_json(JSON_PATH)
    scene = bpy.context.scene
    
    for cam_data in data['cameras']:
        camera_id = cam_data['camera_id']
        intrinsics = cam_data['intrinsics']
        intrinsics['image_width'] = data['image_width']
        intrinsics['image_height'] = data['image_height']
        
        # 1. Create Camera
        cam_name = f"Camera_{camera_id}"
        camera_data = bpy.data.cameras.new(cam_name)
        camera_obj = bpy.data.objects.new(cam_name, camera_data)
        scene.collection.objects.link(camera_obj)
        
        # Set Focal Length
        focal_px = intrinsics['fx']
        camera_data.lens = (36 * focal_px) / data['image_width']
        camera_data.sensor_width = 36
        
        # Set Transform
        camera_obj.matrix_world = c2w_to_blender_matrix(cam_data['c2w'])
        
        # 2. Create Image Plane
        img_filename = f"camera_{int(camera_id):04d}.png"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        
        if os.path.exists(img_path):
            create_image_plane(camera_obj, camera_id, intrinsics, img_path)
        else:
            print(f"Missing image: {img_filename}")

    print("Done! Switch viewport to 'Material Preview' (Z key) to see images.")

if __name__ == "__main__":
    main()