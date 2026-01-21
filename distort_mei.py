import cv2
import numpy as np
import os
import glob

def generate_mei_maps(width, height, xi, k1, k2, p1, p2, gamma1, gamma2, u0, v0):
    # 1. Create pixel grid for the target fisheye image
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 2. Normalize coordinates
    x = (u - u0) / gamma1
    y = (v - v0) / gamma2
    rho2 = x**2 + y**2
    
    # 3. Apply radial and tangential distortion (Mei model specifics)
    # Applying the distortion coefficients to the normalized plane
    radial = 1 + k1 * rho2 + k2 * rho2**2
    x_dist = x * radial + 2 * p1 * x * y + p2 * (rho2 + 2 * x**2)
    y_dist = y * radial + p1 * (rho2 + 2 * y**2) + 2 * p2 * x * y
    
    # 4. Map to Sphere and then back to Perspective
    # Here we solve for the source coordinates. 
    # For a simple "re-distort", we map these distorted points back to a 
    # virtual pinhole camera space.
    map_x = x_dist.astype(np.float32) * gamma1 + u0
    map_y = y_dist.astype(np.float32) * gamma2 + v0
    
    return map_x, map_y

def process_directory(source_dir, target_dir):
    # KITTI-360 MEI Parameters
    # Scaled KITTI-360 MEI Parameters for 960x540 input
    params = {
        'xi': 2.5535139132482758,
        'k1': 8.9370396274089505e-02,
        'k2': 8.5068455478645308,
        'p1': 1.3477698472982495e-03,
        'p2': -7.0340482615055284e-04,
        'gamma1': 1018.58, 
        'gamma2': 572.76,  
        'u0': 479.23,      
        'v0': 269.28,      
        'width': 960,
        'height': 540
    }

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print("Generating distortion maps...")
    map_x, map_y = generate_mei_maps(**params)

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(source_dir, ext)))

    print(f"Processing {len(images)} images...")
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Apply the remapping
        # BORDER_CONSTANT fills the areas outside the perspective frustum with black
        distorted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(target_dir, filename), distorted_img)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    src = "/mnt/home/adrianstarfinger/RayZer/dl3dv_benchmark/6d81c5ab0d480fd43d78b75ff372a8113ad38e2c03f1d69627c009883054d4c2/images_undistort"
    dst = "/mnt/home/adrianstarfinger/RayZer/dl3dv_benchmark/6d81c5ab0d480fd43d78b75ff372a8113ad38e2c03f1d69627c009883054d4c2/images_distort"
    process_directory(src, dst)