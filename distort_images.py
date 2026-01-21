import numpy as np
from PIL import Image
import os

# -------------------------------
# USER SETTINGS
# -------------------------------
input_dir  = "/mnt/home/adrianstarfinger/RayZer/dl3dv_benchmark/0a485338bbdaf19ba9090b874bb36ef0599a9c9a475a382c22903cf5981c6ea6/images_undistort"
output_dir = "/mnt/home/adrianstarfinger/RayZer/dl3dv_benchmark/0a485338bbdaf19ba9090b874bb36ef0599a9c9a475a382c22903cf5981c6ea6/images_distort"

# -------------------------------
# Camera settings
# -------------------------------

# Focal length (in mm)
focal_length_mm = 45.0   # e.g. a 50 mm lens

sensor_width_mm  = 36.0  # e.g. full-frame sensor width
# Pixel pitch (mm per pixel)
pixel_size = sensor_width_mm / 2048  # assuming 1920 pixels width

# Fisheye output resolution
width_fish  = 960  
height_fish = 540

width_persp  = 960           # output perspective image width
height_persp = 540           # output perspective image height
# Polynomial fisheye model coefficients
# θ = k0 + k1*r + k2*r^2 + k3*r^3 + k4*r^4
k0 = -0.009212
k1 = -0.005889
k2 = -0.001903
k3 = 0.000003
k4 = -2.61e-08
# ---------------------------------------------------------


def fisheye_theta_from_r(r):
    return k0 + k1*r + k2*r**2 + k3*r**3 + k4*r**4


def convert_perspective_to_fisheye(img, focal_px):
    img_np = np.array(img)
    H_p, W_p = img_np.shape[0], img_np.shape[1]

    # principal point = image center
    cx_persp = W_p / 2
    cy_persp = H_p / 2

    # fisheye output principal point
    cx_fish = width_fish / 2
    cy_fish = height_fish / 2

    # Prepare output
    ys, xs = np.indices((height_fish, width_fish))

    # Convert fisheye pixel coords → sensor coordinates (mm)
    x_sensor = (xs - cx_fish) * pixel_size
    y_sensor = (ys - cy_fish) * pixel_size

    # radius from sensor center
    r = np.sqrt(x_sensor**2 + y_sensor**2)

    # angle around the optical axis
    phi = np.arctan2(y_sensor, x_sensor)

    # fisheye θ from polynomial
    theta = fisheye_theta_from_r(r)

    # spherical → Cartesian ray direction
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)

    # project into perspective image
    u = (focal_px * (dx / dz)) + cx_persp
    v = (focal_px * (dy / dz)) + cy_persp

    # clamp to image bounds
    # Create mask for valid coordinates
    valid_mask = (u >= 0) & (u < W_p) & (v >= 0) & (v < H_p)
    
    # Use original coordinates but ensure they're within bounds for indexing
    u_clipped = np.clip(u, 0, W_p - 1)
    v_clipped = np.clip(v, 0, H_p - 1)

    # Integer neighbors for bilinear sampling
    u0 = np.floor(u_clipped).astype(int)
    v0 = np.floor(v_clipped).astype(int)
    u1 = np.clip(u0 + 1, 0, W_p - 1)
    v1 = np.clip(v0 + 1, 0, H_p - 1)

    # fractional parts
    du = u_clipped - u0
    dv = v_clipped - v0

    # bilinear interpolation
    top = img_np[v0, u0] * (1 - du)[:, :, None] + img_np[v0, u1] * du[:, :, None]
    bottom = img_np[v1, u0] * (1 - du)[:, :, None] + img_np[v1, u1] * du[:, :, None]
    out = top * (1 - dv)[:, :, None] + bottom * dv[:, :, None]
    image = Image.fromarray(out.astype(np.uint8))
    # apply alpha channel based on valid mask
    alpha = np.where(valid_mask, 255, 0).astype(np.uint8)
    image.putalpha(Image.fromarray(alpha))
    return image



# Inverse: Fish eye to Perspective
def theta_from_r(r):
    return k0 + k1*r + k2*r**2 + k3*r**3 + k4*r**4

def invert_theta_bisection(theta_target, r_min=0.0, r_max=1000.0, max_iter=50, eps=1e-6):
    """
    Robust inversion of θ(r) using bisection
    Returns r such that theta_from_r(r) = theta_target
    """
    # Expand r_max if theta_target is larger than current max
    while theta_from_r(r_max) < theta_target:
        r_max *= 2
        if r_max > 1e6:
            break

    for _ in range(max_iter):
        r_mid = 0.5 * (r_min + r_max)
        f_mid = theta_from_r(r_mid)

        if abs(f_mid - theta_target) < eps:
            return r_mid

        if f_mid < theta_target:
            r_min = r_mid
        else:
            r_max = r_mid

    return r_mid

def convert_fisheye_to_perspective(img_fish, focal_px):
    fish_np = np.array(img_fish)
    H_f, W_f = fish_np.shape[:2]

    # Principal points
    cx_fish = W_f / 2
    cy_fish = H_f / 2
    cx_persp = width_persp / 2
    cy_persp = height_persp / 2

    # Output pixel grid
    ys, xs = np.indices((height_persp, width_persp))

    # Compute normalized perspective ray directions
    dx = (xs - cx_persp) / focal_px
    dy = (ys - cy_persp) / focal_px
    dz = np.ones_like(dx)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm
    dy /= norm
    dz /= norm

    # Convert ray to spherical coordinates
    theta = np.arccos(dz)
    phi   = np.arctan2(dy, dx)

    # Vectorized inversion of θ → r using bisection per pixel
    r = np.zeros_like(theta)
    flat_theta = theta.ravel()
    flat_r = np.zeros_like(flat_theta)
    for i, t in enumerate(flat_theta):
        flat_r[i] = invert_theta_bisection(t)
    r = flat_r.reshape(theta.shape)

    # Map r and phi to fisheye image coordinates
    u = r * np.cos(phi) + cx_fish
    v = r * np.sin(phi) + cy_fish

    # Clamp to image bounds
    u_clipped = np.clip(u, 0, W_f - 1)
    v_clipped = np.clip(v, 0, H_f - 1)

    # Bilinear sampling
    u0 = np.floor(u_clipped).astype(int)
    v0 = np.floor(v_clipped).astype(int)
    u1 = np.clip(u0 + 1, 0, W_f - 1)
    v1 = np.clip(v0 + 1, 0, H_f - 1)

    du = u_clipped - u0
    dv = v_clipped - v0

    top = fish_np[v0, u0] * (1 - du)[:, :, None] + fish_np[v0, u1] * du[:, :, None]
    bottom = fish_np[v1, u0] * (1 - du)[:, :, None] + fish_np[v1, u1] * du[:, :, None]
    out = top * (1 - dv)[:, :, None] + bottom * dv[:, :, None]

    return Image.fromarray(out.astype(np.uint8))

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # perspective focal length in pixels:
    focal_px = focal_length_mm / pixel_size
    print("Focal length in pixels:", focal_px)
    
    # Get all image files from input directory
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"Processing {i+1}/{len(image_files)}: {filename}")
        
        # Load image
        img = Image.open(input_path).convert("RGB")
        
        # Convert to fisheye (distorted)
        out = convert_perspective_to_fisheye(img, focal_px)
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Set transparent pixels to white
        if out.mode == 'RGBA':
            background = Image.new('RGB', out.size, (255, 255, 255))
            background.paste(out, mask=out.split()[3])
            out = background
        
        # Save distorted image
        out.save(output_path)
        
    print(f"\nDone! Processed {len(image_files)} images.")
    print(f"Distorted images saved to: {output_dir}")
