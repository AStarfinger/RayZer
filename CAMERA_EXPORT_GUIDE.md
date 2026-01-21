# Camera Export with Images for Blender

This guide explains how to export cameras with their corresponding images for visualization in Blender.

## What's Exported

When you enable camera export, the system creates:

```
camera_export/
├── cameras/
│   └── cameras.json          # Camera poses and intrinsics with image paths
└── images/
    ├── camera_0000.png
    ├── camera_0001.png
    ├── camera_0002.png
    └── ...
```

## How to Use

### Step 1: Enable Camera Export in Inference

Add `inference.export_cameras = true` to your inference command:

```bash
python inference.py --config "configs/rayzer_dl3dv.yaml" \
    training.dataset_path = "./data/dl3dv10k_one_scene.txt" \
    training.batch_size_per_gpu = 1 \
    training.num_views = 24 \
    training.num_input_views = 16 \
    training.num_target_views = 8 \
    inference.if_inference = true \
    inference.model_path = ./model_checkpoints/rayzer_dl3dv_8_12_12_96k.pt \
    inference_out_root = ./experiments/evaluation/test \
    inference.export_cameras = true
```

The cameras and images will be saved to:
```
./experiments/evaluation/test/rayzer_dl3dv_two_frame_mine/camera_export/
```

### Step 2: Import into Blender

#### Option A: From Blender Scripting Console

1. Open Blender
2. Go to **Scripting** tab
3. Click **+ New** to create a new text file
4. Copy the contents of `blender_import_cameras_v2.py`
5. Set the paths at the top:
   ```python
   json_path = "/path/to/camera_export/cameras/cameras.json"
   images_dir = "/path/to/camera_export/images"
   ```
6. Click **Run Script**

#### Option B: From Blender Text Editor

1. Open `blender_import_cameras_v2.py` in Blender's text editor
2. Adjust the paths
3. Run with **Alt+P**

#### Option C: Command Line

```bash
blender your_scene.blend --python blender_import_cameras_v2.py -- \
    --json_path ./camera_export/cameras/cameras.json \
    --images_dir ./camera_export/images
```

### Step 3: Visualize in Blender

1. **View Cameras in 3D**: Press **0** (numpad) to view from camera, or select a camera in the outliner
2. **See Background Images**: In camera view (numpad 0), you'll see the background image
3. **Adjust Opacity**: 
   - Select the camera
   - Go to **Object Data Properties** → **Background Images**
   - Adjust **Alpha** slider (0-1) for transparency

## Camera Properties Imported

### Extrinsics
- **Position**: XYZ location in world space
- **Rotation**: Quaternion rotation from c2w matrix

### Intrinsics
- **Focal Length**: Computed from fx/fy
- **Principal Point**: cx/cy as shift parameters
- **Sensor Width**: Set to standard 36mm

### Images
- **Background Images**: Each camera has its corresponding rendered/target image
- **Scale**: Set to FIT by default (adjustable)
- **Alpha**: Set to 50% transparency (adjustable)

## Interpreting Camera Poses

The cameras are positioned and oriented according to the predicted camera poses. You can:

1. **Check Consistency**: Do the cameras form a reasonable viewpoint distribution?
2. **Verify Geometry**: Do camera positions match the 3D scene geometry?
3. **Inspect Intrinsics**: Do focal lengths vary as expected?
4. **Visualize Coverage**: Which areas of the scene are covered by cameras?

## Troubleshooting

### Images don't appear in Blender
- Check that image files exist in the `images/` directory
- Try increasing the **Alpha** value in camera properties
- Check that **Background Images** panel shows the images
- Make sure you're in camera view (Numpad 0)

### Camera positions look wrong
- Check the c2w matrices in `cameras.json`
- Verify that your scene has proper scale
- Try adjusting the world-to-camera coordinate transformation

### Focal length seems incorrect
- This depends on sensor width and pixel focal length
- Current formula: `focal_length_mm = (36mm * fx) / image_width`
- Adjust this in the Blender script if needed

## Camera JSON Format

The `cameras.json` file has this structure:

```json
{
  "num_cameras": 24,
  "image_height": 512,
  "image_width": 512,
  "cameras": [
    {
      "camera_id": 0,
      "c2w": [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 1.0]
      ],
      "intrinsics": {
        "fx": 512.5,
        "fy": 512.5,
        "cx": 256.0,
        "cy": 256.0
      },
      "image_path": "images/camera_0000.png"
    },
    ...
  ]
}
```

## Features

✅ Exports predicted camera poses with intrinsics  
✅ Saves corresponding rendered images  
✅ Automatically imports into Blender  
✅ Sets background images for visual inspection  
✅ Handles intrinsic normalization  
✅ Supports bfloat16 tensors  
✅ Flexible command-line interface  

## Next Steps

- Use Blender visualization to verify camera predictions
- Compare input camera poses with predicted poses
- Inspect individual camera views with background images
- Check for artifacts or inconsistencies
- Use this for debugging camera predictor accuracy
