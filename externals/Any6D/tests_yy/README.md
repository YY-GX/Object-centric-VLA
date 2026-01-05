# Any6D Testing with DROID Data

## Overview

Testing Any6D's object pose estimation using DROID robot manipulation data.

**Setup**:
- Anchor: 1st frame (generate 3D mesh)
- Query: Last frame (test pose estimation)

## Data Structure

```
tests_yy/
├── data/
│   ├── 1st_frame/           # Anchor frame
│   │   ├── left000000.png   # RGB from left camera
│   │   └── depth000000.png  # Depth (uint16, millimeters)
│   ├── last_frame/          # Query frame
│   │   ├── left000637.png
│   │   └── depth000637.png
│   └── metadata.json        # Camera intrinsics
└── scripts/
    └── prepare_data.py      # Data preparation script
```

## Workflow

### Step 1: Prepare Anchor Data

**Without SAM2** (manual mask later):
```bash
cd /home/yygx/PANDA/externals/Any6D/tests_yy/scripts
python prepare_data.py --frame 1st_frame --camera left
```

**With SAM2** (auto-segment with bounding box):
```bash
python prepare_data.py --frame 1st_frame --camera left \
    --bbox 300 200 500 400
```

Output: `data/1st_frame_processed_left/`
- `color.png` - RGB image
- `depth.png` - Depth (uint16, mm)
- `K.txt` - Camera intrinsic matrix (3x3)
- `mask.png` - Object mask (if bbox provided)
- `info.json` - Metadata

### Step 2: Run Anchor Stage (Generate 3D Mesh)

```bash
cd /home/yygx/PANDA/externals/Any6D

# If you have mask already:
python run_demo.py \
    --demo_path tests_yy/data/1st_frame_processed_left \
    --save_path tests_yy/results/anchor_obj1 \
    --img_to_3d

# The script will:
# 1. (Optional) Refine mask with SAM2
# 2. Generate 3D mesh with InstantMesh
# 3. Align mesh to coordinate system
# 4. Run initial pose registration
```

Output: `tests_yy/results/anchor_obj1/`
- `mesh_obj1.obj` - Generated 3D mesh
- `final_mesh_obj1.obj` - Aligned mesh
- `obj1_initial_pose.txt` - Initial pose (4x4 matrix)

### Step 3: Prepare Query Data

```bash
cd tests_yy/scripts
python prepare_data.py --frame last_frame --camera left \
    --bbox 350 250 550 450  # Adjust to object location in last frame
```

### Step 4: Run Query Stage (Estimate Pose)

Modify `run_demo.py` or create custom script to load anchor mesh and test on last frame:

```python
import cv2
import numpy as np
import trimesh
from estimater import Any6D

# Load anchor mesh
mesh = trimesh.load('tests_yy/results/anchor_obj1/final_mesh_obj1.obj')
est = Any6D(mesh=mesh, debug_dir='tests_yy/results/query', debug=2)

# Load query frame
K = np.loadtxt('tests_yy/data/last_frame_processed_left/K.txt')
color = cv2.cvtColor(cv2.imread('tests_yy/data/last_frame_processed_left/color.png'), cv2.COLOR_BGR2RGB)
depth = cv2.imread('tests_yy/data/last_frame_processed_left/depth.png', cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
mask = cv2.imread('tests_yy/data/last_frame_processed_left/mask.png', cv2.IMREAD_GRAYSCALE).astype(bool)

# Estimate pose
pred_pose = est.register_any6d(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
print("Predicted pose:\n", pred_pose)
```

## SAM2 Segmentation

### How SAM2 Works in Any6D

**No DINOv2** - SAM2 requires **manual bounding box input**, not automatic detection.

**Function**: `running_sam_box()` in `sam2_instantmesh.py:73`

**Usage**:
1. **Provide bounding box**: [xmin, ymin, xmax, ymax]
2. SAM2 segments the object within the box
3. Returns binary mask

### Finding Bounding Box Coordinates

**Option A**: Use image viewer to find pixel coordinates

**Option B**: Interactive selection script:
```python
import cv2
import numpy as np

img = cv2.imread('data/1st_frame/left000000.png')
bbox = cv2.selectROI('Select Object', img, fromCenter=False)
cv2.destroyAllWindows()
print(f"Bounding box: {bbox[0]} {bbox[1]} {bbox[0]+bbox[2]} {bbox[1]+bbox[3]}")
```

**Option C**: Click points on image:
```bash
# Install if needed: pip install matplotlib
python -c "
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

img = Image.open('data/1st_frame/left000000.png')
fig, ax = plt.subplots()
ax.imshow(img)
plt.title('Click top-left then bottom-right corners')
coords = plt.ginput(2)
print(f'--bbox {int(coords[0][0])} {int(coords[0][1])} {int(coords[1][0])} {int(coords[1][1])}')
plt.close()
"
```

### DINOv2 in Codebase

**DINOv2 exists** but only in **InstantMesh** (3D mesh generation), NOT for object detection.

- Location: `instantmesh/src/models/encoder/dino.py`
- Purpose: Feature extraction for single-image 3D reconstruction
- Not used for detecting objects in scenes

## Camera Intrinsics

From `metadata.json`:
- **Left camera** (ext1): fx=521.84, fy=521.84, cx=638.02, cy=357.27
- **Wrist camera**: fx=730.59, fy=730.59, cx=598.55, cy=357.73

K matrix format:
```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

## Notes

- **Depth format**: uint16 millimeters (0-65535)
- **Depth scale**: Any6D uses `depth_scale=1000.0` to convert mm → meters
- **No depth conversion needed**: Your depth images are already in the correct format
- **SAM2 model**: `sam2.1_hiera_large` (checkpoint auto-loaded from `sam2/checkpoints/`)
