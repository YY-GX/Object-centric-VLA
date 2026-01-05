# Any6D Setup Summary - Server Sync Required

## Context
Using Any6D to get 6D object poses from DROID robot manipulation data (ZED cameras, RGB-D images).

## What We've Accomplished

### 1. Local Installation (RTX 4080 - 16GB VRAM)
- ✅ Created conda environment `Any6D` with Python 3.9
- ✅ Installed PyTorch 2.4.1 + CUDA 12.1
- ✅ Fixed CUDA compilation errors in FoundationPose (modified `foundationpose/bundlesdf/mycuda/common.cu`)
- ✅ Installed all dependencies: NVDiffRast, Kaolin, PyTorch3D, SAM2, InstantMesh, bop_toolkit
- ✅ Downloaded all model checkpoints (~4.2GB total):
  - SAM2: `sam2/checkpoints/sam2.1_hiera_large.pt` (857MB)
  - FoundationPose: `foundationpose/weights/2023-10-28-18-33-37/` and `2024-01-11-20-02-45/`
  - InstantMesh: `instantmesh/ckpts/diffusion_pytorch_model.bin` (1.7GB) + `instant_mesh_large.ckpt` (1.5GB)

### 2. Data Preparation
- ✅ Created `tests_yy/` directory structure
- ✅ Exported DROID data from `test/Sun_Dec_28_05:33:34_2025/` with intrinsics
- ✅ Created `tests_yy/scripts/prepare_data.py` - prepares RGB-D data + SAM2 segmentation
- ✅ Successfully prepared 1st frame anchor data at `tests_yy/data/1st_frame_processed_left/`:
  - `color.png` (1.1MB) - RGB image
  - `depth.png` (456KB) - Depth (uint16, millimeters)
  - `K.txt` - Camera intrinsics (fx=521.84, fy=521.84, cx=638.02, cy=357.27)
  - `mask.png` - SAM2 segmentation mask (bbox: [789, 455, 883, 573])
  - `mask_visualization.png` - Visualization

### 3. Custom Scripts Created
- `tests_yy/scripts/prepare_data.py` - Data preparation with SAM2 segmentation
- `tests_yy/scripts/select_bbox.py` - Interactive bbox selection tool
- `tests_yy/scripts/run_anchor.py` - Run anchor stage (generate 3D mesh)
- `tests_yy/README.md` - Documentation

## Current Blocker: OOM on Local Machine

### Issue
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.00 GiB
GPU 0 has a total capacity of 15.49 GiB (RTX 4080)
```

**InstantMesh Large requires >16GB VRAM** during anchor stage (3D mesh generation).

### Memory Requirements
- **Anchor Stage** (generate 3D mesh): ~15-20GB peak → ❌ RTX 4080 insufficient
- **Query Stage** (estimate poses): ~8-10GB peak → ✅ RTX 4080 sufficient

## Solution: Run Anchor Stage on Server (A6000)

### Server Requirements
- A6000 GPU (48GB VRAM) ✓
- Conda environment with Any6D installed
- Same data and scripts

### Server Setup Steps

1. **Fix editable package paths** (after conda pack transfer):
```bash
conda activate Any6D
cd /mnt/arc/yygx/pkgs_baselines/Any6D

# Reinstall SAM2
cd sam2 && pip install -e . && cd ..

# Reinstall FoundationPose CUDA extensions (CRITICAL)
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 \
    bash foundationpose/build_all_conda.sh

# Verify
python -c "from sam2.modeling.sam2_base import SAM2Base; print('✓ SAM2 working')"
```

2. **Run anchor stage**:
```bash
cd tests_yy/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_anchor.py \
    --data_path ../data/1st_frame_processed_left \
    --output_path ../results/anchor_obj1 \
    --img_to_3d
```

3. **Expected output** (anchor_obj1/):
- `mesh_obj1.obj` - Generated mesh from InstantMesh
- `center_mesh_obj1.obj` - Aligned mesh
- `final_mesh_obj1.obj` - Final mesh
- `obj1_initial_pose.txt` - Initial 6D pose (4x4 matrix)
- `K.txt` - Camera intrinsics

4. **Copy results back to local**:
```bash
scp -r /mnt/arc/yygx/pkgs_baselines/Any6D/tests_yy/results/anchor_obj1 \
    user@local:/home/yygx/PANDA/externals/Any6D/tests_yy/results/
```

## After Server Anchor Stage

Continue on **local RTX 4080** for query stage:
- Prepare last frame data: `tests_yy/data/last_frame_processed_left/`
- Run query stage using anchor mesh to estimate 6D poses
- Query stage fits in 16GB VRAM ✓

## Key Files Modified

1. `foundationpose/bundlesdf/mycuda/common.cu` - Fixed CUDA device function errors
2. `foundationpose/build_all_conda.sh` - Added `--no-build-isolation` flag
3. `tests_yy/scripts/prepare_data.py` - Changes to Any6D root dir for SAM2/Hydra

## Data Source
- Original: `/home/yygx/PANDA/test/Sun_Dec_28_05:33:34_2025/`
- Task: "Pick red cup and place it on the plate"
- Cameras: Left exterior (ext1_cam: 36087771), Wrist (16478870)
- Total frames: 638

## Next Steps on Server
1. Fix SAM2/FoundationPose imports (reinstall editable packages)
2. Run anchor stage with A6000
3. Transfer generated mesh back to local
4. Continue query stage locally
