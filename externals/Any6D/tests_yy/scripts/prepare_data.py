#!/usr/bin/env python3
"""
Prepare DROID data for Any6D processing.

This script:
1. Copies RGB and depth images to Any6D format
2. Creates K.txt (camera intrinsics matrix)
3. Optionally creates masks using SAM2 with bounding boxes

Usage:
    python prepare_data.py --frame 1st_frame --camera left
    python prepare_data.py --frame last_frame --camera wrist --bbox 300 200 500 400
"""

import os
import sys
import json
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add Any6D root to path
ANY6D_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ANY6D_ROOT))

from sam2_instantmesh import running_sam_box


def create_intrinsic_matrix(fx, fy, cx, cy, output_path):
    """Create and save camera intrinsic matrix."""
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    np.savetxt(output_path, K)
    print(f"Saved intrinsic matrix to: {output_path}")
    return K


def prepare_any6d_data(
    frame_dir,
    output_dir,
    camera='left',
    metadata_path=None,
    bbox=None,
    depth_scale=1000.0
):
    """
    Prepare data for Any6D.

    Args:
        frame_dir: Directory containing left*.png and depth*.png
        output_dir: Output directory for Any6D format
        camera: 'left' or 'wrist'
        metadata_path: Path to metadata.json with intrinsics
        bbox: Bounding box [xmin, ymin, xmax, ymax] for SAM2 segmentation
        depth_scale: Depth units to meters conversion (default: 1000.0 for mm)
    """
    frame_dir = Path(frame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find RGB and depth images
    rgb_files = sorted(frame_dir.glob('left*.png'))
    depth_files = sorted(frame_dir.glob('depth*.png'))

    if not rgb_files:
        raise FileNotFoundError(f"No left*.png files found in {frame_dir}")
    if not depth_files:
        raise FileNotFoundError(f"No depth*.png files found in {frame_dir}")

    rgb_path = rgb_files[0]
    depth_path = depth_files[0]

    print(f"Processing:")
    print(f"  RGB: {rgb_path}")
    print(f"  Depth: {depth_path}")

    # Read images
    color = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)

    print(f"  Image shape: {color.shape}")
    print(f"  Depth dtype: {depth.dtype}, range: [{depth.min()}, {depth.max()}]")

    # Save RGB as color.png
    color_output = output_dir / 'color.png'
    cv2.imwrite(str(color_output), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print(f"Saved color image to: {color_output}")

    # Save depth as depth.png (keep uint16 format - Any6D will convert with depth_scale)
    depth_output = output_dir / 'depth.png'
    cv2.imwrite(str(depth_output), depth)
    print(f"Saved depth image to: {depth_output}")
    print(f"  Note: Depth is in uint16 format. Any6D will convert with depth_scale={depth_scale}")

    # Load metadata and create K.txt
    if metadata_path:
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Get intrinsics for specified camera
        intrinsic_key = f"{camera}_cam_intrinsics"
        if intrinsic_key not in metadata:
            raise KeyError(f"'{intrinsic_key}' not found in metadata. Available: {list(metadata.keys())}")

        fx, fy, cx, cy = metadata[intrinsic_key]
        K = create_intrinsic_matrix(fx, fy, cx, cy, output_dir / 'K.txt')

        # Save metadata info
        info = {
            'camera': camera,
            'intrinsics': [fx, fy, cx, cy],
            'rgb_source': str(rgb_path),
            'depth_source': str(depth_path),
            'depth_scale': depth_scale,
            'depth_unit': 'millimeters'
        }
        with open(output_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)

    # Create mask using SAM2 if bbox provided
    if bbox is not None:
        print(f"\nGenerating mask with SAM2 using bbox: {bbox}")
        bbox_array = np.array(bbox)[None, :]  # Shape: (1, 4)

        # Change to Any6D root directory for SAM2 (uses relative paths for Hydra configs)
        original_cwd = os.getcwd()
        os.chdir(ANY6D_ROOT)
        try:
            mask = running_sam_box(color, box=bbox_array)
        finally:
            os.chdir(original_cwd)

        # Save mask
        mask_output = output_dir / 'mask.png'
        cv2.imwrite(str(mask_output), mask.astype(np.uint8) * 255)
        print(f"Saved mask to: {mask_output}")

        # Visualize mask on image
        vis = color.copy()
        vis[mask] = vis[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        vis_output = output_dir / 'mask_visualization.png'
        cv2.imwrite(str(vis_output), cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"Saved mask visualization to: {vis_output}")
    else:
        print("\nNo bounding box provided. Skipping mask generation.")
        print("To generate mask, provide --bbox xmin ymin xmax ymax")

    print(f"\nData preparation complete! Output directory: {output_dir}")
    print("\nNext steps:")
    if bbox is None:
        print("1. Create mask.png manually or re-run with --bbox")
    print("2. Run Any6D anchor stage: python run_demo.py --img_to_3d")


def main():
    parser = argparse.ArgumentParser(description="Prepare DROID data for Any6D")
    parser.add_argument('--frame', required=True, choices=['1st_frame', 'last_frame'],
                       help='Which frame to process')
    parser.add_argument('--camera', default='left', choices=['left', 'wrist'],
                       help='Which camera to use')
    parser.add_argument('--bbox', type=int, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                       help='Bounding box for SAM2 segmentation')
    parser.add_argument('--output', type=str,
                       help='Output directory (default: tests_yy/data/{frame}_processed)')
    parser.add_argument('--depth-scale', type=float, default=1000.0,
                       help='Depth scale (mm to meters, default: 1000.0)')

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    frame_dir = data_dir / args.frame
    metadata_path = data_dir / 'metadata.json'

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = data_dir / f'{args.frame}_processed_{args.camera}'

    # Validate paths
    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Prepare data
    prepare_any6d_data(
        frame_dir=frame_dir,
        output_dir=output_dir,
        camera=args.camera,
        metadata_path=metadata_path,
        bbox=args.bbox,
        depth_scale=args.depth_scale
    )


if __name__ == '__main__':
    main()
