#!/usr/bin/env python3
"""
Run Any6D anchor stage to generate 3D mesh from prepared data.

Usage:
    python run_anchor.py --data_path ../data/1st_frame_processed_left --output_path ../results/anchor_obj1
"""

import os
import sys
import argparse
import trimesh
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

# Add Any6D root to path
ANY6D_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ANY6D_ROOT))

from estimater import Any6D
from foundationpose.Utils import get_bounding_box, align_mesh_to_coordinate
import nvdiffrast.torch as dr
from pytorch_lightning import seed_everything
from sam2_instantmesh import *


def run_anchor_stage(data_path, output_path, use_img_to_3d=True, obj_name='obj1'):
    """
    Run anchor stage to generate 3D mesh and initial pose.

    Args:
        data_path: Path to prepared data (contains color.png, depth.png, mask.png, K.txt)
        output_path: Path to save results
        use_img_to_3d: Whether to use InstantMesh to generate 3D mesh from image
        obj_name: Name for the object
    """
    seed_everything(0)

    # Convert to absolute paths
    data_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running anchor stage:")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_path}")
    print(f"  Use InstantMesh: {use_img_to_3d}")

    # Load data
    depth_scale = 1000.0
    color = cv2.cvtColor(cv2.imread(str(data_path / 'color.png')), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(data_path / 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
    mask = cv2.imread(str(data_path / 'mask.png'), cv2.IMREAD_GRAYSCALE).astype(np.bool_)
    intrinsic = np.loadtxt(str(data_path / 'K.txt'))

    print(f"\nLoaded data:")
    print(f"  Color shape: {color.shape}")
    print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]m")
    print(f"  Mask: {mask.sum()} pixels")
    print(f"  Intrinsics:\n{intrinsic}")

    # Save color image to output
    Image.fromarray(color).save(output_path / 'color.png')

    if use_img_to_3d:
        print("\n=== Stage 1: Generating 3D mesh with InstantMesh ===")

        # Change to Any6D root for SAM2/InstantMesh (they use relative paths)
        original_cwd = os.getcwd()
        os.chdir(ANY6D_ROOT)

        try:
            # Optionally refine mask with SAM2
            cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
            input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
            print(f"Refining mask with SAM2, bbox: [{cmin}, {rmin}, {cmax}, {rmax}]")
            mask_refine = running_sam_box(color, input_box)

            # Preprocess image for InstantMesh
            print("Preprocessing image for InstantMesh...")
            input_image = preprocess_image(color, mask_refine, str(output_path), obj_name)

            # Generate multi-view images with diffusion
            print("Generating multi-view images with diffusion...")
            images = diffusion_image_generation(str(output_path), str(output_path), obj_name, input_image=input_image)

            # Generate 3D mesh with InstantMesh
            print("Generating 3D mesh with InstantMesh...")
            instant_mesh_process(images, str(output_path), obj_name)

            # Load and align mesh
            mesh = trimesh.load(str(output_path / f'mesh_{obj_name}.obj'))
            print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

            print("Aligning mesh to coordinate system...")
            mesh = align_mesh_to_coordinate(mesh)
            mesh.export(str(output_path / f'center_mesh_{obj_name}.obj'))

            mesh = trimesh.load(str(output_path / f'center_mesh_{obj_name}.obj'))

        finally:
            os.chdir(original_cwd)
    else:
        raise ValueError("Without --img_to_3d, you must provide a pre-existing mesh file")

    print("\n=== Stage 2: Initial pose registration ===")

    # Initialize Any6D estimator
    glctx = dr.RasterizeCudaContext()
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=str(output_path), debug=2)

    # Save intrinsics
    np.savetxt(str(output_path / 'K.txt'), intrinsic)

    # Register initial pose
    print("Running pose registration...")
    pred_pose = est.register_any6d(
        K=intrinsic,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=5,
        name='anchor'
    )

    print("\n=== Results ===")
    print(f"Predicted pose:\n{pred_pose}")

    # Save results
    np.savetxt(str(output_path / f'{obj_name}_initial_pose.txt'), pred_pose)
    est.mesh.export(str(output_path / f'final_mesh_{obj_name}.obj'))

    print(f"\nSaved outputs to: {output_path}")
    print(f"  - mesh_{obj_name}.obj: Generated mesh from InstantMesh")
    print(f"  - center_mesh_{obj_name}.obj: Aligned mesh")
    print(f"  - final_mesh_{obj_name}.obj: Final mesh")
    print(f"  - {obj_name}_initial_pose.txt: Initial 6D pose (4x4 matrix)")
    print(f"  - K.txt: Camera intrinsics")

    print("\n✓ Anchor stage complete!")
    print(f"\nNext: Use this mesh and pose for query stage on new frames")

    return pred_pose, est.mesh


def main():
    parser = argparse.ArgumentParser(description="Run Any6D anchor stage")
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to prepared data directory (contains color.png, depth.png, mask.png, K.txt)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save results')
    parser.add_argument('--obj_name', type=str, default='obj1',
                       help='Name for the object (default: obj1)')
    parser.add_argument('--img_to_3d', action='store_true',
                       help='Use InstantMesh+SAM2 to generate 3D mesh from image')

    args = parser.parse_args()

    run_anchor_stage(
        data_path=args.data_path,
        output_path=args.output_path,
        use_img_to_3d=args.img_to_3d,
        obj_name=args.obj_name
    )


if __name__ == '__main__':
    main()
