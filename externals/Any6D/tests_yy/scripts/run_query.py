#!/usr/bin/env python3
"""
Run Any6D query stage to estimate 6D pose using anchor mesh.

Usage:
    python run_query.py --query_data_path ../data/last_frame_processed_left --anchor_path ../results/anchor_obj1
"""

import os
import sys
import argparse
import json
import shutil
import trimesh
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Add Any6D root to path
ANY6D_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ANY6D_ROOT))

from estimater import Any6D
import nvdiffrast.torch as dr
from pytorch_lightning import seed_everything


def run_query_stage(query_data_path, anchor_path, output_path, obj_name='obj1'):
    """
    Run query stage to estimate 6D pose using anchor mesh.

    Args:
        query_data_path: Path to prepared query data (contains color.png, depth.png, mask.png, K.txt)
        anchor_path: Path to anchor results (contains final_mesh_obj1.obj)
        output_path: Path to save query results
        obj_name: Name for the object (default: obj1)
    """
    seed_everything(0)

    # Convert to absolute paths
    query_data_path = Path(query_data_path).resolve()
    anchor_path = Path(anchor_path).resolve()
    base_output_path = Path(output_path).resolve()
    
    # Create subfolder named after basename of query_data_path
    query_basename = query_data_path.name
    output_path = base_output_path / query_basename
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running query stage:")
    print(f"  Query data: {query_data_path}")
    print(f"  Anchor path: {anchor_path}")
    print(f"  Output: {output_path}")

    # Load anchor mesh
    anchor_mesh_path = anchor_path / f'final_mesh_{obj_name}.obj'
    if not anchor_mesh_path.exists():
        # Try alternative name
        anchor_mesh_path = anchor_path / 'final_mesh_anchor.obj'
        if not anchor_mesh_path.exists():
            raise FileNotFoundError(f"Anchor mesh not found. Tried: {anchor_path / f'final_mesh_{obj_name}.obj'} and {anchor_path / 'final_mesh_anchor.obj'}")

    print(f"\nLoading anchor mesh from: {anchor_mesh_path}")
    mesh = trimesh.load(str(anchor_mesh_path))
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Check if query data path exists and has required files
    required_files = ['color.png', 'depth.png', 'mask.png', 'K.txt']
    missing_files = []
    for file in required_files:
        if not (query_data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        error_msg = f"\n❌ Error: Missing required files in query data path '{query_data_path}':\n"
        error_msg += f"  Missing: {', '.join(missing_files)}\n\n"
        error_msg += "To prepare the query data, run:\n"
        error_msg += f"  python prepare_data.py --frame last_frame --camera left --bbox <xmin> <ymin> <xmax> <ymax>\n\n"
        error_msg += "Example (using same bbox as first frame):\n"
        error_msg += f"  python prepare_data.py --frame last_frame --camera left --bbox 789 455 883 573\n"
        raise FileNotFoundError(error_msg)

    # Load query data
    depth_scale = 1000.0
    color_img = cv2.imread(str(query_data_path / 'color.png'))
    if color_img is None:
        raise ValueError(f"Failed to load color image from: {query_data_path / 'color.png'}")
    color = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    
    depth = cv2.imread(str(query_data_path / 'depth.png'), cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to load depth image from: {query_data_path / 'depth.png'}")
    depth = depth.astype(np.float32) / depth_scale
    
    mask = cv2.imread(str(query_data_path / 'mask.png'), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask image from: {query_data_path / 'mask.png'}")
    mask = mask.astype(np.bool_)
    
    intrinsic = np.loadtxt(str(query_data_path / 'K.txt'))

    print(f"\nLoaded query data:")
    print(f"  Color shape: {color.shape}")
    print(f"  Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]m")
    print(f"  Mask: {mask.sum()} pixels")
    print(f"  Intrinsics:\n{intrinsic}")

    # Save color image to output
    Image.fromarray(color).save(output_path / 'color.png')

    print("\n=== Running pose estimation ===")

    # Initialize Any6D estimator with anchor mesh (use temp debug dir, but set debug=0 to minimize output)
    glctx = dr.RasterizeCudaContext()
    # Use a temporary debug directory that we'll ignore (debug=0 means minimal/no output)
    temp_debug_dir = output_path / '.temp_debug'
    temp_debug_dir.mkdir(parents=True, exist_ok=True)
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=str(temp_debug_dir), debug=0)

    # Estimate pose
    print("Running pose estimation...")
    pred_pose = est.register_any6d(
        K=intrinsic,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=5,
        name='query'
    )

    print("\n=== Results ===")
    print(f"Predicted pose (camera coordinates):\n{pred_pose}")
    print("\nNote: Pose is camera-centric (camera as origin).")
    print("      The pose matrix transforms points from object coordinates to camera coordinates.")

    # Load camera extrinsics from metadata
    metadata_path = query_data_path.parent / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if 'ext1_cam_extrinsics' not in metadata:
        raise KeyError(f"'ext1_cam_extrinsics' not found in metadata.json")
    
    # Convert extrinsics from [x, y, z, rx, ry, rz] to 4x4 matrix
    # ext1_cam_extrinsics represents camera pose in world coordinates
    # Format: [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z]
    # This is typically world-to-camera transformation (camera pose expressed in world frame)
    ext1_extrinsics = np.array(metadata['ext1_cam_extrinsics'])
    cam_pos = ext1_extrinsics[:3]
    cam_rotvec = ext1_extrinsics[3:]
    
    # Create world-to-camera transformation matrix (camera pose in world frame)
    cam_rot = R.from_rotvec(cam_rotvec).as_matrix()
    world_to_cam = np.eye(4)
    world_to_cam[:3, :3] = cam_rot
    world_to_cam[:3, 3] = cam_pos
    
    # Invert to get camera-to-world transformation
    cam_to_world = np.linalg.inv(world_to_cam)
    
    # Transform object pose from camera coordinates to world coordinates
    # pred_pose is object-to-camera transformation
    # world_pose = cam_to_world @ pred_pose gives object-to-world transformation
    world_pose = cam_to_world @ pred_pose
    
    print(f"\nCamera-to-world transformation:\n{cam_to_world}")
    print(f"\nPredicted pose (world coordinates):\n{world_pose}")
    
    # Save poses as JSON
    pose_data = {
        'pose_camera': pred_pose.tolist(),
        'pose_world': world_pose.tolist(),
        'camera_extrinsics': {
            'translation': cam_pos.tolist(),
            'rotation_axis_angle': cam_rotvec.tolist(),
            'matrix': cam_to_world.tolist()
        },
        'note': 'pose_camera: object-to-camera transformation (camera as origin)',
        'note2': 'pose_world: object-to-world transformation (world as origin)'
    }
    
    with open(output_path / 'query_pose.json', 'w') as f:
        json.dump(pose_data, f, indent=2)
    
    # Visualize pose in camera coordinates
    print("\nCreating visualization with coordinate axes (camera coordinates)...")
    vis_image_camera = visualize_pose_axes(color.copy(), pred_pose, intrinsic, axis_length=0.10, axis_labels=['X', 'Y', 'Z'])
    Image.fromarray(vis_image_camera).save(output_path / 'color_with_axes_camera.png')
    
    # Visualize pose in world coordinates
    # For world visualization, show the world coordinate frame axes at the object's location
    print("Creating visualization with coordinate axes (world coordinates)...")
    # Get object position in world coordinates and create world frame axes pose
    obj_pos_world = world_pose[:3, 3]
    # Create a pose representing world frame axes at object position (identity rotation = world frame)
    world_axes_pose_world = np.eye(4)
    world_axes_pose_world[:3, 3] = obj_pos_world
    # Transform to camera coordinates
    world_axes_pose_camera = world_to_cam @ world_axes_pose_world
    vis_image_world = visualize_pose_axes(color.copy(), world_axes_pose_camera, intrinsic, axis_length=0.10, axis_labels=['WX', 'WY', 'WZ'])
    Image.fromarray(vis_image_world).save(output_path / 'color_with_axes_world.png')

    # Clean up temporary debug directory if it exists
    if temp_debug_dir.exists():
        try:
            shutil.rmtree(temp_debug_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

    print(f"\nSaved outputs to: {output_path}")
    print(f"  - query_pose.json: Estimated 6D pose in both camera and world coordinates")
    print(f"  - color.png: Raw query RGB image")
    print(f"  - color_with_axes_camera.png: Color image with axes in camera coordinates")
    print(f"  - color_with_axes_world.png: Color image with axes in world coordinates")

    print("\n✓ Query stage complete!")

    return pred_pose


def visualize_pose_axes(image, pose, K, axis_length=0.10, axis_labels=['X', 'Y', 'Z']):
    """
    Draw coordinate axes on image using a pose in camera coordinates.
    
    Args:
        image: RGB image (H, W, 3)
        pose: 4x4 transformation matrix (in camera coordinates)
        K: 3x3 camera intrinsic matrix
        axis_length: Length of axes in meters
        axis_labels: Labels for X, Y, Z axes (default: ['X', 'Y', 'Z'])
    
    Returns:
        Image with axes drawn (RGB)
    """
    # Define 3D points for axes (in local coordinate system, homogeneous)
    origin = np.array([0, 0, 0, 1])
    x_axis = np.array([axis_length, 0, 0, 1])
    y_axis = np.array([0, axis_length, 0, 1])
    z_axis = np.array([0, 0, axis_length, 1])
    
    # Project points to 2D
    def project_point(point_3d_homogeneous, pose_matrix, K):
        # Transform from local coordinates to camera coordinates
        point_cam_homogeneous = pose_matrix @ point_3d_homogeneous
        point_cam = point_cam_homogeneous[:3]
        
        if point_cam[2] <= 0:  # Behind camera
            return None
        
        # Project to image plane
        point_2d_homogeneous = K @ point_cam
        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]  # Normalize
        return point_2d.astype(int)
    
    # Project all points
    origin_2d = project_point(origin, pose, K)
    x_end_2d = project_point(x_axis, pose, K)
    y_end_2d = project_point(y_axis, pose, K)
    z_end_2d = project_point(z_axis, pose, K)
    
    if origin_2d is None:
        print("Warning: Origin is behind camera, skipping axis visualization")
        return image
    
    # Draw axes with colors: X=red, Y=green, Z=blue
    vis = image.copy()
    line_thickness = 3
    
    if x_end_2d is not None:
        cv2.line(vis, tuple(origin_2d), tuple(x_end_2d), (255, 0, 0), line_thickness)  # Red for X
        cv2.circle(vis, tuple(x_end_2d), 5, (255, 0, 0), -1)
        cv2.putText(vis, axis_labels[0], tuple(x_end_2d + np.array([5, 5])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    if y_end_2d is not None:
        cv2.line(vis, tuple(origin_2d), tuple(y_end_2d), (0, 255, 0), line_thickness)  # Green for Y
        cv2.circle(vis, tuple(y_end_2d), 5, (0, 255, 0), -1)
        cv2.putText(vis, axis_labels[1], tuple(y_end_2d + np.array([5, 5])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if z_end_2d is not None:
        cv2.line(vis, tuple(origin_2d), tuple(z_end_2d), (0, 0, 255), line_thickness)  # Blue for Z
        cv2.circle(vis, tuple(z_end_2d), 5, (0, 0, 255), -1)
        cv2.putText(vis, axis_labels[2], tuple(z_end_2d + np.array([5, 5])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw origin
    cv2.circle(vis, tuple(origin_2d), 6, (255, 255, 255), -1)
    cv2.circle(vis, tuple(origin_2d), 6, (0, 0, 0), 2)
    
    return vis


def main():
    parser = argparse.ArgumentParser(description="Run Any6D query stage")
    parser.add_argument('--query_data_path', type=str, required=True,
                       help='Path to prepared query data directory (contains color.png, depth.png, mask.png, K.txt)')
    parser.add_argument('--anchor_path', type=str, required=True,
                       help='Path to anchor results directory (contains final_mesh_obj1.obj)')
    parser.add_argument('--output_path', type=str, default='../results/6d_pose_prediction',
                       help='Path to save query results (default: ../results/6d_pose_prediction)')
    parser.add_argument('--obj_name', type=str, default='obj1',
                       help='Name for the object (default: obj1)')

    args = parser.parse_args()

    run_query_stage(
        query_data_path=args.query_data_path,
        anchor_path=args.anchor_path,
        output_path=args.output_path,
        obj_name=args.obj_name
    )


if __name__ == '__main__':
    main()

