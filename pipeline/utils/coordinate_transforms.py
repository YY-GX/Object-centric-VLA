#!/usr/bin/env python3
"""
Coordinate Transforms - Transform between camera and robot frames.

This module provides coordinate transformation utilities for converting
between camera frame and robot base frame. Requires camera calibration.

NOTE: This is a placeholder implementation. You need to calibrate your
camera and fill in the actual transformation matrices.
"""

import numpy as np
from typing import Dict
from scipy.spatial.transform import Rotation as R
import warnings


def camera_to_robot_frame(
    point_camera: np.ndarray,
    camera_pose: Dict
) -> np.ndarray:
    """
    Transform point from camera frame to robot base frame.

    Args:
        point_camera: 3D point in camera frame [x, y, z]
        camera_pose: Camera pose relative to robot base with:
            - position: np.ndarray [x, y, z]
            - orientation: np.ndarray [w, x, y, z] quaternion

    Returns:
        3D point in robot base frame [x, y, z]
    """
    # Extract camera pose
    t_camera = camera_pose["position"]  # Translation from robot base to camera
    q_camera = camera_pose["orientation"]  # Quaternion (wxyz)

    # Convert quaternion to rotation matrix
    # scipy uses xyzw format
    q_xyzw = np.array([q_camera[1], q_camera[2], q_camera[3], q_camera[0]])
    R_camera = R.from_quat(q_xyzw).as_matrix()

    # Transform: p_robot = R_camera @ p_camera + t_camera
    point_robot = R_camera @ point_camera + t_camera

    return point_robot


def robot_to_camera_frame(
    point_robot: np.ndarray,
    camera_pose: Dict
) -> np.ndarray:
    """
    Transform point from robot base frame to camera frame.

    Args:
        point_robot: 3D point in robot base frame [x, y, z]
        camera_pose: Camera pose relative to robot base

    Returns:
        3D point in camera frame [x, y, z]
    """
    # Extract camera pose
    t_camera = camera_pose["position"]
    q_camera = camera_pose["orientation"]

    # Convert quaternion to rotation matrix
    q_xyzw = np.array([q_camera[1], q_camera[2], q_camera[3], q_camera[0]])
    R_camera = R.from_quat(q_xyzw).as_matrix()

    # Inverse transform: p_camera = R_camera^T @ (p_robot - t_camera)
    point_camera = R_camera.T @ (point_robot - t_camera)

    return point_camera


def pixel_to_3d_point(
    pixel_coords: np.ndarray,
    depth: float,
    camera_intrinsics: Dict
) -> np.ndarray:
    """
    Convert pixel coordinates + depth to 3D point in camera frame.

    Uses pinhole camera model: p = depth * K^{-1} @ [u, v, 1]

    Args:
        pixel_coords: Pixel coordinates [u, v] (column, row)
        depth: Depth value (meters)
        camera_intrinsics: Camera intrinsic parameters with:
            - fx, fy: Focal lengths
            - cx, cy: Principal point
            - K: 3x3 intrinsic matrix (optional, computed from f/c if not provided)

    Returns:
        3D point in camera frame [x, y, z]
    """
    # Get intrinsic matrix
    if "K" in camera_intrinsics:
        K = camera_intrinsics["K"]
    else:
        # Build K from focal length and principal point
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    # Inverse intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Homogeneous pixel coordinates
    pixel_homog = np.array([pixel_coords[0], pixel_coords[1], 1.0])

    # Unproject to 3D
    point_camera = depth * (K_inv @ pixel_homog)

    return point_camera


def get_default_camera_pose() -> Dict:
    """
    Get default/placeholder camera pose.

    This is a placeholder. You need to calibrate your camera to get
    the actual pose relative to robot base.

    Returns:
        Camera pose dict
    """
    warnings.warn(
        "Using placeholder camera pose. "
        "Calibrate your camera to get accurate transformations."
    )

    # Placeholder: camera mounted above table, looking down
    return {
        "position": np.array([0.5, 0.0, 0.6]),  # 50cm forward, 60cm up
        "orientation": np.array([0.7071, 0.7071, 0.0, 0.0])  # 90deg pitch (looking down)
    }


def get_default_camera_intrinsics() -> Dict:
    """
    Get default/placeholder camera intrinsics.

    This is a placeholder. You need to calibrate your camera to get
    accurate intrinsic parameters.

    Returns:
        Camera intrinsics dict
    """
    warnings.warn(
        "Using placeholder camera intrinsics. "
        "Calibrate your camera for accurate 3D reconstruction."
    )

    # Placeholder for typical RGB camera (640x480)
    return {
        "fx": 525.0,  # Focal length X
        "fy": 525.0,  # Focal length Y
        "cx": 320.0,  # Principal point X
        "cy": 240.0,  # Principal point Y
        "width": 640,
        "height": 480
    }


if __name__ == "__main__":
    # Test coordinate transforms
    print("Testing coordinate transforms...\n")

    # Get placeholder camera pose
    camera_pose = get_default_camera_pose()
    print(f"Camera pose (placeholder):")
    print(f"  Position: {camera_pose['position']}")
    print(f"  Orientation: {camera_pose['orientation']}")
    print()

    # Test camera to robot transform
    point_camera = np.array([0.0, 0.0, 1.0])  # 1m in front of camera
    point_robot = camera_to_robot_frame(point_camera, camera_pose)
    print(f"Point in camera frame: {point_camera}")
    print(f"Point in robot frame: {point_robot}")
    print()

    # Test reverse transform
    point_camera_back = robot_to_camera_frame(point_robot, camera_pose)
    print(f"Reverse transform: {point_camera_back}")
    print(f"Error: {np.linalg.norm(point_camera - point_camera_back):.6f}m")
    print()

    # Test pixel to 3D
    camera_intrinsics = get_default_camera_intrinsics()
    pixel_coords = np.array([320, 240])  # Center pixel
    depth = 1.0  # 1m depth
    point_3d = pixel_to_3d_point(pixel_coords, depth, camera_intrinsics)
    print(f"Pixel {pixel_coords} at depth {depth}m:")
    print(f"  3D point (camera frame): {point_3d}")
    print()

    # Transform to robot frame
    point_robot = camera_to_robot_frame(point_3d, camera_pose)
    print(f"  3D point (robot frame): {point_robot}")
