#!/usr/bin/env python3
"""
Camera Utilities for Real Robot Pipeline.

Provides functions for camera data extraction and pose transformations.
"""

import numpy as np
from typing import Dict
from scipy.spatial.transform import Rotation as R


def get_camera_data(
    obs: Dict,
    camera_role: str,
    robot_config: Dict,
    data_type: str = 'image',
    side: str = 'left'
) -> np.ndarray:
    """
    Extract camera data by role and convert depth to meters.

    Args:
        obs: Observation dict from robot_env
        camera_role: 'left' or 'wrist'
        robot_config: Robot configuration dict
        data_type: 'image' or 'depth'
        side: 'left' or 'right'

    Returns:
        Image (H,W,3) or depth in meters (H,W) as float32
    """
    serial = robot_config['cameras'][f'{camera_role}_camera_id']
    key = f'{serial}_{side}'
    data = obs[data_type][key]

    # Convert depth from millimeters to meters
    if data_type == 'depth':
        data = data.astype(np.float32) / 1000.0

    return data


def transform_pose_camera_to_base(
    pose_cam: np.ndarray,
    camera_role: str,
    robot_config: Dict
) -> np.ndarray:
    """
    Transform object pose from camera frame to robot base frame.

    Args:
        pose_cam: 4x4 pose matrix in camera frame (from FoundationPose)
        camera_role: 'left' or 'wrist'
        robot_config: Robot configuration dict

    Returns:
        4x4 pose matrix in robot base frame
    """
    # Get camera extrinsics [x, y, z, roll, pitch, yaw]
    extrinsics_key = f'{camera_role}_camera'
    extrinsics = robot_config['camera_extrinsics'][extrinsics_key]

    # Build T_base_from_cam (4x4 transformation matrix)
    T_base_from_cam = np.eye(4, dtype=np.float32)
    T_base_from_cam[:3, :3] = R.from_euler('xyz', extrinsics[3:6]).as_matrix()
    T_base_from_cam[:3, 3] = extrinsics[:3]

    # Transform pose from camera frame to base frame
    pose_base = T_base_from_cam @ pose_cam

    return pose_base


def get_camera_intrinsics(camera_role: str, robot_config: Dict) -> np.ndarray:
    """
    Get camera intrinsic matrix for specified camera.

    Args:
        camera_role: 'left' or 'wrist'
        robot_config: Robot configuration dict

    Returns:
        3x3 camera intrinsic matrix
    """
    intrinsics_key = f'{camera_role}_camera'
    intrinsics = robot_config['camera_intrinsics'][intrinsics_key]

    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ], dtype=np.float32)

    return K
