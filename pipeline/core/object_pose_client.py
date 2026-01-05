#!/usr/bin/env python3
"""
Object Pose Client - ZMQ client for FoundationPose server.

Provides clean interface for object registration and tracking.
Connects to FoundationPose server running in docker.
"""

import time
import numpy as np
import zmq
import pickle
from typing import Dict, Optional, List
from pathlib import Path


class ObjectPoseClient:
    """
    ZMQ client for FoundationPose server.

    Provides APIs:
    - register(): Register object and get initial pose
    - start_tracking(): Start tracking with registration results
    - get_pose(): Get current tracked object pose
    - end_tracking(): Stop tracking
    """

    def __init__(self, server_url: str = "localhost", port: int = 5558, timeout: int = 60000):
        """
        Initialize FoundationPose client.

        Args:
            server_url: Server hostname/IP
            port: Server port
            timeout: Request timeout in milliseconds
        """
        self.server_address = f"tcp://{server_url}:{port}"
        self.timeout = timeout

        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.connect(self.server_address)

        print(f"✓ Connected to FoundationPose server at {self.server_address}")

    def register(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        object_name: str,
        yoloe_prompt: str,
        mesh_path: str,
        debug_dir: str,
        conf: float = 0.1,
        iteration: int = 1,
        debug: int = 0
    ) -> Optional[Dict]:
        """
        Register object: segment and estimate initial 6D pose.

        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            K: Camera intrinsics (3, 3)
            object_name: Object identifier (e.g., "red_cup")
            yoloe_prompt: Text prompt for YOLOE (e.g., "red cup")
            mesh_path: Path to object mesh file
            debug_dir: Directory to save debug images
            conf: YOLOE confidence threshold
            iteration: Registration refinement iterations
            debug: Debug level (0-2)

        Returns:
            {
                'success': bool,
                'pose': [[...], [...], [...], [...]],  # 4x4 matrix as nested list
                'confidence': float,
                'mask_pixels': int,
                'message': str,
                'debug_images': {'mask_path': str, 'pose_path': str}
            }
            or None on error
        """
        request = {
            'api': 'register',
            'rgb': rgb,
            'depth': depth,
            'K': K,
            'object_name': object_name,
            'yoloe_prompt': yoloe_prompt,
            'mesh_path': mesh_path,
            'debug_dir': debug_dir,
            'conf': conf,
            'iteration': iteration,
            'debug': debug
        }

        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            return response
        except zmq.error.Again:
            print(f"✗ Registration timeout for {object_name}")
            return None
        except Exception as e:
            print(f"✗ Registration error: {e}")
            return None

    def start_tracking(
        self,
        K: np.ndarray,
        mesh_path: str,
        initial_pose: np.ndarray,
        object_name: str
    ) -> Optional[Dict]:
        """
        Start tracking object using registration results.

        Args:
            K: Camera intrinsics (3, 3)
            mesh_path: Path to object mesh file
            initial_pose: Initial 6D pose from registration (4, 4)
            object_name: Object identifier

        Returns:
            {
                'success': bool,
                'message': str,
                'object_name': str
            }
            or None on error
        """
        request = {
            'api': 'start_tracking',
            'K': K,
            'mesh_path': mesh_path,
            'initial_pose': initial_pose.tolist() if isinstance(initial_pose, np.ndarray) else initial_pose,
            'object_name': object_name
        }

        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            return response
        except zmq.error.Again:
            print(f"✗ Start tracking timeout for {object_name}")
            return None
        except Exception as e:
            print(f"✗ Start tracking error: {e}")
            return None

    def get_pose(
        self,
        object_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        iteration: int = 5
    ) -> Optional[Dict]:
        """
        Get current tracked object pose.

        Args:
            object_name: Object identifier
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            K: Camera intrinsics (3, 3)
            iteration: Tracking refinement iterations

        Returns:
            {
                'success': bool,
                'pose': [[...], [...], [...], [...]],  # 4x4 matrix
                'confidence': float,
                'timestamp': float
            }
            or None on error
        """
        request = {
            'api': 'get_pose',
            'object_name': object_name,
            'rgb': rgb,
            'depth': depth,
            'K': K,
            'iteration': iteration
        }

        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            return response
        except zmq.error.Again:
            print(f"✗ Get pose timeout")
            return None
        except Exception as e:
            print(f"✗ Get pose error: {e}")
            return None

    def end_tracking(self, object_name: str = None) -> Optional[Dict]:
        """
        Stop tracking and free resources.

        Args:
            object_name: Object to stop tracking (None = stop all)

        Returns:
            {
                'success': bool,
                'message': str
            }
            or None on error
        """
        request = {
            'api': 'end_tracking',
            'object_name': object_name
        }

        try:
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            return response
        except zmq.error.Again:
            print(f"✗ End tracking timeout")
            return None
        except Exception as e:
            print(f"✗ End tracking error: {e}")
            return None

    def close(self):
        """Close ZMQ connection."""
        self.socket.close()
        self.context.term()


if __name__ == "__main__":
    # Test client
    print("Testing ObjectPoseClient...")

    client = ObjectPoseClient("localhost", 5558)

    # Test connection with end_tracking (safe no-op)
    result = client.end_tracking()
    if result:
        print(f"✅ Connection test successful: {result['message']}")
    else:
        print(f"✗ Connection test failed")

    client.close()
