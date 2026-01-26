#!/usr/bin/env python3
"""
VLA Client - Interface to VLA policy server.

This module wraps the WebSocket client for querying the VLA policy server.
Supports both OpenVLA-OFT (cartesian) and Pi 0.5 (joint) models.
"""

import numpy as np
from typing import Dict, Literal
import warnings

try:
    from openpi_client import websocket_client_policy
    from openpi_client import image_tools
except ImportError:
    print("Warning: openpi_client not available. VLAClient may not work.")
    websocket_client_policy = None
    image_tools = None


# Valid model types
VLA_MODEL_TYPES = Literal["pi05", "openvla-oft"]


class VLAClient:
    """
    Client for VLA policy server via WebSocket.

    Sends observations (images + proprio) and language instruction,
    receives predicted action chunks.

    Supports two model types:
    - "pi05": Uses joint_position (7D), outputs 8D actions (7 joint velocities + gripper)
    - "openvla-oft": Uses cartesian_position (6D), outputs 7D actions (6 cartesian + gripper)
    """

    def __init__(self, host: str, port: int, model_type: str = "pi05"):
        """
        Initialize VLA client and connect to server.

        Args:
            host: VLA server hostname/IP (e.g., "localhost", "192.168.1.100")
            port: VLA server port (default 8009)
            model_type: VLA model type ("pi05" or "openvla-oft")
        """
        self.server_host = host
        self.server_port = port
        self.model_type = model_type

        # Validate model type
        if model_type not in ["pi05", "openvla-oft"]:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'pi05' or 'openvla-oft'")

        # Set expected action dimensions based on model type
        if model_type == "pi05":
            self.action_dim = 8  # 7 joint velocities + 1 gripper
            self.proprio_key = "joint_position"
        else:  # openvla-oft
            self.action_dim = 7  # 6 cartesian deltas + 1 gripper
            self.proprio_key = "cartesian_position"

        if websocket_client_policy is None:
            warnings.warn("openpi_client not available. VLAClient will not work.")
            self.policy_client = None
            return

        print(f"Connecting to VLA server at {self.server_host}:{self.server_port}...")
        print(f"   Model type: {self.model_type}")
        print(f"   Proprio key: {self.proprio_key}")
        print(f"   Action dim: {self.action_dim}")

        try:
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                self.server_host, self.server_port
            )
            print(f"Connected to VLA server")
        except Exception as e:
            print(f"Failed to connect to VLA server: {e}")
            self.policy_client = None

    def predict(
        self,
        observations: Dict,
        language_instruction: str,
        open_loop_horizon: int = 8
    ) -> np.ndarray:
        """
        Query VLA server for action chunk prediction.

        Args:
            observations: Robot observations with:
                - left_image: np.ndarray (H, W, 3) RGB
                - wrist_image: np.ndarray (H, W, 3) RGB
                - For pi05: joint_position: np.ndarray (7,) joint angles
                - For openvla-oft: cartesian_position: np.ndarray (6,) [x, y, z, roll, pitch, yaw]
                - gripper_position: np.ndarray (1,) [gripper_state]
            language_instruction: Skill language (e.g., "pick black bowl")
            open_loop_horizon: Number of actions to predict (usually 8 or 16)

        Returns:
            Action chunk: np.ndarray (H, action_dim)
            - For pi05: (H, 8) = [j1_vel, j2_vel, ..., j7_vel, gripper]
            - For openvla-oft: (H, 7) = [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if self.policy_client is None:
            raise RuntimeError("VLA client not connected. Cannot predict actions.")

        # Prepare request data based on model type
        # Both models are wrist-only: send zeros for exterior image, use wrist image
        # Only proprioceptive input differs: joint_position (pi05) vs cartesian_position (openvla-oft)
        exterior_image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.model_type == "pi05":
            proprio_value = observations["joint_position"]
        else:  # openvla-oft
            proprio_value = observations["cartesian_position"]

        # Resize images to 224x224
        if image_tools is not None:
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    exterior_image, 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    observations["wrist_image"], 224, 224
                ),
                f"observation/{self.proprio_key}": proprio_value,
                "observation/gripper_position": observations["gripper_position"],
                "prompt": language_instruction
            }
        else:
            # Fallback without image resizing
            request_data = {
                "observation/exterior_image_1_left": exterior_image,
                "observation/wrist_image_left": observations["wrist_image"],
                f"observation/{self.proprio_key}": proprio_value,
                "observation/gripper_position": observations["gripper_position"],
                "prompt": language_instruction
            }

        # Query VLA server
        try:
            response = self.policy_client.infer(request_data)
            action_chunk = response["actions"]

            # Validate shape
            if action_chunk.shape[1] != self.action_dim:
                raise ValueError(
                    f"Expected action dim {self.action_dim} for {self.model_type}, "
                    f"got {action_chunk.shape[1]}"
                )

            return action_chunk

        except Exception as e:
            print(f"VLA prediction failed: {e}")
            raise

    def close(self):
        """Close WebSocket connection."""
        if self.policy_client is not None:
            try:
                # WebsocketClientPolicy may not have explicit close method
                # Connection will close when object is destroyed
                pass
            except:
                pass


if __name__ == "__main__":
    # Test VLA client (requires server running)
    print("Testing VLAClient...\n")
    print("Note: This test requires VLA server running at localhost:8009")
    print()

    # Test Pi 0.5 client
    try:
        print("Testing Pi 0.5 model...")
        client = VLAClient("localhost", 8009, model_type="pi05")

        # Create mock observations for Pi 0.5
        mock_obs = {
            "left_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "joint_position": np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            "gripper_position": np.array([0.0])
        }

        instruction = "pick black bowl"

        print(f"Querying VLA server with instruction: '{instruction}'")
        action_chunk = client.predict(mock_obs, instruction, open_loop_horizon=8)

        print(f"Received action chunk: shape {action_chunk.shape}")
        print(f"   First action: {action_chunk[0]}")

        client.close()

    except Exception as e:
        print(f"Test failed: {e}")
        print("   Make sure VLA server is running")
