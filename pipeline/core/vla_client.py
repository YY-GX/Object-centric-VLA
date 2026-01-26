#!/usr/bin/env python3
"""
VLA Client - Interface to VLA policy server.

This module wraps the WebSocket client for querying the VLA policy server.
Based on evaluate_openvla_oft.py and WebsocketClientPolicy.
"""

import numpy as np
from typing import Dict
import warnings

try:
    from openpi_client import websocket_client_policy
    from openpi_client import image_tools
except ImportError:
    print("⚠️  Warning: openpi_client not available. VLAClient may not work.")
    websocket_client_policy = None
    image_tools = None


class VLAClient:
    """
    Client for VLA policy server via WebSocket.

    Sends observations (images + proprio) and language instruction,
    receives predicted action chunks.
    """

    def __init__(self, host: str, port: int):
        """
        Initialize VLA client and connect to server.

        Args:
            host: VLA server hostname/IP (e.g., "localhost", "192.168.1.100")
            port: VLA server port (default 8008 for OpenVLA-OFT)
        """
        self.server_host = host
        self.server_port = port

        if websocket_client_policy is None:
            warnings.warn("openpi_client not available. VLAClient will not work.")
            self.policy_client = None
            return

        print(f"🌐 Connecting to VLA server at {self.server_host}:{self.server_port}...")

        try:
            self.policy_client = websocket_client_policy.WebsocketClientPolicy(
                self.server_host, self.server_port
            )
            print(f"✅ Connected to VLA server")
        except Exception as e:
            print(f"❌ Failed to connect to VLA server: {e}")
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
                - cartesian_position: np.ndarray (6,) [x, y, z, roll, pitch, yaw]
                - gripper_position: np.ndarray (1,) [gripper_state]
            language_instruction: Skill language (e.g., "pick black bowl")
            open_loop_horizon: Number of actions to predict (usually 8 or 16)

        Returns:
            Action chunk: np.ndarray (H, 7) where 7 = [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if self.policy_client is None:
            raise RuntimeError("VLA client not connected. Cannot predict actions.")

        # Prepare request data (resize images to 224x224)
        if image_tools is not None:
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    observations["left_image"], 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    observations["wrist_image"], 224, 224
                ),
                "observation/cartesian_position": observations["cartesian_position"],
                "observation/gripper_position": observations["gripper_position"],
                "prompt": language_instruction
            }
        else:
            # Fallback without image resizing
            request_data = {
                "observation/exterior_image_1_left": observations["left_image"],
                "observation/wrist_image_left": observations["wrist_image"],
                "observation/cartesian_position": observations["cartesian_position"],
                "observation/gripper_position": observations["gripper_position"],
                "prompt": language_instruction
            }

        # Query VLA server
        try:
            response = self.policy_client.infer(request_data)
            action_chunk = response["actions"]

            # Validate shape
            assert action_chunk.shape[1] == 7, f"Expected action dim 7, got {action_chunk.shape[1]}"

            return action_chunk

        except Exception as e:
            print(f"❌ VLA prediction failed: {e}")
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
    print("Note: This test requires VLA server running at localhost:8008")
    print()

    try:
        client = VLAClient("localhost", 8008)

        # Create mock observations
        mock_obs = {
            "left_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist_image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "cartesian_position": np.array([0.4, 0.0, 0.2, 0.0, 0.0, 0.0]),
            "gripper_position": np.array([0.0])
        }

        instruction = "pick black bowl"

        print(f"Querying VLA server with instruction: '{instruction}'")
        action_chunk = client.predict(mock_obs, instruction, open_loop_horizon=8)

        print(f"✅ Received action chunk: shape {action_chunk.shape}")
        print(f"   First action: {action_chunk[0]}")

        client.close()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   Make sure VLA server is running")
