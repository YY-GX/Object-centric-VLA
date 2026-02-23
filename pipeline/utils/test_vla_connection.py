#!/usr/bin/env python3
"""
VLA Server Connection Test Script.

Tests the connection to the VLA policy server and diagnoses common issues.

Usage:
    python utils/test_vla_connection.py [--host HOST] [--port PORT] [--model MODEL]

Examples:
    python utils/test_vla_connection.py
    python utils/test_vla_connection.py --host 192.168.1.100 --port 8000
    python utils/test_vla_connection.py --model openvla-oft
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np


def test_connection(host: str, port: int, model_type: str, num_tests: int = 3):
    """
    Test VLA server connection with multiple inference requests.

    Args:
        host: Server hostname/IP
        port: Server port
        model_type: "pi05" or "openvla-oft"
        num_tests: Number of test inferences to run
    """
    print("=" * 60)
    print("VLA Server Connection Test")
    print("=" * 60)
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Model: {model_type}")
    print("=" * 60)
    print()

    # Step 1: Check if openpi_client is available
    print("[1/4] Checking openpi_client installation...")
    try:
        from openpi_client import websocket_client_policy
        from openpi_client import image_tools
        print("      openpi_client found")
    except ImportError as e:
        print(f"      FAILED: openpi_client not installed: {e}")
        print("\n      Install with: pip install openpi-client")
        return False

    # Step 2: Test basic connection
    print("\n[2/4] Testing WebSocket connection...")
    start = time.time()
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host, port)
        connect_time = time.time() - start
        print(f"      Connected in {connect_time:.2f}s")
    except Exception as e:
        print(f"      FAILED: Cannot connect to {host}:{port}")
        print(f"      Error: {e}")
        print("\n      Possible causes:")
        print("        - VLA server not running")
        print("        - Wrong host/port")
        print("        - Firewall blocking connection")
        print("        - Network issues")
        return False

    # Step 3: Prepare test data
    print("\n[3/4] Preparing test inference data...")

    # Create mock observations
    mock_wrist_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_exterior_image = np.zeros((224, 224, 3), dtype=np.uint8)

    if model_type == "pi05":
        proprio_key = "joint_position"
        proprio_value = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    else:
        proprio_key = "cartesian_position"
        proprio_value = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0])

    request_data = {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            mock_exterior_image, 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            mock_wrist_image, 224, 224
        ),
        f"observation/{proprio_key}": proprio_value,
        "observation/gripper_position": np.array([0.0]),
        "prompt": "pick the object"
    }
    print("      Test data prepared")

    # Step 4: Run test inferences
    print(f"\n[4/4] Running {num_tests} test inferences...")
    latencies = []

    for i in range(num_tests):
        print(f"      Test {i+1}/{num_tests}...", end=" ", flush=True)
        start = time.time()
        try:
            response = client.infer(request_data)
            latency = time.time() - start
            latencies.append(latency)

            action_shape = response["actions"].shape
            print(f"OK ({latency:.2f}s, action shape: {action_shape})")

        except Exception as e:
            print(f"FAILED")
            print(f"\n      Error: {e}")
            print("\n      Possible causes:")
            print("        - Server overloaded/GPU memory full")
            print("        - Model not loaded correctly")
            print("        - Keepalive ping timeout (server too slow)")
            print("        - Network instability")
            return False

    # Summary
    print("\n" + "=" * 60)
    print("TEST PASSED")
    print("=" * 60)
    print(f"  Latencies: {[f'{l:.2f}s' for l in latencies]}")
    print(f"  Average:   {np.mean(latencies):.2f}s")
    print(f"  Min:       {np.min(latencies):.2f}s")
    print(f"  Max:       {np.max(latencies):.2f}s")
    print("=" * 60)

    # Warnings
    avg_latency = np.mean(latencies)
    if avg_latency > 2.0:
        print("\n WARNING: High latency detected (>2s)")
        print("          This may cause timeout issues during execution.")
        print("          Possible causes:")
        print("            - GPU overloaded")
        print("            - Model too large for GPU memory")
        print("            - Network congestion")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test VLA server connection")
    parser.add_argument("--host", type=str, default=None,
                        help="VLA server host (default: from config)")
    parser.add_argument("--port", type=int, default=None,
                        help="VLA server port (default: from config)")
    parser.add_argument("--model", type=str, default=None, choices=["pi05", "openvla-oft"],
                        help="VLA model type (default: from config)")
    parser.add_argument("--num-tests", type=int, default=3,
                        help="Number of test inferences (default: 3)")

    args = parser.parse_args()

    # Load defaults from config if not specified
    config_path = Path(__file__).parent.parent / "config" / "real_robot_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        default_host = config.get("vla_server", {}).get("host", "localhost")
        default_port = config.get("vla_server", {}).get("port", 8000)
        default_model = config.get("vla_model", "pi05")
    else:
        default_host = "localhost"
        default_port = 8000
        default_model = "pi05"

    host = args.host or default_host
    port = args.port or default_port
    model = args.model or default_model

    success = test_connection(host, port, model, args.num_tests)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
