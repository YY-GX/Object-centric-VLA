#!/usr/bin/env python3
"""
Logging Utils - Video recording and CSV logging.

This module provides utilities for recording videos and logging
action/state data to CSV files during task execution.
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import cv2


class VideoRecorder:
    """
    Record side-by-side video of camera views.

    Saves videos in MP4 format with configurable FPS.
    """

    def __init__(self, save_dir: Path, fps: int = 20):
        """
        Initialize video recorder.

        Args:
            save_dir: Directory to save videos
            fps: Video frame rate
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        self.frames = []
        self.video_writer = None
        self.current_filename = None

        print(f"📹 VideoRecorder initialized:")
        print(f"   Save dir: {self.save_dir}")
        print(f"   FPS: {self.fps}")

    def start_recording(self, filename: str):
        """
        Start a new video recording.

        Args:
            filename: Video filename (without extension)
        """
        self.frames = []
        self.current_filename = str(self.save_dir / f"{filename}.mp4")
        print(f"🎬 Started recording: {filename}")

    def add_frame(self, left_image: np.ndarray, wrist_image: np.ndarray):
        """
        Add side-by-side frame (left camera + wrist camera).

        Args:
            left_image: Left/external camera image (H, W, 3) RGB
            wrist_image: Wrist camera image (H, W, 3) RGB
        """
        if self.current_filename is None:
            print("⚠️  No active recording. Call start_recording() first.")
            return

        # Ensure uint8 format
        if left_image.dtype != np.uint8:
            left_image = (left_image * 255).astype(np.uint8)
        if wrist_image.dtype != np.uint8:
            wrist_image = (wrist_image * 255).astype(np.uint8)

        # Concatenate side-by-side
        combined = np.concatenate([left_image, wrist_image], axis=1)

        # Store frame (RGB format)
        self.frames.append(combined)

    def save(self):
        """Save recorded video to file."""
        if not self.frames:
            print("⚠️  No frames to save")
            return

        if self.current_filename is None:
            print("⚠️  No active recording")
            return

        # Get frame dimensions
        height, width = self.frames[0].shape[:2]

        # Create video writer (convert RGB to BGR for OpenCV)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.current_filename, fourcc, self.fps, (width, height)
        )

        # Write frames
        for frame in self.frames:
            # Convert RGB to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

        # Release writer
        self.video_writer.release()

        print(f"✅ Saved video: {self.current_filename} ({len(self.frames)} frames)")

        # Reset
        self.frames = []
        self.current_filename = None

    def __del__(self):
        """Cleanup on deletion."""
        if self.video_writer is not None:
            self.video_writer.release()


class CSVLogger:
    """
    Log action and state data to CSV files.

    Creates two CSV files:
    - actions.csv: Timestep, action values
    - states.csv: Timestep, robot state values
    """

    def __init__(self, save_path: Path, prefix: str = ""):
        """
        Initialize CSV logger.

        Args:
            save_path: Directory to save CSV files
            prefix: Filename prefix (e.g., "trial_1")
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

        # Open CSV files
        self.action_file = open(self.save_path / f"{prefix}_actions.csv", 'w', newline='')
        self.state_file = open(self.save_path / f"{prefix}_states.csv", 'w', newline='')

        self.action_writer = csv.writer(self.action_file)
        self.state_writer = csv.writer(self.state_file)

        # Write headers
        self.action_writer.writerow([
            "timestep", "dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"
        ])
        self.state_writer.writerow([
            "timestep", "x", "y", "z", "roll", "pitch", "yaw", "gripper"
        ])

        print(f"📊 CSVLogger initialized:")
        print(f"   Actions: {self.save_path}/{prefix}_actions.csv")
        print(f"   States: {self.save_path}/{prefix}_states.csv")

    def log_step(self, timestep: int, action: np.ndarray, state: Dict):
        """
        Log a single timestep.

        Args:
            timestep: Current timestep number
            action: Action array [7] = [dx, dy, dz, droll, dpitch, dyaw, gripper]
            state: State dict with:
                - cartesian_position: [x, y, z, roll, pitch, yaw]
                - gripper_position: [gripper]
        """
        # Log action
        action_row = [timestep] + action.flatten().tolist()
        self.action_writer.writerow(action_row)

        # Log state
        cartesian = state["cartesian_position"]
        gripper = state["gripper_position"]

        # Convert to list if numpy array
        if hasattr(cartesian, 'tolist'):
            cartesian = cartesian.tolist()
        if hasattr(gripper, '__getitem__'):
            gripper_val = float(gripper[0]) if hasattr(gripper, '__len__') and len(gripper) > 0 else float(gripper)
        else:
            gripper_val = float(gripper)

        if len(cartesian) == 6:
            # Full 6D state
            state_row = [timestep] + cartesian + [gripper_val]
        else:
            # Only 3D position (pad with zeros for orientation)
            state_row = [timestep] + cartesian[:3] + [0.0, 0.0, 0.0, gripper_val]

        self.state_writer.writerow(state_row)

    def close(self):
        """Close CSV files."""
        self.action_file.close()
        self.state_file.close()
        print(f"✅ Closed CSV files")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.action_file.close()
            self.state_file.close()
        except:
            pass


if __name__ == "__main__":
    # Test logging utils
    import tempfile
    import time

    print("Testing VideoRecorder and CSVLogger...\n")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test VideoRecorder
        print("Testing VideoRecorder...")
        video_recorder = VideoRecorder(tmpdir / "videos", fps=10)

        video_recorder.start_recording("test_video")

        # Add some frames
        for i in range(10):
            # Create dummy images (gradient pattern)
            left_img = np.tile(
                np.linspace(0, 255, 256).reshape(256, 1),
                (1, 256)
            ).astype(np.uint8)
            left_img = np.stack([left_img] * 3, axis=-1)

            wrist_img = np.tile(
                np.linspace(255, 0, 256).reshape(256, 1),
                (1, 256)
            ).astype(np.uint8)
            wrist_img = np.stack([wrist_img] * 3, axis=-1)

            video_recorder.add_frame(left_img, wrist_img)

        video_recorder.save()
        print()

        # Test CSVLogger
        print("Testing CSVLogger...")
        csv_logger = CSVLogger(tmpdir / "logs", prefix="test")

        # Log some steps
        for t in range(5):
            action = np.random.randn(7)
            state = {
                "cartesian_position": np.random.randn(6),
                "gripper_position": np.array([np.random.rand()])
            }
            csv_logger.log_step(t, action, state)

        csv_logger.close()
        print()

        # Check created files
        print("Created files:")
        for f in tmpdir.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(tmpdir)} ({f.stat().st_size} bytes)")
