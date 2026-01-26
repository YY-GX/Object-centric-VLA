#!/usr/bin/env python3
"""
Full Pipeline Video Recorder.

Records the entire pipeline execution (registration, motion planning, VLA)
from both 3rd-person and wrist camera views using a separate thread.

This uses RobotEnv as the single camera source to avoid ZED SDK conflicts.
"""

import os
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class FullPipelineVideoRecorder:
    """
    Thread-based video recorder that captures the full pipeline execution.

    Records from both 3rd-person (left) and wrist cameras continuously
    throughout the entire pipeline run.

    Usage:
        recorder = FullPipelineVideoRecorder(robot_env, robot_config, save_dir)
        recorder.start_recording()
        # ... run entire pipeline ...
        recorder.stop_and_save("task_name")
    """

    def __init__(
        self,
        robot_env: Any,
        robot_config: Dict,
        save_dir: str,
        fps: int = 15
    ):
        """
        Initialize the video recorder.

        Args:
            robot_env: RobotEnv instance (single source for camera data)
            robot_config: Robot configuration dict with camera IDs
            save_dir: Directory to save videos
            fps: Recording frame rate (default 15 to match DROID)
        """
        self.robot_env = robot_env
        self.robot_config = robot_config
        self.save_dir = Path(save_dir)
        self.fps = fps

        # Camera IDs for extraction
        self.left_camera_id = robot_config["cameras"]["left_camera_id"]
        self.wrist_camera_id = robot_config["cameras"]["wrist_camera_id"]

        # Recording state
        self.recording = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Frame storage
        self.frames_3rd: List[np.ndarray] = []
        self.frames_wrist: List[np.ndarray] = []
        self.timestamps: List[float] = []

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"FullPipelineVideoRecorder initialized:")
        print(f"   Save dir: {self.save_dir}")
        print(f"   FPS: {self.fps}")
        print(f"   Left camera: {self.left_camera_id}")
        print(f"   Wrist camera: {self.wrist_camera_id}")

    def start_recording(self) -> None:
        """Start recording in a background thread."""
        if self.recording:
            print("Warning: Recording already in progress")
            return

        # Clear previous frames
        with self._lock:
            self.frames_3rd = []
            self.frames_wrist = []
            self.timestamps = []

        self.recording = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print("Started full pipeline video recording")

    def _capture_loop(self) -> None:
        """Continuous capture loop running in background thread."""
        frame_interval = 1.0 / self.fps

        while self.recording:
            start_time = time.time()

            try:
                # Get observation from robot env
                obs = self.robot_env.get_observation()

                # Extract images
                left_img = self._extract_image(obs, self.left_camera_id)
                wrist_img = self._extract_image(obs, self.wrist_camera_id)

                # Store frames thread-safely
                with self._lock:
                    if left_img is not None:
                        self.frames_3rd.append(left_img.copy())
                    if wrist_img is not None:
                        self.frames_wrist.append(wrist_img.copy())
                    self.timestamps.append(time.time())

            except Exception as e:
                # Don't crash the thread on errors, just skip this frame
                print(f"Warning: Frame capture error: {e}")

            # Sleep to maintain target FPS
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def _extract_image(self, obs: Dict, camera_id: str) -> Optional[np.ndarray]:
        """Extract and convert image from observation dict."""
        image_obs = obs.get("image", {})

        for key, img in image_obs.items():
            if camera_id in key and "left" in key:
                # Convert BGRA to RGB: strip alpha, reverse BGR to RGB
                if img.shape[-1] == 4:
                    img = img[..., :3]  # Strip alpha
                rgb_img = img[..., ::-1]  # BGR to RGB
                return rgb_img

        return None

    def stop_and_save(self, filename_prefix: str = "full_rollout") -> Dict[str, str]:
        """
        Stop recording and save videos.

        Args:
            filename_prefix: Prefix for video filenames

        Returns:
            Dict with paths to saved videos
        """
        if not self.recording:
            print("Warning: No recording in progress")
            return {}

        # Stop the capture thread
        self.recording = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Get frames thread-safely
        with self._lock:
            frames_3rd = list(self.frames_3rd)
            frames_wrist = list(self.frames_wrist)

        saved_paths = {}

        if not frames_3rd and not frames_wrist:
            print("Warning: No frames captured")
            return saved_paths

        print(f"Saving full pipeline videos ({len(frames_3rd)} frames)...")

        # Try to import moviepy
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            print("Warning: moviepy not installed, trying cv2 fallback")
            return self._save_with_cv2(frames_3rd, frames_wrist, filename_prefix)

        # Save 3rd person video
        if frames_3rd:
            try:
                video_path = str(self.save_dir / f"{filename_prefix}_3rd_person.mp4")
                clip = ImageSequenceClip(frames_3rd, fps=self.fps)
                clip.write_videofile(video_path, codec="libx264", verbose=False, logger=None)
                saved_paths["3rd_person"] = video_path
                print(f"   Saved: {video_path}")
            except Exception as e:
                print(f"   Failed to save 3rd person video: {e}")

        # Save wrist video
        if frames_wrist:
            try:
                video_path = str(self.save_dir / f"{filename_prefix}_wrist.mp4")
                clip = ImageSequenceClip(frames_wrist, fps=self.fps)
                clip.write_videofile(video_path, codec="libx264", verbose=False, logger=None)
                saved_paths["wrist"] = video_path
                print(f"   Saved: {video_path}")
            except Exception as e:
                print(f"   Failed to save wrist video: {e}")

        # Clear frames after saving
        with self._lock:
            self.frames_3rd = []
            self.frames_wrist = []
            self.timestamps = []

        return saved_paths

    def _save_with_cv2(
        self,
        frames_3rd: List[np.ndarray],
        frames_wrist: List[np.ndarray],
        filename_prefix: str
    ) -> Dict[str, str]:
        """Fallback video saving using cv2."""
        import cv2

        saved_paths = {}

        def save_video(frames: List[np.ndarray], path: str) -> bool:
            if not frames:
                return False

            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(path, fourcc, self.fps, (width, height))

            for frame in frames:
                # Convert RGB to BGR for cv2
                bgr_frame = frame[..., ::-1]
                writer.write(bgr_frame)

            writer.release()
            return True

        # Save 3rd person video
        if frames_3rd:
            video_path = str(self.save_dir / f"{filename_prefix}_3rd_person.mp4")
            if save_video(frames_3rd, video_path):
                saved_paths["3rd_person"] = video_path
                print(f"   Saved (cv2): {video_path}")

        # Save wrist video
        if frames_wrist:
            video_path = str(self.save_dir / f"{filename_prefix}_wrist.mp4")
            if save_video(frames_wrist, video_path):
                saved_paths["wrist"] = video_path
                print(f"   Saved (cv2): {video_path}")

        return saved_paths

    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self.recording

    def get_frame_count(self) -> int:
        """Get current number of captured frames."""
        with self._lock:
            return len(self.frames_3rd)

    def __del__(self):
        """Cleanup on deletion."""
        if self.recording:
            self.recording = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)


if __name__ == "__main__":
    # Test the recorder (requires robot env)
    print("FullPipelineVideoRecorder test")
    print("Note: This test requires RobotEnv to be available")
    print()

    # Mock test without actual robot
    class MockRobotEnv:
        def get_observation(self):
            return {
                "image": {
                    "36087771_left": np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8),
                    "16478870_left": np.random.randint(0, 255, (720, 1280, 4), dtype=np.uint8),
                }
            }

    mock_config = {
        "cameras": {
            "left_camera_id": "36087771",
            "wrist_camera_id": "16478870"
        }
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = FullPipelineVideoRecorder(
            MockRobotEnv(),
            mock_config,
            tmpdir,
            fps=10
        )

        recorder.start_recording()
        print("Recording for 2 seconds...")
        time.sleep(2)

        paths = recorder.stop_and_save("test_recording")
        print(f"Saved paths: {paths}")

        # Check files exist
        for name, path in paths.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"   {name}: {size} bytes")
