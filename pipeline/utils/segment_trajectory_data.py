#!/usr/bin/env python3
"""
Trajectory Data Segmentation Utility

Interactive tool to segment robot trajectory data (SVO, MP4, H5 files) into labeled
segments (e.g., "pick" and "place" phases).

Usage:
    # Interactive GUI mode (default)
    python segment_trajectory_data.py /path/to/trajectory_folder

    # CLI mode (for headless)
    python segment_trajectory_data.py /path/to/trajectory_folder --cli

    # Batch mode with predefined segments
    python segment_trajectory_data.py /path/to/trajectory_folder \
        --segments '[{"name": "pick", "start": 5.0, "end": 25.0},
                     {"name": "place", "start": 40.0, "end": 65.0}]'

    # Options
    --camera SERIAL    # Which camera to preview (default: external 36087771)
    --output DIR       # Output directory (default: same as input)
    --skip-svo         # Skip SVO segmentation
    --svo-to-mp4       # Convert SVO segments to MP4 format

Controls (GUI mode):
    - Space: Play/Pause
    - Left/Right arrows: Frame-by-frame navigation
    - [ / ]: Mark segment start/end
    - n: Start new segment (prompts for name)
    - s: Save segments and exit
    - q: Quit without saving
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

# Optional ZED SDK import
try:
    import pyzed.sl as sl
    PYZED_AVAILABLE = True
except ImportError:
    PYZED_AVAILABLE = False

# Debug mode path
DEBUG_TRAJECTORY_PATH = "/home/yygx/PANDA/droid/data/success/2026-01-24/debug/Sat_Jan_24_20:03:05_2026"


@dataclass
class SegmentInfo:
    """Information about a single segment of the trajectory."""
    name: str           # e.g., "pick", "place"
    start_time: float   # seconds
    end_time: float     # seconds
    start_frame: int    # video frame index
    end_frame: int      # video frame index
    start_timestep: int # H5 timestep index
    end_timestep: int   # H5 timestep index


@dataclass
class TrajectoryInfo:
    """Information about the trajectory folder structure."""
    root_path: Path
    h5_path: Optional[Path]
    metadata_paths: List[Path]
    mp4_paths: Dict[str, Path]  # serial -> path
    svo_paths: Dict[str, Path]  # serial -> path
    video_fps: float
    video_total_frames: int
    h5_total_timesteps: int


def validate_trajectory_folder(folder_path: Path) -> TrajectoryInfo:
    """
    Validate trajectory folder structure and discover files.

    Expected structure:
        {trajectory_folder}/
        ├── trajectory.h5
        ├── metadata_*.json
        └── recordings/
            ├── MP4/
            │   ├── 16478870.mp4  (wrist camera)
            │   └── 36087771.mp4  (external camera)
            └── SVO/
                ├── 16478870.svo
                └── 36087771.svo
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Trajectory folder not found: {folder_path}")

    # Find H5 file
    h5_files = list(folder_path.glob("*.h5"))
    h5_path = h5_files[0] if h5_files else None

    # Find metadata files
    metadata_paths = list(folder_path.glob("metadata*.json"))

    # Find MP4 files
    mp4_dir = folder_path / "recordings" / "MP4"
    mp4_paths = {}
    if mp4_dir.exists():
        for mp4_file in mp4_dir.glob("*.mp4"):
            serial = mp4_file.stem
            mp4_paths[serial] = mp4_file

    # Find SVO files
    svo_dir = folder_path / "recordings" / "SVO"
    svo_paths = {}
    if svo_dir.exists():
        for svo_file in svo_dir.glob("*.svo"):
            serial = svo_file.stem
            svo_paths[serial] = svo_file

    # Get video info from first MP4
    video_fps = 30.0  # default
    video_total_frames = 0
    if mp4_paths:
        first_mp4 = list(mp4_paths.values())[0]
        cap = cv2.VideoCapture(str(first_mp4))
        if cap.isOpened():
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    # Get H5 timesteps
    h5_total_timesteps = 0
    if h5_path and h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            h5_total_timesteps = get_h5_length(f)

    return TrajectoryInfo(
        root_path=folder_path,
        h5_path=h5_path,
        metadata_paths=metadata_paths,
        mp4_paths=mp4_paths,
        svo_paths=svo_paths,
        video_fps=video_fps,
        video_total_frames=video_total_frames,
        h5_total_timesteps=h5_total_timesteps
    )


def get_h5_length(hdf5_file: h5py.File, keys_to_ignore: List[str] = None) -> int:
    """Get the length (number of timesteps) from an H5 file."""
    if keys_to_ignore is None:
        keys_to_ignore = []

    length = None
    for key in hdf5_file.keys():
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_h5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            continue

        if curr_length is not None:
            if length is None:
                length = curr_length
            else:
                length = min(length, curr_length)

    return length or 0


def time_to_frame_index(time_sec: float, fps: float) -> int:
    """Convert time in seconds to video frame index."""
    return int(time_sec * fps)


def frame_to_timestep_index(frame_idx: int, total_frames: int, total_timesteps: int) -> int:
    """
    Map video frame index to H5 timestep index.
    Assumes nearly 1:1 mapping with linear interpolation.
    """
    if total_frames == 0 or total_timesteps == 0:
        return 0
    ratio = total_timesteps / total_frames
    return int(frame_idx * ratio)


def timestep_to_frame_index(timestep_idx: int, total_frames: int, total_timesteps: int) -> int:
    """Map H5 timestep index to video frame index."""
    if total_timesteps == 0 or total_frames == 0:
        return 0
    ratio = total_frames / total_timesteps
    return int(timestep_idx * ratio)


class VideoPreviewPlayer:
    """OpenCV GUI for video playback and segment marking."""

    # Color palette for segments (BGR)
    SEGMENT_COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Default segment names for auto-naming mode
    DEFAULT_SEGMENT_NAMES = ["pick", "place"]

    def __init__(self, video_path: Path, traj_info: TrajectoryInfo, auto_names: bool = True):
        self.video_path = video_path
        self.traj_info = traj_info
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame_idx = 0
        self.playing = False
        self.segments: List[SegmentInfo] = []

        # Current segment being marked
        self.current_segment_name: Optional[str] = None
        self.current_segment_start: Optional[int] = None

        # Auto-naming mode
        self.auto_names = auto_names
        self.next_auto_name_idx = 0

        self.window_name = "Trajectory Segmentation"

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay information on the frame."""
        display = frame.copy()

        # Calculate current time
        current_time = self.current_frame_idx / self.fps
        total_time = self.total_frames / self.fps

        # Status bar at top
        cv2.rectangle(display, (0, 0), (self.width, 80), (40, 40, 40), -1)

        # Time and frame info
        time_text = f"Time: {current_time:.2f}s / {total_time:.2f}s"
        frame_text = f"Frame: {self.current_frame_idx} / {self.total_frames - 1}"
        timestep = frame_to_timestep_index(
            self.current_frame_idx,
            self.traj_info.video_total_frames,
            self.traj_info.h5_total_timesteps
        )
        timestep_text = f"Timestep: {timestep} / {self.traj_info.h5_total_timesteps - 1}"

        cv2.putText(display, time_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, frame_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, timestep_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Play/Pause status
        status = "PLAYING" if self.playing else "PAUSED"
        status_color = (0, 255, 0) if self.playing else (0, 165, 255)
        cv2.putText(display, status, (self.width - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Current segment being marked
        if self.current_segment_name:
            marking_text = f"Marking: {self.current_segment_name} (start: {self.current_segment_start})"
            cv2.putText(display, marking_text, (250, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Draw segment markers on timeline at bottom
        timeline_y = self.height - 30
        timeline_height = 20
        cv2.rectangle(display, (0, timeline_y - 5),
                      (self.width, self.height), (40, 40, 40), -1)

        # Timeline background
        cv2.rectangle(display, (10, timeline_y),
                      (self.width - 10, timeline_y + timeline_height),
                      (100, 100, 100), -1)

        # Draw existing segments on timeline
        for i, seg in enumerate(self.segments):
            color = self.SEGMENT_COLORS[i % len(self.SEGMENT_COLORS)]
            x1 = int(10 + (seg.start_frame / self.total_frames) * (self.width - 20))
            x2 = int(10 + (seg.end_frame / self.total_frames) * (self.width - 20))
            cv2.rectangle(display, (x1, timeline_y), (x2, timeline_y + timeline_height),
                          color, -1)
            # Segment name
            cv2.putText(display, seg.name, (x1 + 2, timeline_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Draw current segment start marker
        if self.current_segment_start is not None:
            x = int(10 + (self.current_segment_start / self.total_frames) * (self.width - 20))
            cv2.line(display, (x, timeline_y - 5), (x, timeline_y + timeline_height + 5),
                     (0, 255, 255), 2)

        # Current position marker
        pos_x = int(10 + (self.current_frame_idx / self.total_frames) * (self.width - 20))
        cv2.line(display, (pos_x, timeline_y - 5),
                 (pos_x, timeline_y + timeline_height + 5), (255, 255, 255), 2)

        # Help text at bottom
        help_text = "SPACE:Play/Pause | </>:Frame | [:Start | ]:End | n:New | s:Save | q:Quit"
        cv2.putText(display, help_text, (10, self.height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    def get_next_auto_name(self) -> str:
        """Get the next automatic segment name."""
        if self.next_auto_name_idx < len(self.DEFAULT_SEGMENT_NAMES):
            return self.DEFAULT_SEGMENT_NAMES[self.next_auto_name_idx]
        else:
            # Beyond default names, use segment_N
            return f"segment_{self.next_auto_name_idx + 1}"

    def prompt_segment_name(self) -> Optional[str]:
        """Prompt user for segment name via terminal."""
        print("\n" + "=" * 50)
        name = input("Enter segment name (e.g., 'pick', 'place'): ").strip()
        if not name:
            print("Cancelled - no name provided")
            return None
        return name

    def mark_segment_start(self):
        """Mark the start of a new segment."""
        if self.current_segment_name is None:
            # Get segment name (auto or prompt)
            if self.auto_names:
                name = self.get_next_auto_name()
            else:
                name = self.prompt_segment_name()

            if name:
                self.current_segment_name = name
                self.current_segment_start = self.current_frame_idx
                print(f"Segment '{name}' start marked at frame {self.current_frame_idx}")
        else:
            # Update start position
            self.current_segment_start = self.current_frame_idx
            print(f"Segment '{self.current_segment_name}' start updated to frame {self.current_frame_idx}")

    def mark_segment_end(self):
        """Mark the end of the current segment."""
        if self.current_segment_name is None or self.current_segment_start is None:
            print("No segment started. Press 'n' to start a new segment first.")
            return

        if self.current_frame_idx <= self.current_segment_start:
            print("End frame must be after start frame!")
            return

        # Create segment
        start_time = self.current_segment_start / self.fps
        end_time = self.current_frame_idx / self.fps
        start_timestep = frame_to_timestep_index(
            self.current_segment_start,
            self.traj_info.video_total_frames,
            self.traj_info.h5_total_timesteps
        )
        end_timestep = frame_to_timestep_index(
            self.current_frame_idx,
            self.traj_info.video_total_frames,
            self.traj_info.h5_total_timesteps
        )

        segment = SegmentInfo(
            name=self.current_segment_name,
            start_time=start_time,
            end_time=end_time,
            start_frame=self.current_segment_start,
            end_frame=self.current_frame_idx,
            start_timestep=start_timestep,
            end_timestep=end_timestep
        )

        self.segments.append(segment)
        print(f"Segment '{self.current_segment_name}' created: "
              f"frames {self.current_segment_start}-{self.current_frame_idx}, "
              f"time {start_time:.2f}s-{end_time:.2f}s, "
              f"timesteps {start_timestep}-{end_timestep}")

        # Increment auto name index for next segment
        if self.auto_names:
            self.next_auto_name_idx += 1

        # Reset current segment
        self.current_segment_name = None
        self.current_segment_start = None

    def run(self) -> List[SegmentInfo]:
        """Run the interactive video player. Returns list of segments."""
        cv2.namedWindow(self.window_name)

        print("\n" + "=" * 60)
        print("TRAJECTORY SEGMENTATION - Interactive Mode")
        print("=" * 60)
        print(f"Video: {self.video_path}")
        print(f"Total frames: {self.total_frames}, FPS: {self.fps:.1f}")
        print(f"Total timesteps: {self.traj_info.h5_total_timesteps}")
        if self.auto_names:
            print(f"Auto-naming: {self.DEFAULT_SEGMENT_NAMES} (use --no-auto-names to disable)")
        print("\nControls:")
        print("  SPACE      - Play/Pause")
        print("  LEFT/RIGHT - Frame-by-frame (or A/D)")
        if self.auto_names:
            print(f"  [          - Mark segment start (auto: next is '{self.get_next_auto_name()}')")
        else:
            print("  [          - Mark segment start (prompts for name)")
        print("  ]          - Mark segment end")
        print("  n          - Start new segment (same as [)")
        print("  s          - Save segments and exit")
        print("  q          - Quit without saving")
        print("=" * 60)

        try:
            while True:
                frame = self.get_current_frame()
                if frame is None:
                    break

                display = self.draw_overlay(frame)
                cv2.imshow(self.window_name, display)

                # Handle keyboard input
                # Use waitKeyEx to get full key code (arrow keys need this)
                wait_time = 1 if self.playing else 30
                key_full = cv2.waitKeyEx(wait_time)
                key = key_full & 0xFF

                # Arrow key codes vary by system:
                # Left:  81 (Linux), 2424832, 65361
                # Right: 83 (Linux), 2555904, 65363
                LEFT_ARROW_CODES = {81, 2424832, 65361}
                RIGHT_ARROW_CODES = {83, 2555904, 65363}

                if key == ord(' '):  # Space - toggle play/pause
                    self.playing = not self.playing

                elif key_full in LEFT_ARROW_CODES or key == ord(',') or key == ord('a'):  # Left
                    self.playing = False
                    self.current_frame_idx = max(0, self.current_frame_idx - 1)

                elif key_full in RIGHT_ARROW_CODES or key == ord('.') or key == ord('d'):  # Right
                    self.playing = False
                    self.current_frame_idx = min(self.total_frames - 1,
                                                  self.current_frame_idx + 1)

                elif key == ord('[') or key == ord('n'):  # Start segment
                    self.mark_segment_start()

                elif key == ord(']'):  # End segment
                    self.mark_segment_end()

                elif key == ord('s'):  # Save and exit
                    print("\nSaving segments...")
                    break

                elif key == ord('q'):  # Quit without saving
                    print("\nQuit without saving")
                    self.segments = []
                    break

                # Auto-advance when playing
                if self.playing:
                    self.current_frame_idx += 1
                    if self.current_frame_idx >= self.total_frames:
                        self.current_frame_idx = self.total_frames - 1
                        self.playing = False

        finally:
            cv2.destroyAllWindows()
            self.cap.release()

        return self.segments


class CLISegmentLabeler:
    """CLI fallback for headless environments."""

    def __init__(self, traj_info: TrajectoryInfo):
        self.traj_info = traj_info

    def run(self) -> List[SegmentInfo]:
        """Run CLI-based segment labeling."""
        print("\n" + "=" * 60)
        print("TRAJECTORY SEGMENTATION - CLI Mode")
        print("=" * 60)
        print(f"Total frames: {self.traj_info.video_total_frames}")
        print(f"Total time: {self.traj_info.video_total_frames / self.traj_info.video_fps:.2f}s")
        print(f"Total timesteps: {self.traj_info.h5_total_timesteps}")
        print("=" * 60)

        segments = []

        while True:
            print("\nOptions:")
            print("  1. Add a new segment")
            print("  2. List current segments")
            print("  3. Done - save segments")
            print("  4. Quit without saving")

            choice = input("\nChoice: ").strip()

            if choice == '1':
                segment = self._add_segment()
                if segment:
                    segments.append(segment)
                    print(f"Added segment: {segment.name}")

            elif choice == '2':
                if segments:
                    print("\nCurrent segments:")
                    for i, seg in enumerate(segments):
                        print(f"  {i+1}. {seg.name}: "
                              f"{seg.start_time:.2f}s - {seg.end_time:.2f}s "
                              f"(frames {seg.start_frame}-{seg.end_frame})")
                else:
                    print("No segments defined yet.")

            elif choice == '3':
                print(f"\nSaving {len(segments)} segments...")
                break

            elif choice == '4':
                print("Quit without saving")
                return []

        return segments

    def _add_segment(self) -> Optional[SegmentInfo]:
        """Add a new segment via CLI prompts."""
        try:
            name = input("Segment name (e.g., 'pick'): ").strip()
            if not name:
                return None

            print(f"Enter times in seconds (total: {self.traj_info.video_total_frames / self.traj_info.video_fps:.2f}s)")
            start_time = float(input("Start time (seconds): "))
            end_time = float(input("End time (seconds): "))

            if end_time <= start_time:
                print("Error: End time must be after start time")
                return None

            fps = self.traj_info.video_fps
            start_frame = time_to_frame_index(start_time, fps)
            end_frame = time_to_frame_index(end_time, fps)

            start_timestep = frame_to_timestep_index(
                start_frame,
                self.traj_info.video_total_frames,
                self.traj_info.h5_total_timesteps
            )
            end_timestep = frame_to_timestep_index(
                end_frame,
                self.traj_info.video_total_frames,
                self.traj_info.h5_total_timesteps
            )

            return SegmentInfo(
                name=name,
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                start_timestep=start_timestep,
                end_timestep=end_timestep
            )

        except ValueError as e:
            print(f"Invalid input: {e}")
            return None


def segment_h5_file(
    input_path: Path,
    output_path: Path,
    start_idx: int,
    end_idx: int
) -> bool:
    """
    Segment an H5 file by slicing all datasets along the first dimension.

    Args:
        input_path: Path to source H5 file
        output_path: Path for output H5 file
        start_idx: Start timestep index (inclusive)
        end_idx: End timestep index (exclusive)

    Returns:
        True if successful
    """
    def copy_group(src_group: h5py.Group, dst_group: h5py.Group,
                   start: int, end: int, keys_to_ignore: List[str] = None):
        """Recursively copy and slice groups/datasets."""
        if keys_to_ignore is None:
            keys_to_ignore = []

        for key in src_group.keys():
            if key in keys_to_ignore:
                continue

            item = src_group[key]

            if isinstance(item, h5py.Group):
                # Create group and recurse
                new_group = dst_group.create_group(key)
                # Copy group attributes
                for attr_name, attr_value in item.attrs.items():
                    new_group.attrs[attr_name] = attr_value
                copy_group(item, new_group, start, end, keys_to_ignore)

            elif isinstance(item, h5py.Dataset):
                # Slice dataset along first dimension
                if len(item.shape) == 0:
                    # Scalar dataset - copy as is
                    dst_group.create_dataset(key, data=item[()])
                else:
                    # Slice along first dimension
                    sliced_data = item[start:end]
                    dst_group.create_dataset(key, data=sliced_data)

                # Copy dataset attributes
                for attr_name, attr_value in item.attrs.items():
                    dst_group[key].attrs[attr_name] = attr_value

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(input_path, 'r') as src:
            with h5py.File(output_path, 'w') as dst:
                # Copy root attributes
                for attr_name, attr_value in src.attrs.items():
                    dst.attrs[attr_name] = attr_value

                # Copy and slice all groups/datasets
                copy_group(src, dst, start_idx, end_idx)

        return True

    except Exception as e:
        print(f"Error segmenting H5 file: {e}")
        return False


def segment_mp4_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_time: float,
    duration: float
) -> bool:
    """
    Segment MP4 file using FFmpeg with stream copy (fast, no re-encoding).

    Args:
        input_path: Path to source MP4 file
        output_path: Path for output MP4 file
        start_time: Start time in seconds
        duration: Duration in seconds

    Returns:
        True if successful
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', str(input_path),
        '-t', str(duration),
        '-c', 'copy',
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Falling back to OpenCV re-encoding...")
        return False


def segment_mp4_opencv(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int
) -> bool:
    """
    Segment MP4 file using OpenCV (slower, re-encodes).
    Fallback when FFmpeg is not available.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Failed to open video: {input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"Failed to create video writer: {output_path}")
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()
        writer.release()
        return True

    except Exception as e:
        print(f"Error segmenting MP4 with OpenCV: {e}")
        return False


def segment_mp4_file(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    start_frame: int,
    end_frame: int
) -> bool:
    """
    Segment MP4 file, trying FFmpeg first, falling back to OpenCV.
    """
    duration = end_time - start_time

    # Try FFmpeg first (faster)
    if segment_mp4_ffmpeg(input_path, output_path, start_time, duration):
        return True

    # Fall back to OpenCV
    return segment_mp4_opencv(input_path, output_path, start_frame, end_frame)


def segment_svo_file(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    convert_to_mp4: bool = True
) -> bool:
    """
    Segment SVO file by reading frames and writing to new file.

    Note: Native SVO trimming is not supported by ZED SDK.
    We export the segment as MP4 instead.

    Args:
        input_path: Path to source SVO file
        output_path: Path for output file (MP4 or SVO)
        start_frame: Start frame index
        end_frame: End frame index
        convert_to_mp4: If True, output as MP4. If False, skip (no native SVO trimming)

    Returns:
        True if successful
    """
    if not PYZED_AVAILABLE:
        print(f"Warning: pyzed not available, skipping SVO segmentation for {input_path}")
        return False

    if not convert_to_mp4:
        print(f"Warning: Native SVO trimming not supported, skipping {input_path}")
        return False

    # Change output extension to .mp4
    output_path = output_path.with_suffix('.mp4')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize ZED SDK
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(input_path))
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open SVO file: {err}")
            return False

        # Get video properties
        sdk_version = zed.get_sdk_version()
        use_sdk_4 = sdk_version.startswith("4.")

        if use_sdk_4:
            fps = zed.get_camera_information().camera_configuration.fps
            resolution = zed.get_camera_information().camera_configuration.resolution
            width, height = resolution.width, resolution.height
        else:
            fps = zed.get_camera_information().camera_fps
            resolution = zed.get_camera_information().camera_resolution
            width, height = resolution.width, resolution.height

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"Failed to create video writer: {output_path}")
            zed.close()
            return False

        # Read and write frames
        img_container = sl.Mat()
        rt_params = sl.RuntimeParameters()

        # Seek to start frame
        zed.set_svo_position(start_frame)

        for _ in range(end_frame - start_frame):
            grabbed = zed.grab(rt_params)

            if grabbed == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(img_container, sl.VIEW.LEFT)
                rgb = cv2.cvtColor(img_container.get_data(), cv2.COLOR_RGBA2RGB)
                writer.write(rgb)
            elif use_sdk_4 and grabbed == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            else:
                break

        writer.release()
        zed.close()
        return True

    except Exception as e:
        print(f"Error segmenting SVO file: {e}")
        return False


def update_metadata(
    input_path: Path,
    output_path: Path,
    segment: SegmentInfo
) -> bool:
    """
    Copy and update metadata JSON with segment information.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load original metadata
        with open(input_path, 'r') as f:
            metadata = json.load(f)

        # Add segment information
        metadata['segment'] = asdict(segment)
        metadata['original_metadata_path'] = str(input_path)

        # Update any frame/timestep counts if present
        if 'num_timesteps' in metadata:
            metadata['original_num_timesteps'] = metadata['num_timesteps']
            metadata['num_timesteps'] = segment.end_timestep - segment.start_timestep

        # Save updated metadata
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"Error updating metadata: {e}")
        return False


class TrajectorySegmenter:
    """Main class that orchestrates the segmentation process."""

    def __init__(
        self,
        traj_folder: Path,
        output_dir: Optional[Path] = None,
        camera_serial: str = "36087771",
        skip_svo: bool = False,
        svo_to_mp4: bool = True,
        auto_names: bool = True
    ):
        self.traj_folder = traj_folder
        self.output_dir = output_dir or traj_folder
        self.camera_serial = camera_serial
        self.skip_svo = skip_svo
        self.svo_to_mp4 = svo_to_mp4
        self.auto_names = auto_names

        # Validate and discover files
        self.traj_info = validate_trajectory_folder(traj_folder)

        print("\n" + "=" * 60)
        print("TRAJECTORY INFO")
        print("=" * 60)
        print(f"Root: {self.traj_info.root_path}")
        print(f"H5 file: {self.traj_info.h5_path}")
        print(f"Metadata files: {len(self.traj_info.metadata_paths)}")
        print(f"MP4 files: {list(self.traj_info.mp4_paths.keys())}")
        print(f"SVO files: {list(self.traj_info.svo_paths.keys())}")
        print(f"Video FPS: {self.traj_info.video_fps:.1f}")
        print(f"Video frames: {self.traj_info.video_total_frames}")
        print(f"H5 timesteps: {self.traj_info.h5_total_timesteps}")
        print("=" * 60)

    def get_preview_video(self) -> Optional[Path]:
        """Get the video file to use for preview."""
        # Try specified camera first
        if self.camera_serial in self.traj_info.mp4_paths:
            return self.traj_info.mp4_paths[self.camera_serial]

        # Fall back to any available MP4
        if self.traj_info.mp4_paths:
            return list(self.traj_info.mp4_paths.values())[0]

        return None

    def run_interactive(self, cli_mode: bool = False) -> List[SegmentInfo]:
        """Run interactive segmentation (GUI or CLI mode)."""
        if cli_mode:
            labeler = CLISegmentLabeler(self.traj_info)
            return labeler.run()

        # GUI mode
        video_path = self.get_preview_video()
        if video_path is None:
            print("No MP4 files found for preview. Use --cli mode.")
            return []

        player = VideoPreviewPlayer(video_path, self.traj_info, auto_names=self.auto_names)
        return player.run()

    def apply_segments(self, segments: List[SegmentInfo]) -> bool:
        """Apply segments to create segmented output folders."""
        if not segments:
            print("No segments to apply.")
            return False

        print("\n" + "=" * 60)
        print("APPLYING SEGMENTS")
        print("=" * 60)

        for segment in segments:
            print(f"\nProcessing segment: {segment.name}")
            print(f"  Time: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
            print(f"  Frames: {segment.start_frame} - {segment.end_frame}")
            print(f"  Timesteps: {segment.start_timestep} - {segment.end_timestep}")

            # Create output folder for this segment
            segment_dir = self.output_dir / segment.name
            segment_dir.mkdir(parents=True, exist_ok=True)

            success = True

            # Segment H5 file
            if self.traj_info.h5_path:
                h5_output = segment_dir / self.traj_info.h5_path.name
                print(f"  Segmenting H5: {h5_output.name}...", end=" ")
                if segment_h5_file(
                    self.traj_info.h5_path,
                    h5_output,
                    segment.start_timestep,
                    segment.end_timestep
                ):
                    print("OK")
                else:
                    print("FAILED")
                    success = False

            # Segment MP4 files
            mp4_output_dir = segment_dir / "recordings" / "MP4"
            for serial, mp4_path in self.traj_info.mp4_paths.items():
                mp4_output = mp4_output_dir / mp4_path.name
                print(f"  Segmenting MP4 ({serial}): {mp4_output.name}...", end=" ")
                if segment_mp4_file(
                    mp4_path,
                    mp4_output,
                    segment.start_time,
                    segment.end_time,
                    segment.start_frame,
                    segment.end_frame
                ):
                    print("OK")
                else:
                    print("FAILED")
                    success = False

            # Segment SVO files (if not skipped)
            if not self.skip_svo:
                svo_output_dir = segment_dir / "recordings" / "SVO"
                for serial, svo_path in self.traj_info.svo_paths.items():
                    if self.svo_to_mp4:
                        # Output as MP4
                        svo_output = svo_output_dir / (svo_path.stem + ".mp4")
                        print(f"  Segmenting SVO->MP4 ({serial}): {svo_output.name}...", end=" ")
                    else:
                        svo_output = svo_output_dir / svo_path.name
                        print(f"  Segmenting SVO ({serial}): {svo_output.name}...", end=" ")

                    if segment_svo_file(
                        svo_path,
                        svo_output,
                        segment.start_frame,
                        segment.end_frame,
                        convert_to_mp4=self.svo_to_mp4
                    ):
                        print("OK")
                    else:
                        print("SKIPPED")

            # Copy and update metadata files
            for metadata_path in self.traj_info.metadata_paths:
                metadata_output = segment_dir / metadata_path.name
                print(f"  Updating metadata: {metadata_output.name}...", end=" ")
                if update_metadata(metadata_path, metadata_output, segment):
                    print("OK")
                else:
                    print("FAILED")
                    success = False

            # Save segment info
            segment_info_path = segment_dir / "segment_info.json"
            with open(segment_info_path, 'w') as f:
                json.dump(asdict(segment), f, indent=2)
            print(f"  Saved segment info: {segment_info_path.name}")

            if success:
                print(f"  Segment '{segment.name}' completed successfully!")
            else:
                print(f"  Segment '{segment.name}' completed with some errors.")

        print("\n" + "=" * 60)
        print("SEGMENTATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Segments created: {[s.name for s in segments]}")

        return True


def parse_segments_json(segments_str: str) -> List[SegmentInfo]:
    """Parse segments from JSON string (for batch mode)."""
    try:
        segments_data = json.loads(segments_str)
        segments = []

        for seg in segments_data:
            # Basic validation
            if 'name' not in seg or 'start' not in seg or 'end' not in seg:
                raise ValueError("Each segment must have 'name', 'start', and 'end' fields")

            # For batch mode, we calculate frame/timestep indices later
            # when we have the trajectory info
            segments.append({
                'name': seg['name'],
                'start_time': float(seg['start']),
                'end_time': float(seg['end'])
            })

        return segments

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment robot trajectory data into labeled phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'trajectory_folder',
        type=str,
        nargs='?',
        default=None,
        help='Path to the trajectory folder (not required if --debug is used)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help=f'Debug mode: use hardcoded test path ({DEBUG_TRAJECTORY_PATH})'
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Use CLI mode instead of GUI (for headless environments)'
    )

    parser.add_argument(
        '--segments',
        type=str,
        default=None,
        help='JSON string with predefined segments for batch mode. '
             'Format: [{"name": "pick", "start": 5.0, "end": 25.0}, ...]'
    )

    parser.add_argument(
        '--camera',
        type=str,
        default='36087771',
        help='Camera serial number for video preview (default: 36087771)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: same as input trajectory folder)'
    )

    parser.add_argument(
        '--skip-svo',
        action='store_true',
        help='Skip SVO file segmentation'
    )

    parser.add_argument(
        '--svo-to-mp4',
        action='store_true',
        default=True,
        help='Convert SVO segments to MP4 format (default: True)'
    )

    parser.add_argument(
        '--no-auto-names',
        action='store_true',
        help='Disable auto-naming (prompts for segment names instead of using pick/place)'
    )

    args = parser.parse_args()

    # Handle debug mode
    if args.debug:
        print("\n" + "=" * 60)
        print("DEBUG MODE ENABLED")
        print("=" * 60)
        traj_folder = Path(DEBUG_TRAJECTORY_PATH)
        print(f"Using debug path: {traj_folder}")
    elif args.trajectory_folder:
        traj_folder = Path(args.trajectory_folder)
    else:
        print("Error: trajectory_folder is required (or use --debug)")
        parser.print_help()
        sys.exit(1)

    # Validate trajectory folder
    if not traj_folder.exists():
        print(f"Error: Trajectory folder not found: {traj_folder}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output) if args.output else traj_folder

    # Create segmenter
    segmenter = TrajectorySegmenter(
        traj_folder=traj_folder,
        output_dir=output_dir,
        camera_serial=args.camera,
        skip_svo=args.skip_svo,
        svo_to_mp4=args.svo_to_mp4,
        auto_names=not args.no_auto_names
    )

    # Get segments
    if args.segments:
        # Batch mode - parse predefined segments
        print("\nBatch mode: Using predefined segments")
        try:
            segments_data = parse_segments_json(args.segments)

            # Convert to SegmentInfo objects with calculated frame/timestep indices
            fps = segmenter.traj_info.video_fps
            total_frames = segmenter.traj_info.video_total_frames
            total_timesteps = segmenter.traj_info.h5_total_timesteps

            segments = []
            for seg_data in segments_data:
                start_frame = time_to_frame_index(seg_data['start_time'], fps)
                end_frame = time_to_frame_index(seg_data['end_time'], fps)
                start_timestep = frame_to_timestep_index(start_frame, total_frames, total_timesteps)
                end_timestep = frame_to_timestep_index(end_frame, total_frames, total_timesteps)

                segments.append(SegmentInfo(
                    name=seg_data['name'],
                    start_time=seg_data['start_time'],
                    end_time=seg_data['end_time'],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_timestep=start_timestep,
                    end_timestep=end_timestep
                ))

            print(f"Parsed {len(segments)} segments:")
            for seg in segments:
                print(f"  - {seg.name}: {seg.start_time:.2f}s - {seg.end_time:.2f}s")

        except ValueError as e:
            print(f"Error parsing segments: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        segments = segmenter.run_interactive(cli_mode=args.cli)

    if not segments:
        print("\nNo segments defined. Exiting.")
        sys.exit(0)

    # Apply segments
    segmenter.apply_segments(segments)


if __name__ == '__main__':
    main()
