#!/usr/bin/env python3
"""
Create Reference Images from Video or Live Camera

Interactive tool to create reference images for YOLOE visual prompt detection.
Extracts frames from a video OR captures live from ZED camera.

Usage:
    # From video:
    python create_ref_images.py --video /path/to/video.mp4 --object red_cup

    # From live ZED camera (3rd person view):
    python create_ref_images.py --live --object red_cup --camera_id 36087771

Controls:
    - Click and drag to draw bounding box
    - Press SPACE to confirm and move to next frame
    - Press 'r' to reset current box
    - Press 's' to skip frame (sample another random frame from video)
    - Press 'q' to quit
    - (Live mode) Press 'c' in terminal to capture next frame
"""

import argparse
import cv2
import json
import numpy as np
import random
from pathlib import Path


class BBoxDrawer:
    """Interactive bounding box drawer."""

    def __init__(self, window_name="Draw BBox"):
        self.window_name = window_name
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.bbox = None
        self.image = None
        self.display_image = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self._update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self._finalize_bbox()
            self._update_display()

    def _update_display(self):
        self.display_image = self.image.copy()
        if self.start_point and self.end_point:
            cv2.rectangle(self.display_image, self.start_point, self.end_point, (0, 255, 0), 2)
        cv2.imshow(self.window_name, self.display_image)

    def _finalize_bbox(self):
        if self.start_point and self.end_point:
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size
                self.bbox = [x1, y1, x2, y2]

    def reset(self):
        self.start_point = None
        self.end_point = None
        self.bbox = None
        self._update_display()

    def draw(self, image, title="", allow_skip=False):
        """Show image and let user draw bbox. Returns bbox [x1,y1,x2,y2], None (quit), or "SKIP"."""
        self.image = image.copy()
        self.display_image = image.copy()
        self.bbox = None
        self.start_point = None
        self.end_point = None

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Add title text
        if title:
            cv2.putText(self.display_image, title, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            help_text = "SPACE=confirm, R=reset, S=skip, Q=quit" if allow_skip else "SPACE=confirm, R=reset, Q=quit"
            cv2.putText(self.display_image, help_text,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(self.window_name, self.display_image)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space - confirm
                if self.bbox:
                    return self.bbox
                else:
                    print("No bbox drawn! Draw a box first.")

            elif key == ord('r'):  # Reset
                self.reset()
                if title:
                    cv2.putText(self.display_image, title, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(self.window_name, self.display_image)

            elif key == ord('s') and allow_skip:  # Skip - sample another frame
                print("Skipping this frame, will sample another...")
                return "SKIP"

            elif key == ord('q'):  # Quit
                return None


def capture_live_frame(camera_id: str) -> np.ndarray:
    """Capture a single frame from ZED camera using pyzed."""
    try:
        import pyzed.sl as sl
    except ImportError:
        raise ImportError("pyzed not installed. Install ZED SDK or use --video mode instead.")

    # Initialize camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(int(camera_id))
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera {camera_id}: {err}")

    # Capture frame
    image = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()
        # ZED returns BGRA, convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        zed.close()
        raise RuntimeError("Failed to grab frame from ZED camera")

    zed.close()
    return frame_bgr


def capture_live_frames_interactive(camera_id: str, object_name: str, num_frames: int = 5) -> list:
    """
    Interactively capture frames from live ZED camera.

    Flow:
    1. Capture frame, show bbox drawing window
    2. After bbox drawn, wait for user to press 'c' in terminal
    3. Repeat until num_frames captured

    Returns:
        List of (frame_idx, frame, bbox) tuples
    """
    drawer = BBoxDrawer("Live Capture - Create Reference Images")
    frames_data = []

    print(f"\n{'='*60}")
    print(f"LIVE CAPTURE MODE")
    print(f"Camera ID: {camera_id}")
    print(f"Object: {object_name}")
    print(f"Frames to capture: {num_frames}")
    print(f"{'='*60}")

    for i in range(num_frames):
        print(f"\n{'='*50}")
        print(f"Frame {i+1}/{num_frames}")
        print(f"{'='*50}")

        # Capture frame
        print("Capturing frame from ZED camera...")
        try:
            frame = capture_live_frame(camera_id)
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return frames_data

        print(f"Captured frame: {frame.shape}")
        print("\nDraw bounding box around the object")
        print("SPACE=confirm, R=reset, Q=quit")

        # Draw bbox
        bbox = drawer.draw(frame, f"Frame {i+1}/{num_frames} - {object_name}")

        if bbox is None:
            print("Cancelled by user")
            cv2.destroyAllWindows()
            return frames_data

        print(f"BBox: {bbox}")
        frames_data.append((i, frame, bbox))

        cv2.destroyAllWindows()

        # Wait for user to reposition object (except for last frame)
        if i < num_frames - 1:
            print(f"\n>>> Move/reposition the object, then press 'c' + Enter to capture next frame...")
            while True:
                user_input = input().strip().lower()
                if user_input == 'c':
                    break
                elif user_input == 'q':
                    print("Cancelled by user")
                    return frames_data
                else:
                    print("Press 'c' to continue or 'q' to quit")

    return frames_data


def extract_frames(video_path: str, num_frames: int = 3) -> list:
    """Extract evenly distributed frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate evenly distributed frame indices
    if num_frames == 1:
        frame_indices = [total_frames // 2]  # Just middle
    elif num_frames == 2:
        frame_indices = [0, total_frames - 1]  # First and last
    else:
        # Evenly distribute: first, ..., last
        frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
        else:
            print(f"Warning: Failed to read frame {idx}")

    cap.release()
    return frames


def get_next_index(object_name: str, output_dir: Path) -> int:
    """Find the next available index for object_name (to extend, not overwrite)."""
    import re
    max_index = -1

    # Check existing image files
    for img_path in output_dir.glob(f"{object_name}_*.jpg"):
        match = re.match(rf'^{re.escape(object_name)}_(\d+)\.jpg$', img_path.name)
        if match:
            idx = int(match.group(1))
            max_index = max(max_index, idx)

    # Also check bboxes.json
    bboxes_file = output_dir / "bboxes.json"
    if bboxes_file.exists():
        with open(bboxes_file, 'r') as f:
            bboxes = json.load(f)
        for key in bboxes.keys():
            match = re.match(rf'^{re.escape(object_name)}_(\d+)$', key)
            if match:
                idx = int(match.group(1))
                max_index = max(max_index, idx)

    return max_index + 1


def save_results(object_name: str, frames_data: list, output_dir: Path):
    """Save reference images and update bboxes.json (extends existing, doesn't overwrite)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find starting index to extend existing refs
    start_index = get_next_index(object_name, output_dir)
    if start_index > 0:
        print(f"Found existing refs for '{object_name}', extending from index {start_index}")

    # Load existing bboxes.json
    bboxes_file = output_dir / "bboxes.json"
    if bboxes_file.exists():
        with open(bboxes_file, 'r') as f:
            bboxes = json.load(f)
    else:
        bboxes = {"_comment": "Bounding boxes for reference images. Format: [x1, y1, x2, y2]"}

    # Save each frame and bbox
    saved_indices = []
    for i, (frame_idx, frame, bbox) in enumerate(frames_data):
        idx = start_index + i
        # Save image
        img_name = f"{object_name}_{idx}.jpg"
        img_path = output_dir / img_name
        cv2.imwrite(str(img_path), frame)
        print(f"Saved: {img_path}")

        # Add bbox entry
        bbox_key = f"{object_name}_{idx}"
        bboxes[bbox_key] = bbox
        saved_indices.append(idx)

    # Save updated bboxes.json
    with open(bboxes_file, 'w') as f:
        json.dump(bboxes, f, indent=4)
    print(f"Updated: {bboxes_file}")

    return saved_indices


def main():
    parser = argparse.ArgumentParser(description="Create reference images from video or live camera")
    parser.add_argument('--video', type=str, default=None, help='Path to video file')
    parser.add_argument('--live', action='store_true',
                        help='Use live ZED camera capture instead of video')
    parser.add_argument('--camera_id', type=str, default='36087771',
                        help='ZED camera serial number for live mode (default: 36087771)')
    parser.add_argument('--object', type=str, required=True, help='Object name (e.g., red_cup)')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames to capture (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: pipeline/data/yoloe_ref_images)')
    parser.add_argument('--third_view', action='store_true',
                        help='Save to 3rd view directory (yoloe_ref_images_3rd_view)')

    args = parser.parse_args()

    # Validate arguments
    if not args.live and not args.video:
        parser.error("Either --video or --live must be specified")

    # Setup output directory
    # Live mode defaults to 3rd view directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.third_view or args.live:
        output_dir = Path(__file__).parent.parent / "data" / "yoloe_ref_images" / "yoloe_ref_images_3rd_view"
    else:
        output_dir = Path(__file__).parent.parent / "data" / "yoloe_ref_images"

    print(f"Mode: {'LIVE CAMERA' if args.live else 'VIDEO'}")
    if args.live:
        print(f"Camera ID: {args.camera_id}")
    else:
        print(f"Video: {args.video}")
    print(f"Object: {args.object}")
    print(f"Num frames: {args.num_frames}")
    print(f"Output: {output_dir}")
    print()

    if args.live:
        # Live camera capture mode
        frames_data = capture_live_frames_interactive(
            camera_id=args.camera_id,
            object_name=args.object,
            num_frames=args.num_frames
        )

        if not frames_data:
            print("No frames captured")
            return

    else:
        # Video extraction mode
        print(f"Extracting {args.num_frames} evenly distributed frames...")
        frames = extract_frames(args.video, args.num_frames)
        print(f"Extracted {len(frames)} frames")
        print()

        # Get video info for random resampling on skip
        cap = cv2.VideoCapture(args.video)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Track which frame indices have been used
        used_indices = set(f[0] for f in frames)

        # Interactive bbox drawing
        drawer = BBoxDrawer("Create Reference Images")
        frames_data = []
        target_frames = args.num_frames
        frame_queue = list(frames)  # Frames to process
        processed_count = 0

        while processed_count < target_frames and frame_queue:
            frame_idx, frame = frame_queue.pop(0)

            print(f"\n{'='*50}")
            print(f"Frame {processed_count+1}/{target_frames} (video index {frame_idx})")
            print("Draw bounding box around the object")
            print("SPACE=confirm, R=reset, S=skip (sample new), Q=quit")

            bbox = drawer.draw(frame, f"Frame {processed_count+1}/{target_frames} - {args.object}", allow_skip=True)

            if bbox is None:
                print("Cancelled by user")
                cv2.destroyAllWindows()
                return

            if bbox == "SKIP":
                # Sample a new random frame from unused frames
                available_indices = [i for i in range(total_video_frames) if i not in used_indices]
                if available_indices:
                    new_idx = random.choice(available_indices)
                    used_indices.add(new_idx)

                    # Read the new frame
                    cap = cv2.VideoCapture(args.video)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                    ret, new_frame = cap.read()
                    cap.release()

                    if ret:
                        frame_queue.append((new_idx, new_frame))
                        print(f"Sampled new frame at index {new_idx}")
                    else:
                        print(f"Warning: Failed to read frame {new_idx}, trying another...")
                        frame_queue.append((frame_idx, frame))  # Re-add current frame to try again
                else:
                    print("No more frames available to sample!")
                continue

            print(f"BBox: {bbox}")
            frames_data.append((frame_idx, frame, bbox))
            processed_count += 1

        cv2.destroyAllWindows()

    # Save results
    print(f"\n{'='*50}")
    print("Saving reference images...")
    saved_indices = save_results(args.object, frames_data, output_dir)

    print(f"\n{'='*50}")
    print(f"Done! Created {len(frames_data)} reference images:")
    for idx in saved_indices:
        print(f"  - {args.object}_{idx}.jpg with bbox")
    print(f"\nYou can now use these with yoloe_visual_server.py")


if __name__ == '__main__':
    main()
