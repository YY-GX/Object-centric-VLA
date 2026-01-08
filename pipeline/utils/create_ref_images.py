#!/usr/bin/env python3
"""
Create Reference Images from Video

Interactive tool to create reference images for YOLOE visual prompt detection.
Extracts first, middle, last frames from a video and lets you draw bounding boxes.

Usage:
    python create_ref_images.py --video /path/to/video.mp4 --object red_cup

Controls:
    - Click and drag to draw bounding box
    - Press SPACE to confirm and move to next frame
    - Press 'r' to reset current box
    - Press 'q' to quit
"""

import argparse
import cv2
import json
import numpy as np
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

    def draw(self, image, title=""):
        """Show image and let user draw bbox. Returns bbox [x1,y1,x2,y2] or None."""
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
            cv2.putText(self.display_image, "Draw bbox, SPACE=confirm, R=reset, Q=quit",
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

            elif key == ord('q'):  # Quit
                return None


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
    parser = argparse.ArgumentParser(description="Create reference images from video")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--object', type=str, required=True, help='Object name (e.g., red_cup)')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames to extract (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: pipeline/data/yoloe_ref_images)')
    parser.add_argument('--third_view', action='store_true',
                        help='Save to 3rd view directory (yoloe_ref_images_3rd_view)')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.third_view:
        output_dir = Path(__file__).parent.parent / "data" / "yoloe_ref_images" / "yoloe_ref_images_3rd_view"
    else:
        output_dir = Path(__file__).parent.parent / "data" / "yoloe_ref_images"

    print(f"Video: {args.video}")
    print(f"Object: {args.object}")
    print(f"Num frames: {args.num_frames}")
    print(f"Third view mode: {args.third_view}")
    print(f"Output: {output_dir}")
    print()

    # Extract frames
    print(f"Extracting {args.num_frames} evenly distributed frames...")
    frames = extract_frames(args.video, args.num_frames)
    print(f"Extracted {len(frames)} frames")
    print()

    # Interactive bbox drawing
    drawer = BBoxDrawer("Create Reference Images")
    frames_data = []
    num_frames = len(frames)

    for i, (frame_idx, frame) in enumerate(frames):
        print(f"\n{'='*50}")
        print(f"Frame {i+1}/{num_frames} (index {frame_idx})")
        print("Draw bounding box around the object")
        print("SPACE=confirm, R=reset, Q=quit")

        bbox = drawer.draw(frame, f"Frame {i+1}/{num_frames} - {args.object}")

        if bbox is None:
            print("Cancelled by user")
            cv2.destroyAllWindows()
            return

        print(f"BBox: {bbox}")
        frames_data.append((frame_idx, frame, bbox))

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
