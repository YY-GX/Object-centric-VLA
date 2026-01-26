#!/usr/bin/env python3
"""
YOLOE Visual Prompt Detection Test Script

Tests cross-image detection using visual prompts (reference images).

Reference images should be saved in pipeline/data/yoloe_ref_images/:
    - red_cup.jpg (or .png)
    - plate.jpg
    - etc.

Optionally, provide bounding boxes in a JSON file:
    pipeline/data/yoloe_ref_images/bboxes.json
    {
        "red_cup": [x1, y1, x2, y2],  // or null for full image
        "plate": [100, 50, 400, 350]
    }

Usage:
    # Register all reference images and test on a target image
    python test_yoloe_visual_detection.py --target /path/to/image.jpg

    # Register specific objects only
    python test_yoloe_visual_detection.py --target /path/to/image.jpg --objects red_cup plate

    # Test on video
    python test_yoloe_visual_detection.py --video /path/to/video.mp4 --objects red_cup

Requirements:
    - YOLOE visual server running: python yoloe_visual_server.py --port 5559
"""

import argparse
import json
import zmq
import pickle
import numpy as np
import cv2
from pathlib import Path


class YOLOEVisualClient:
    """ZMQ client for YOLOE visual prompt server."""

    def __init__(self, host="localhost", port=5559, timeout=60000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to YOLOE Visual Server at {host}:{port}")

    def register(self, object_name: str, image: np.ndarray, bbox: list = None) -> dict:
        """Register reference image for an object."""
        request = {
            'api': 'register',
            'object_name': object_name,
            'image': image,
            'bbox': bbox
        }
        self.socket.send(pickle.dumps(request))
        return pickle.loads(self.socket.recv())

    def detect(self, object_name: str, image: np.ndarray, conf: float = 0.1) -> dict:
        """Detect object in target image using stored visual embedding."""
        request = {
            'api': 'detect',
            'object_name': object_name,
            'image': image,
            'conf': conf
        }
        self.socket.send(pickle.dumps(request))
        return pickle.loads(self.socket.recv())

    def list_objects(self) -> dict:
        """List all registered objects."""
        request = {'api': 'list'}
        self.socket.send(pickle.dumps(request))
        return pickle.loads(self.socket.recv())

    def clear(self, object_name: str = '__all__') -> dict:
        """Clear registered object(s)."""
        request = {'api': 'clear', 'object_name': object_name}
        self.socket.send(pickle.dumps(request))
        return pickle.loads(self.socket.recv())

    def close(self):
        self.socket.close()
        self.context.term()


def get_base_name(object_name: str) -> str:
    """Extract base name from object_name (e.g., 'red_cup_0' -> 'red_cup')."""
    import re
    match = re.match(r'^(.+)_(\d+)$', object_name)
    if match:
        return match.group(1)
    return object_name


def load_reference_images(ref_dir: Path, objects: list = None) -> dict:
    """
    Load reference images from directory.

    Supports variants like red_cup_0.jpg, red_cup_1.jpg, red_cup_2.jpg
    When objects=['red_cup'], loads all red_cup_* variants.

    Returns: {object_name: {'image': np.ndarray, 'bbox': [x1,y1,x2,y2] or None}}
    """
    refs = {}

    # Load bboxes config if exists
    bboxes_file = ref_dir / "bboxes.json"
    bboxes = {}
    if bboxes_file.exists():
        with open(bboxes_file, 'r') as f:
            bboxes = json.load(f)
        print(f"Loaded bboxes config: {bboxes_file}")

    # Find all image files
    for img_path in ref_dir.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        if img_path.stem == 'bboxes':
            continue

        object_name = img_path.stem
        base_name = get_base_name(object_name)

        # Skip if not in requested objects list (check both full name and base name)
        if objects:
            if object_name not in objects and base_name not in objects:
                continue

        # Load image (convert BGR to RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get bbox if specified
        bbox = bboxes.get(object_name)

        refs[object_name] = {
            'image': img_rgb,
            'bbox': bbox,
            'path': str(img_path),
            'base_name': base_name
        }
        print(f"Loaded reference: {object_name} (base: {base_name}) from {img_path.name}, bbox={bbox}")

    return refs


def draw_detections(image: np.ndarray, detections: dict, color=(0, 255, 0)) -> np.ndarray:
    """Draw detection results on image."""
    overlay = image.copy()

    masks = detections.get('masks', [])
    bboxes = detections.get('bboxes', [])
    confidences = detections.get('confidences', [])
    object_name = detections.get('object_name', 'object')

    for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
        # Draw bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{object_name}: {conf:.2f}"
        cv2.putText(overlay, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw mask if available
        if i < len(masks):
            mask = np.array(masks[i], dtype=np.uint8)
            colored_mask = np.zeros_like(overlay)
            colored_mask[:, :] = color
            mask_region = mask > 0
            overlay[mask_region] = cv2.addWeighted(
                overlay, 0.7, colored_mask, 0.3, 0)[mask_region]

    return overlay


def test_on_image(client: YOLOEVisualClient, image_path: str, objects: list,
                  conf: float, save_dir: Path) -> dict:
    """Test detection on a single image."""
    print(f"\nTesting on image: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image")
        return {}

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img.copy()

    # Colors for different objects
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]

    results = {}
    for i, obj_name in enumerate(objects):
        print(f"  Detecting: {obj_name}")
        response = client.detect(obj_name, img_rgb, conf)

        if 'error' in response:
            print(f"    Error: {response['error']}")
            continue

        num_detections = len(response.get('masks', []))
        print(f"    Found: {num_detections} detections")

        if num_detections > 0:
            color = colors[i % len(colors)]
            overlay = draw_detections(overlay, response, color)
            results[obj_name] = response

    # Save result
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{Path(image_path).stem}_visual_detection.jpg"
        cv2.imwrite(str(save_path), overlay)
        print(f"Saved: {save_path}")

    return results


def test_on_video(client: YOLOEVisualClient, video_path: str, objects: list,
                  conf: float, save_dir: Path, sample_rate: int = 10):
    """Test detection on video with sampling."""
    from datetime import datetime

    print(f"\nProcessing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output with timestamp + object names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    objects_str = "_".join(objects)
    output_folder = save_dir / f"{timestamp}_{objects_str}"
    output_folder.mkdir(parents=True, exist_ok=True)
    frames_dir = output_folder / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / "output.mp4"
    output_fps = max(1, fps // sample_rate)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"  Sample rate: every {sample_rate} frames")
    print(f"  Objects: {objects}")

    # Colors for different objects
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
    object_colors = {name: colors[i % len(colors)] for i, name in enumerate(objects)}

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame.copy()

            for obj_name in objects:
                try:
                    response = client.detect(obj_name, frame_rgb, conf)
                    if 'error' not in response and len(response.get('masks', [])) > 0:
                        overlay = draw_detections(overlay, response, object_colors[obj_name])
                except Exception as e:
                    pass

            out.write(overlay)
            frame_path = frames_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), overlay)
            processed += 1

            if processed % 10 == 0:
                print(f"  Processed {processed} frames (frame {frame_idx}/{total_frames})...")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Processed {processed} frames")
    print(f"  Video: {output_path}")
    print(f"  Frames: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test YOLOE Visual Prompt Detection")
    parser.add_argument('--target', type=str, help='Target image to detect in')
    parser.add_argument('--video', type=str, help='Target video to detect in')
    parser.add_argument('--objects', nargs='+', help='Specific objects to test')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5559, help='Server port')
    parser.add_argument('--ref_dir', type=str, default=None,
                        help='Reference images directory (default: pipeline/data/yoloe_ref_images)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: pipeline/data/yoloe_test/visual)')
    parser.add_argument('--sample_rate', type=int, default=10, help='Video sample rate')
    parser.add_argument('--skip_register', action='store_true',
                        help='Skip registration (use already registered objects)')

    args = parser.parse_args()

    # Setup paths
    ref_dir = Path(args.ref_dir) if args.ref_dir else \
              Path(__file__).parent.parent / "data" / "yoloe_ref_images"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.video:
        output_dir = Path(__file__).parent.parent / "data" / "yoloe_test" / "visual_videos"
    else:
        output_dir = Path(__file__).parent.parent / "data" / "yoloe_test" / "visual"

    # Connect to server
    try:
        client = YOLOEVisualClient(host=args.host, port=args.port)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure YOLOE visual server is running:")
        print("  cd /home/yygx/PANDA/externals/yoloe")
        print("  conda activate yoloe")
        print("  python tests_yy/scripts/yoloe_visual_server.py --port 5559")
        return

    # Register reference images (unless skipped)
    if not args.skip_register:
        print(f"\n{'='*60}")
        print("Registering Reference Images")
        print(f"{'='*60}")

        if not ref_dir.exists():
            print(f"Error: Reference directory not found: {ref_dir}")
            print("Please create the directory and add reference images:")
            print(f"  mkdir -p {ref_dir}")
            print(f"  # Add images like: red_cup.jpg, plate.png, etc.")
            return

        refs = load_reference_images(ref_dir, args.objects)

        if not refs:
            print("No reference images found!")
            return

        for obj_name, ref_data in refs.items():
            print(f"\nRegistering: {obj_name}")
            response = client.register(obj_name, ref_data['image'], ref_data['bbox'])
            if response.get('success'):
                print(f"  Success! VPE shape: {response.get('vpe_shape')}")
            else:
                print(f"  Failed: {response.get('error')}")

    # List registered objects
    registered = client.list_objects()
    print(f"\nServer response: {registered}")

    # Use base_objects for detection (server will try all variants)
    base_objects = registered.get('base_objects', [])

    if args.objects:
        # Filter to requested objects
        objects_to_test = [o for o in args.objects if o in base_objects]
    else:
        objects_to_test = base_objects

    print(f"\nBase objects to test: {objects_to_test}")
    print(f"Grouped variants: {registered.get('grouped', {})}")

    if not objects_to_test:
        print("No objects to test!")
        return

    # Run detection
    print(f"\n{'='*60}")
    print("Running Detection")
    print(f"{'='*60}")

    if args.target:
        test_on_image(client, args.target, objects_to_test, args.conf, output_dir)
    elif args.video:
        test_on_video(client, args.video, objects_to_test, args.conf, output_dir, args.sample_rate)
    else:
        print("Error: Must specify --target (image) or --video")
        print("Example:")
        print("  python test_yoloe_visual_detection.py --target /path/to/image.jpg")
        print("  python test_yoloe_visual_detection.py --video /path/to/video.mp4")

    client.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
