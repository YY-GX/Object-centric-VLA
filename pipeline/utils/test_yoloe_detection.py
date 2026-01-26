#!/usr/bin/env python3
"""
YOLOE Detection Test Script

Tests whether all objects in the config can be detected by YOLOE server.

Usage:
    # Test with an image file
    python test_yoloe_detection.py --image /path/to/image.jpg

    # Test with video file (outputs video with detection overlay)
    python test_yoloe_detection.py --video /path/to/video.mp4 --objects red_cup plate

    # Test with camera capture (requires robot env)
    python test_yoloe_detection.py --camera

    # Test specific objects only
    python test_yoloe_detection.py --image /path/to/image.jpg --objects red_cup plate

Requirements:
    - YOLOE server running: python yoloe_server.py (port 5557)
"""

import argparse
import json
import zmq
import pickle
import numpy as np
import cv2
from pathlib import Path


class YOLOEClient:
    """Simple ZMQ client for YOLOE server."""

    def __init__(self, host="localhost", port=5557, timeout=30000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to YOLOE server at {host}:{port}")

    def detect(self, image: np.ndarray, text_prompt: str, conf: float = 0.1) -> dict:
        """
        Detect objects using text prompt.

        Args:
            image: RGB image (H, W, 3) uint8
            text_prompt: Text description of object to detect
            conf: Confidence threshold

        Returns:
            Response dict with masks, bboxes, confidences
        """
        request = {
            'image': image,
            'mode': 'text',
            'text_prompt': text_prompt,
            'conf': conf
        }

        print(f"    [DEBUG] Sending request with text_prompt='{text_prompt}'")
        self.socket.send(pickle.dumps(request))
        response = pickle.loads(self.socket.recv())
        print(f"    [DEBUG] Received response with {len(response.get('masks', []))} detections")

        return response

    def close(self):
        self.socket.close()
        self.context.term()


def load_config():
    """Load the real robot config."""
    config_path = Path(__file__).parent.parent / "config" / "real_robot_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def load_image(image_path: str) -> np.ndarray:
    """Load image from file and convert to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    # Convert BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process_video(client: YOLOEClient, video_path: str, output_dir: Path,
                  objects: dict, conf: float = 0.1, sample_rate: int = 10) -> None:
    """
    Process video and output with detection overlay.

    Args:
        client: YOLOE client
        video_path: Path to input video
        output_dir: Directory to save output video
        objects: Dict of {object_name: prompt}
        conf: Confidence threshold
        sample_rate: Process every N frames (default 10)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(video_path).stem
    frames_dir = output_dir / f"{video_name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Setup output video (at reduced fps due to sampling)
    output_path = output_dir / f"{video_name}_yoloe.mp4"
    output_fps = max(1, fps // sample_rate)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"  Sample rate: every {sample_rate} frames ({total_frames // sample_rate} frames to process)")
    print(f"  Output video: {output_path}")
    print(f"  Output frames: {frames_dir}")
    print(f"  Objects to detect: {list(objects.keys())}")

    # Color map for different objects
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
    object_colors = {name: colors[i % len(colors)] for i, name in enumerate(objects.keys())}

    frame_idx = 0
    processed_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every sample_rate frames
        if frame_idx % sample_rate == 0:
            # Convert BGR to RGB for detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame.copy()

            # Detect each object
            for obj_name, prompt in objects.items():
                try:
                    response = client.detect(frame_rgb, prompt, conf)
                    if 'error' not in response and len(response.get('masks', [])) > 0:
                        color = object_colors[obj_name]
                        bboxes = response.get('bboxes', [])
                        confidences = response.get('confidences', [])
                        masks = response.get('masks', [])

                        for bbox, conf_val, mask_list in zip(bboxes, confidences, masks):
                            # Draw bbox
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                            label = f"{obj_name}: {conf_val:.2f}"
                            cv2.putText(overlay, label, (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            # Draw mask overlay
                            mask = np.array(mask_list, dtype=np.uint8)
                            colored_mask = np.zeros_like(overlay)
                            colored_mask[:, :] = color
                            mask_region = mask > 0
                            overlay[mask_region] = cv2.addWeighted(
                                overlay, 0.7, colored_mask, 0.3, 0)[mask_region]
                except Exception as e:
                    pass  # Skip detection errors for this frame

            # Save frame to video and as image
            out.write(overlay)
            frame_path = frames_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), overlay)
            processed_count += 1

            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} frames (frame {frame_idx}/{total_frames})...")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Processed {processed_count} frames")
    print(f"  Video: {output_path}")
    print(f"  Frames: {frames_dir}")


def capture_from_camera() -> np.ndarray:
    """Capture image from robot camera (requires droid env)."""
    import sys
    sys.path.insert(0, "/home/yygx/PANDA/droid")
    from droid.robot_env import RobotEnv

    print("Initializing robot environment...")
    env = RobotEnv()

    print("Capturing image from camera...")
    obs = env.get_observation()

    # Get left camera image (varied camera)
    from droid.misc.parameters import varied_camera_1_id
    camera_key = f"{varied_camera_1_id}_left"

    if "image" in obs and camera_key in obs["image"]:
        img = obs["image"][camera_key]
        # Convert BGRA to RGB if needed
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif len(img.shape) == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError(f"Camera image not found for {camera_key}")


def test_detection(client: YOLOEClient, image: np.ndarray, object_name: str,
                   prompt: str, conf: float = 0.1, save_dir: Path = None) -> dict:
    """
    Test detection for a single object.

    Returns:
        Dict with detection results
    """
    print(f"\n  Testing: {object_name}")
    print(f"    Prompt: '{prompt}'")

    try:
        response = client.detect(image, prompt, conf)

        if 'error' in response:
            print(f"    Error: {response['error']}")
            return {'detected': False, 'error': response['error']}

        num_detections = len(response.get('masks', []))
        confidences = response.get('confidences', [])
        bboxes = response.get('bboxes', [])

        if num_detections > 0:
            max_conf = max(confidences)
            print(f"    Detected: YES ({num_detections} instances)")
            print(f"    Max confidence: {max_conf:.3f}")
            print(f"    Bboxes: {bboxes}")

            # Save visualization if requested
            if save_dir:
                # Save detection with bboxes
                save_path = save_dir / f"{object_name}_detection.jpg"
                vis_img = image.copy()
                for bbox, conf_val in zip(bboxes, confidences):
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, f"{conf_val:.2f}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                print(f"    Saved bbox: {save_path}")

                # Save mask images
                masks = response.get('masks', [])
                for i, mask_list in enumerate(masks):
                    mask = np.array(mask_list, dtype=np.uint8)

                    # Save binary mask
                    mask_path = save_dir / f"{object_name}_mask_{i}.png"
                    cv2.imwrite(str(mask_path), mask * 255)

                    # Save mask overlay on original image
                    overlay_path = save_dir / f"{object_name}_overlay_{i}.jpg"
                    overlay = image.copy()
                    # Create colored mask overlay (green)
                    colored_mask = np.zeros_like(overlay)
                    colored_mask[:, :, 1] = mask * 255  # Green channel
                    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                    # Draw contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                    print(f"    Saved mask {i}: {mask_path}")
                    print(f"    Saved overlay {i}: {overlay_path}")

            return {
                'detected': True,
                'count': num_detections,
                'max_confidence': max_conf,
                'confidences': confidences,
                'bboxes': bboxes
            }
        else:
            print(f"    Detected: NO")
            return {'detected': False, 'count': 0}

    except zmq.error.Again:
        print(f"    Error: Timeout waiting for YOLOE server")
        return {'detected': False, 'error': 'timeout'}
    except Exception as e:
        print(f"    Error: {e}")
        return {'detected': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Test YOLOE detection for all objects")
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to input video (mp4)')
    parser.add_argument('--camera', action='store_true', help='Capture from robot camera')
    parser.add_argument('--objects', nargs='+', help='Specific objects to test (default: all)')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--host', type=str, default='localhost', help='YOLOE server host')
    parser.add_argument('--port', type=int, default=5557, help='YOLOE server port')
    parser.add_argument('--save', action='store_true', help='Save detection visualizations')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for video')
    parser.add_argument('--sample_rate', type=int, default=10, help='Process every N frames (default: 10)')

    args = parser.parse_args()

    # Load config
    config = load_config()
    yoloe_prompts = config.get('yoloe_prompts', {})

    # Filter objects if specified
    if args.objects:
        objects_to_test = {k: v for k, v in yoloe_prompts.items()
                         if k in args.objects and not k.startswith('_')}
    else:
        objects_to_test = {k: v for k, v in yoloe_prompts.items()
                         if not k.startswith('_')}

    # Handle video mode
    if args.video:
        output_dir = Path(args.output_dir) if args.output_dir else \
                     Path(__file__).parent.parent / "data" / "yoloe_test" / "videos"
        try:
            client = YOLOEClient(host=args.host, port=args.port)
            process_video(client, args.video, output_dir, objects_to_test, args.conf, args.sample_rate)
            client.close()
        except Exception as e:
            print(f"Error: {e}")
        return

    # Load image
    if args.image:
        print(f"Loading image: {args.image}")
        image = load_image(args.image)
    elif args.camera:
        image = capture_from_camera()
    else:
        print("Error: Must specify --image, --video, or --camera")
        print("Examples:")
        print("  python test_yoloe_detection.py --image /path/to/image.jpg")
        print("  python test_yoloe_detection.py --video /path/to/video.mp4 --objects red_cup plate")
        return

    print(f"Image shape: {image.shape}")
    print(f"\nObjects to test: {list(objects_to_test.keys())}")

    # Setup save directory
    save_dir = None
    if args.save:
        save_dir = Path(__file__).parent.parent / "data" / "yoloe_test"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {save_dir}")

    # Connect to YOLOE server
    try:
        client = YOLOEClient(host=args.host, port=args.port)
    except Exception as e:
        print(f"Error connecting to YOLOE server: {e}")
        print("Make sure YOLOE server is running:")
        print("  cd /home/yygx/PANDA/externals/yoloe")
        print("  conda activate yoloe")
        print("  python tests_yy/scripts/yoloe_server.py")
        return

    # Test each object
    print("\n" + "="*60)
    print("YOLOE Detection Test Results")
    print("="*60)

    results = {}
    detected_count = 0

    for object_name, prompt in objects_to_test.items():
        result = test_detection(client, image, object_name, prompt,
                               conf=args.conf, save_dir=save_dir)
        results[object_name] = result
        if result.get('detected', False):
            detected_count += 1

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total objects tested: {len(objects_to_test)}")
    print(f"Detected: {detected_count}")
    print(f"Not detected: {len(objects_to_test) - detected_count}")

    print("\nDetailed results:")
    for object_name, result in results.items():
        status = "OK" if result.get('detected') else "FAIL"
        conf_str = f" (conf: {result.get('max_confidence', 0):.3f})" if result.get('detected') else ""
        print(f"  [{status}] {object_name}{conf_str}")

    # List failed objects with suggestions
    failed = [name for name, r in results.items() if not r.get('detected')]
    if failed:
        print(f"\nObjects not detected: {failed}")
        print("Suggestions:")
        print("  1. Make sure the objects are visible in the test image")
        print("  2. Try adjusting the YOLOE prompts in real_robot_config.json")
        print("  3. Lower the confidence threshold with --conf 0.05")

    client.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
