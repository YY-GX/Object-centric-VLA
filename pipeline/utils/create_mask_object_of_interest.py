#!/usr/bin/env python3
"""
Create Mask of Objects of Interest for Trajectory Data

Processes H5 trajectory files to generate segmentation masks for objects of interest.
Uses YOLOE visual prompt server to detect objects in wrist camera images.

For each timestep:
1. Extract frame from wrist camera MP4 (16478870)
2. Detect specified objects using YOLOE visual prompts
3. Union all object masks into a single mask
4. Save masks to H5 file

Usage:
    # Process specific trajectory directory
    python create_mask_object_of_interest.py --trajectory_dir /path/to/trajectory/

    # Process all trajectories in a data directory
    python create_mask_object_of_interest.py --data_dir /path/to/droid/data/success/

    # Process with specific objects (override config)
    python create_mask_object_of_interest.py --trajectory_dir /path/to/traj --objects red_cup robot_gripper

Requirements:
    - YOLOE visual server running: python yoloe_visual_server.py --port 5559
    - Reference images registered for all objects
"""

import argparse
import json
import fnmatch
import h5py
import cv2
import zmq
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


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

    def close(self):
        self.socket.close()
        self.context.term()


def load_task_config(config_path: Path) -> dict:
    """Load task-to-objects mapping config."""
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Remove comment keys
    return {k: v for k, v in config.items() if not k.startswith('_')}


def get_objects_for_task(task_name: str, config: dict) -> list:
    """Get objects of interest for a given task name."""
    # Exact match first
    if task_name in config:
        return config[task_name]

    # Pattern matching (wildcards)
    for pattern, objects in config.items():
        if fnmatch.fnmatch(task_name, pattern):
            return objects

    # Default fallback
    if '_default' in config:
        return config['_default']

    return []


def extract_task_name(trajectory_dir: Path) -> str:
    """
    Extract task name from trajectory directory path.

    Handles patterns like:
    - .../success/2026-01-07/pick_the_red_cup/Wed_Jan__7_05:00:10_2026/
    - .../success/2026-01-07/Tue_Jan__6_23:00:00_2026/  (no task folder)
    """
    parent = trajectory_dir.parent

    # Check if parent looks like a date (YYYY-MM-DD) or a task name
    if parent.name and not parent.name[0].isdigit():
        # Parent is likely task name
        return parent.name

    return ""


def get_base_name(object_name: str) -> str:
    """Extract base name from object_name (e.g., 'red_cup_0' -> 'red_cup')."""
    import re
    match = re.match(r'^(.+)_(\d+)$', object_name)
    if match:
        return match.group(1)
    return object_name


def load_reference_images(ref_dir: Path, objects: list = None) -> dict:
    """Load reference images from directory."""
    refs = {}

    bboxes_file = ref_dir / "bboxes.json"
    bboxes = {}
    if bboxes_file.exists():
        with open(bboxes_file, 'r') as f:
            bboxes = json.load(f)

    for img_path in ref_dir.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        if img_path.stem.startswith('_'):
            continue

        object_name = img_path.stem
        base_name = get_base_name(object_name)

        if objects:
            if object_name not in objects and base_name not in objects:
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox = bboxes.get(object_name)

        refs[object_name] = {
            'image': img_rgb,
            'bbox': bbox,
            'base_name': base_name
        }

    return refs


def union_masks(masks: list, shape: tuple, fallback_full: bool = True) -> np.ndarray:
    """
    Union multiple masks into one.

    Args:
        masks: List of masks to union
        shape: Expected output shape (H, W)
        fallback_full: If True, return full mask (all ones) when no masks detected.
                       If False, return empty mask (all zeros).
    """
    if not masks:
        if fallback_full:
            # No detection - use full image as mask (conservative approach)
            return np.ones(shape, dtype=np.uint8)
        else:
            return np.zeros(shape, dtype=np.uint8)

    result = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        mask_arr = np.array(mask, dtype=np.uint8)
        if mask_arr.shape == shape:
            result = np.maximum(result, mask_arr)

    return result


def process_trajectory(
    trajectory_dir: Path,
    client: YOLOEVisualClient,
    objects: list,
    wrist_camera: str = "16478870",
    conf: float = 0.1,
    sample_rate: int = 1,
    overwrite: bool = False,
    visualize: bool = False,
    fallback_full: bool = True,
    save_debug_video: Path = None
) -> dict:
    """
    Process a single trajectory directory.

    Args:
        save_debug_video: If provided, save overlay video to this path

    Returns:
        dict with processing stats
    """
    h5_path = trajectory_dir / "trajectory.h5"
    mp4_path = trajectory_dir / "recordings" / "MP4" / f"{wrist_camera}.mp4"

    if not h5_path.exists():
        return {'error': f"H5 file not found: {h5_path}"}

    if not mp4_path.exists():
        return {'error': f"MP4 file not found: {mp4_path}"}

    # Open video
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return {'error': f"Failed to open video: {mp4_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check existing masks in H5
    with h5py.File(h5_path, 'r') as f:
        if 'observation/mask_objects_of_interest' in f and not overwrite:
            cap.release()
            return {'skipped': True, 'reason': 'masks already exist'}

    print(f"Processing: {trajectory_dir.name}")
    print(f"  Video: {total_frames} frames, {width}x{height}")
    print(f"  Objects: {objects}")

    # Setup debug video writer if requested
    debug_video_writer = None
    if save_debug_video:
        save_debug_video.parent.mkdir(parents=True, exist_ok=True)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        debug_video_writer = cv2.VideoWriter(str(save_debug_video), fourcc, fps, (width, height))
        print(f"  Saving debug video to: {save_debug_video}")

    # Process frames
    masks_all = []
    overlay_frames = []  # Store for debug video
    detection_counts = {obj: 0 for obj in objects}

    frame_indices = list(range(0, total_frames, sample_rate))

    for frame_idx in tqdm(frame_indices, desc="  Frames", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            masks_all.append(np.zeros((height, width), dtype=np.uint8))
            if debug_video_writer:
                # Write black frame
                debug_video_writer.write(np.zeros((height, width, 3), dtype=np.uint8))
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect each object
        frame_masks = []
        for obj_name in objects:
            try:
                response = client.detect(obj_name, frame_rgb, conf)
                if 'error' not in response and len(response.get('masks', [])) > 0:
                    # Take first (highest confidence) mask
                    mask = response['masks'][0]
                    frame_masks.append(mask)
                    detection_counts[obj_name] += 1
            except Exception as e:
                pass

        # Union masks (fallback to full mask if no detections)
        union = union_masks(frame_masks, (height, width), fallback_full=fallback_full)
        masks_all.append(union)

        # Create overlay for debug video
        if debug_video_writer:
            overlay = frame.copy()
            # Green overlay where mask is 1
            mask_region = union > 0
            overlay[mask_region] = (overlay[mask_region] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            debug_video_writer.write(overlay)

        # Visualization (optional)
        if visualize and frame_idx % 30 == 0:
            overlay = frame.copy()
            overlay[union > 0] = [0, 255, 0]
            vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.imshow("Mask", vis)
            cv2.waitKey(1)

    cap.release()
    if debug_video_writer:
        debug_video_writer.release()
        print(f"  Debug video saved: {save_debug_video}")
    if visualize:
        cv2.destroyAllWindows()

    # Convert to numpy array
    masks_array = np.array(masks_all, dtype=np.uint8)

    # Ensure masks match H5 timesteps (video frames may differ from H5 timesteps)
    with h5py.File(h5_path, 'r') as f:
        h5_timesteps = f['observation/robot_state/cartesian_position'].shape[0]

    if len(masks_all) != h5_timesteps:
        # Interpolate/resample masks to match H5 timesteps
        print(f"  Resampling {len(masks_all)} masks to {h5_timesteps} H5 timesteps")
        masks_resampled = []
        for i in range(h5_timesteps):
            # Map H5 timestep to mask index
            src_idx = min(int(i * len(masks_all) / h5_timesteps), len(masks_all) - 1)
            masks_resampled.append(masks_all[src_idx])
        masks_array = np.array(masks_resampled, dtype=np.uint8)

    # Save to H5 using 'a' mode (append) - this ONLY ADDS new data, does NOT modify existing data
    # This matches DROID's h5py usage pattern
    with h5py.File(h5_path, 'a') as f:
        dataset_name = 'observation/mask_objects_of_interest'

        if dataset_name in f:
            if overwrite:
                del f[dataset_name]
            else:
                return {'skipped': True, 'reason': 'masks already exist'}

        # Create dataset (no compression, matching DROID's format)
        f.create_dataset(
            dataset_name,
            data=masks_array,
            dtype=np.uint8
        )

        # Store metadata as attributes (same pattern as DROID)
        f[dataset_name].attrs['objects'] = json.dumps(objects)
        f[dataset_name].attrs['wrist_camera'] = wrist_camera
        f[dataset_name].attrs['conf_threshold'] = conf
        f[dataset_name].attrs['created_at'] = datetime.now().isoformat()

    print(f"  Saved masks: {masks_array.shape}")
    print(f"  Detection counts: {detection_counts}")

    return {
        'success': True,
        'num_frames': len(masks_all),
        'mask_shape': masks_array.shape,
        'detection_counts': detection_counts
    }


def find_trajectory_dirs(data_dir: Path) -> list:
    """Find all trajectory directories under data_dir."""
    trajectory_dirs = []

    for h5_path in data_dir.rglob("trajectory.h5"):
        trajectory_dirs.append(h5_path.parent)

    return sorted(trajectory_dirs)


def main():
    parser = argparse.ArgumentParser(description="Create masks for objects of interest")
    parser.add_argument('--trajectory_dir', type=str, help='Single trajectory directory to process')
    parser.add_argument('--data_dir', type=str, help='Data directory to process all trajectories')
    parser.add_argument('--objects', nargs='+', help='Objects to detect (overrides config)')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path (default: pipeline/data/yoloe_ref_images/post_processing_mask.json)')
    parser.add_argument('--ref_dir', type=str, default=None,
                        help='Reference images directory (default: pipeline/data/yoloe_ref_images)')
    parser.add_argument('--wrist_camera', type=str, default='16478870',
                        help='Wrist camera ID (default: 16478870)')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--host', type=str, default='localhost', help='YOLOE server host')
    parser.add_argument('--port', type=int, default=5559, help='YOLOE server port')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing masks')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--skip_register', action='store_true',
                        help='Skip registration (assume objects already registered)')
    parser.add_argument('--no_fallback_full', action='store_true',
                        help='Use empty mask (zeros) when no detection, instead of full mask (ones)')
    parser.add_argument('--save_debug_videos', type=str, default='/home/yygx/PANDA/pipeline/data/debug_mask_videos',
                        help='Save one overlay video per task folder to this directory for debugging')

    args = parser.parse_args()

    if not args.trajectory_dir and not args.data_dir:
        print("Error: Must specify --trajectory_dir or --data_dir")
        return

    # Setup paths
    script_dir = Path(__file__).parent
    config_path = Path(args.config) if args.config else \
                  script_dir.parent / "data" / "yoloe_ref_images" / "post_processing_mask.json"
    ref_dir = Path(args.ref_dir) if args.ref_dir else \
              script_dir.parent / "data" / "yoloe_ref_images"

    # Load config
    config = load_task_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Connect to server
    try:
        client = YOLOEVisualClient(host=args.host, port=args.port)
    except Exception as e:
        print(f"Error connecting to YOLOE server: {e}")
        print("Make sure YOLOE visual server is running:")
        print("  cd /home/yygx/PANDA/externals/yoloe")
        print("  conda activate yoloe")
        print("  python tests_yy/scripts/yoloe_visual_server.py --port 5559")
        return

    # Get trajectory directories to process
    if args.trajectory_dir:
        trajectory_dirs = [Path(args.trajectory_dir)]
    else:
        trajectory_dirs = find_trajectory_dirs(Path(args.data_dir))
        print(f"Found {len(trajectory_dirs)} trajectory directories")

    # Collect all unique objects needed
    if args.objects:
        all_objects = set(args.objects)
    else:
        all_objects = set()
        for traj_dir in trajectory_dirs:
            task_name = extract_task_name(traj_dir)
            objects = get_objects_for_task(task_name, config)
            all_objects.update(objects)

        # Also check for _default
        if '_default' in config:
            all_objects.update(config['_default'])

    print(f"Objects to detect: {all_objects}")

    # Register reference images if needed
    if not args.skip_register and all_objects:
        print(f"\n{'='*60}")
        print("Registering Reference Images")
        print(f"{'='*60}")

        refs = load_reference_images(ref_dir, list(all_objects))
        if not refs:
            print("Warning: No reference images found!")

        for obj_name, ref_data in refs.items():
            print(f"Registering: {obj_name}")
            response = client.register(obj_name, ref_data['image'], ref_data['bbox'])
            if response.get('success'):
                print(f"  Success! VPE shape: {response.get('vpe_shape')}")
            else:
                print(f"  Failed: {response.get('error')}")

    # Check what's registered
    registered = client.list_objects()
    base_objects = registered.get('base_objects', [])
    print(f"\nRegistered base objects: {base_objects}")

    # Process trajectories
    print(f"\n{'='*60}")
    print("Processing Trajectories")
    print(f"{'='*60}")

    stats = {'processed': 0, 'skipped': 0, 'errors': 0}

    # Track which tasks have had debug videos saved (one per task folder)
    tasks_with_debug_video = set()
    debug_video_dir = Path(args.save_debug_videos) if args.save_debug_videos else None
    debug_videos_saved = []

    for traj_dir in trajectory_dirs:
        # Determine objects for this trajectory
        if args.objects:
            objects = args.objects
        else:
            task_name = extract_task_name(traj_dir)
            objects = get_objects_for_task(task_name, config)
            if not objects:
                print(f"\nSkipping {traj_dir.name}: no objects configured for task '{task_name}'")
                stats['skipped'] += 1
                continue

        # Filter to registered objects
        objects = [o for o in objects if o in base_objects]
        if not objects:
            print(f"\nSkipping {traj_dir.name}: none of the required objects are registered")
            stats['skipped'] += 1
            continue

        # Determine if we should save debug video for this trajectory
        # (only for first trajectory of each task folder)
        task_name = extract_task_name(traj_dir)
        save_debug_video_path = None
        if debug_video_dir and task_name and task_name not in tasks_with_debug_video:
            save_debug_video_path = debug_video_dir / f"{task_name}_mask_overlay.mp4"
            tasks_with_debug_video.add(task_name)

        print(f"\n{'-'*40}")
        result = process_trajectory(
            traj_dir,
            client,
            objects,
            wrist_camera=args.wrist_camera,
            conf=args.conf,
            sample_rate=args.sample_rate,
            overwrite=args.overwrite,
            visualize=args.visualize,
            fallback_full=not args.no_fallback_full,
            save_debug_video=save_debug_video_path
        )

        if result.get('success'):
            stats['processed'] += 1
            if save_debug_video_path:
                debug_videos_saved.append(save_debug_video_path)
        elif result.get('skipped'):
            print(f"  Skipped: {result.get('reason')}")
            stats['skipped'] += 1
            # Remove from tasks_with_debug_video so next trajectory can try
            if task_name in tasks_with_debug_video and save_debug_video_path:
                tasks_with_debug_video.discard(task_name)
        elif result.get('error'):
            print(f"  Error: {result.get('error')}")
            stats['errors'] += 1
            # Remove from tasks_with_debug_video so next trajectory can try
            if task_name in tasks_with_debug_video and save_debug_video_path:
                tasks_with_debug_video.discard(task_name)

    client.close()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")

    # Print debug videos info
    if debug_videos_saved:
        print(f"\n  Debug videos saved ({len(debug_videos_saved)} videos):")
        for video_path in debug_videos_saved:
            print(f"    - {video_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
