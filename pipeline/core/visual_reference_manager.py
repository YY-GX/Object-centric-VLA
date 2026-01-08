#!/usr/bin/env python3
"""
Visual Reference Manager for YOLOE Registration.

Handles registration of visual reference images to YOLOE visual server
for both 3rd-view (FoundationPose) and wrist-view (distractor detection).
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional

from .yoloe_obj_detector_client import YOLOEObjectDetectorClient


def register_3rd_view_references(
    objects: Set[str],
    robot_config: Dict,
    pipeline_root: Path
) -> None:
    """
    Pre-register 3rd-view visual reference images to YOLOE visual server.

    Used for FoundationPose object detection from 3rd-view camera.
    Loads reference images from yoloe_visual_ref_dir and registers them
    with bounding boxes from bboxes.json.

    Args:
        objects: Set of object names to register
        robot_config: Robot configuration dict
        pipeline_root: Path to pipeline root directory
    """
    visual_ref_dir = robot_config.get("yoloe_visual_ref_dir", "")
    if not visual_ref_dir:
        print("⚠️  yoloe_visual_ref_dir not configured, skipping visual ref registration")
        return

    # Resolve path relative to pipeline root
    if not os.path.isabs(visual_ref_dir):
        visual_ref_dir = str(pipeline_root / visual_ref_dir)

    visual_ref_path = Path(visual_ref_dir)
    if not visual_ref_path.exists():
        print(f"⚠️  Visual ref directory not found: {visual_ref_path}")
        return

    # Load bboxes.json if exists
    bboxes_file = visual_ref_path / "bboxes.json"
    bboxes = {}
    if bboxes_file.exists():
        with open(bboxes_file, 'r') as f:
            bboxes = json.load(f)
        print(f"✓ Loaded bboxes from {bboxes_file}")

    # Connect to YOLOE visual server
    visual_server_config = robot_config.get("yoloe_visual_server", {})
    visual_host = visual_server_config.get("host", "localhost")
    visual_port = visual_server_config.get("port", 5559)

    print(f"\n📷 Pre-registering visual references to YOLOE visual server ({visual_host}:{visual_port})...")

    visual_client = YOLOEObjectDetectorClient(host=visual_host, port=visual_port)

    try:
        for obj_name in objects:
            # Find reference images for this object (format: objname_0.jpg, objname_1.jpg, etc.)
            ref_images = sorted(visual_ref_path.glob(f"{obj_name}_*.jpg"))
            if not ref_images:
                ref_images = sorted(visual_ref_path.glob(f"{obj_name}_*.png"))

            if not ref_images:
                print(f"  ⚠️  No visual references found for '{obj_name}'")
                continue

            for ref_img_path in ref_images:
                ref_name = ref_img_path.stem  # e.g., "red_cup_0"

                # Load image
                img = cv2.imread(str(ref_img_path))
                if img is None:
                    print(f"  ⚠️  Failed to load {ref_img_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get bbox if available
                bbox = bboxes.get(ref_name, None)

                # Register to visual server
                response = visual_client.register(
                    object_name=ref_name,
                    image=img_rgb,
                    bbox=bbox
                )

                if response.get('success', False):
                    print(f"  ✓ Registered '{ref_name}' (bbox: {bbox})")
                else:
                    print(f"  ✗ Failed to register '{ref_name}': {response.get('error', 'unknown')}")

    finally:
        visual_client.close()

    print(f"✓ Visual reference registration complete\n")


def register_wrist_view_references(
    scene_objects: Set[str],
    yoloe_client: YOLOEObjectDetectorClient,
    pipeline_root: Path
) -> None:
    """
    Register wrist-view visual references for distractor detection.

    Loads reference images from pipeline/data/yoloe_ref_images/ (wrist view)
    and registers them to YOLOE visual server.
    Called when is_visual_ref_yoloe=True and is_randome_erasing=True.

    Args:
        scene_objects: Set of object names in the scene
        yoloe_client: Connected YOLOE client for registration
        pipeline_root: Path to pipeline root directory
    """
    # Use default wrist ref directory
    wrist_ref_dir = pipeline_root / "data" / "yoloe_ref_images"
    if not wrist_ref_dir.exists():
        print(f"⚠️  Wrist visual ref directory not found: {wrist_ref_dir}")
        return

    # Load bboxes.json
    bboxes_file = wrist_ref_dir / "bboxes.json"
    bboxes = {}
    if bboxes_file.exists():
        with open(bboxes_file, 'r') as f:
            bboxes = json.load(f)

    print(f"\n📷 Registering wrist-view visual references...")
    print(f"  Scene objects: {scene_objects}")

    for obj_name in scene_objects:
        # Find reference images for this object
        ref_images = sorted(wrist_ref_dir.glob(f"{obj_name}_*.jpg"))
        if not ref_images:
            ref_images = sorted(wrist_ref_dir.glob(f"{obj_name}_*.png"))

        if not ref_images:
            print(f"  ⚠️  No wrist refs found for '{obj_name}'")
            continue

        for ref_img_path in ref_images:
            ref_name = ref_img_path.stem

            img = cv2.imread(str(ref_img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bbox = bboxes.get(ref_name, None)

            response = yoloe_client.register(
                object_name=ref_name,
                image=img_rgb,
                bbox=bbox
            )

            if response.get('success', False):
                print(f"  ✓ Registered '{ref_name}'")
            else:
                print(f"  ✗ Failed: '{ref_name}': {response.get('error', 'unknown')}")

    print(f"✓ Wrist visual reference registration complete\n")
