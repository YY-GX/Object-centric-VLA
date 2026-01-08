#!/usr/bin/env python3
"""
Object Registration for FoundationPose.

Handles initial object pose registration using FoundationPose server.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Any

from .camera_utils import get_camera_data, transform_pose_camera_to_base
from .visual_reference_manager import register_3rd_view_references, register_wrist_view_references


def run_registration_phase(
    robot_env: Any,
    pose_client: Any,
    robot_config: Dict,
    task_config: Dict,
    skill_sequence: List[str],
    yoloe_client: Optional[Any],
    pipeline_root: Path,
    get_skill_info_func
) -> Dict[str, Dict]:
    """
    Run registration for all objects in task.

    For each unique object:
    1. Capture current RGB + depth
    2. Call FoundationPose server to register
    3. Save debug images
    4. Ask user to check results
    5. Return registration results

    Args:
        robot_env: Robot environment for capturing observations
        pose_client: FoundationPose client (ObjectPoseClient)
        robot_config: Robot configuration dict
        task_config: Task configuration dict
        skill_sequence: List of skill names to execute
        yoloe_client: YOLOE client for visual references (optional)
        pipeline_root: Path to pipeline root directory
        get_skill_info_func: Function to get skill info from task config

    Returns:
        Dict mapping object_name -> {pose_camera, pose_base, mesh_path, timestamp}
    """
    # Get unique objects from all skills
    unique_objects = set()
    for skill_name in skill_sequence:
        skill_info = get_skill_info_func(skill_name, task_config)
        if skill_info and "target_object" in skill_info:
            unique_objects.add(skill_info["target_object"])

    print(f"Objects to register: {list(unique_objects)}\n")

    # Check if using visual reference mode
    use_visual_ref = robot_config.get("is_visual_ref_yoloe", False)
    if use_visual_ref:
        print(f"🎯 Visual Reference Mode ENABLED")
        # Pre-register 3rd-view visual references for FoundationPose
        register_3rd_view_references(unique_objects, robot_config, pipeline_root)
        # Pre-register wrist-view visual references for distractor detection
        if yoloe_client is not None:
            # Get scene objects for wrist registration
            scene_objects = _get_scene_objects_for_task(task_config, skill_sequence, get_skill_info_func)
            register_wrist_view_references(scene_objects, yoloe_client, pipeline_root)
    else:
        print(f"📝 Text Prompt Mode (default)")

    # Initialize registration dict
    registration_dict = {}

    # Load configs
    K_config = robot_config["camera_intrinsics"]
    K = np.array([
        [K_config["fx"], 0, K_config["cx"]],
        [0, K_config["fy"], K_config["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)

    object_meshes = robot_config["object_meshes"]
    yoloe_prompts = robot_config["yoloe_prompts"]
    reg_config = robot_config["registration"]

    # Create debug directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path(reg_config["debug_dir"]) / timestamp
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Register each object
    for obj_idx, object_name in enumerate(sorted(unique_objects), 1):
        print(f"{'='*60}")
        print(f"Registering object {obj_idx}/{len(unique_objects)}: {object_name}")
        print(f"{'='*60}\n")

        # Get current observation
        print("📸 Capturing current frame...")
        obs = robot_env.get_observation()
        rgb = get_camera_data(obs, 'left', robot_config, 'image')
        depth = get_camera_data(obs, 'left', robot_config, 'depth')
        print(f"✓ Captured RGB: {rgb.shape}, Depth: {depth.shape} (converted to meters)\n")

        # Get mesh path and YOLOE prompt
        if object_name not in object_meshes:
            print(f"✗ Mesh path not found for '{object_name}' in config!")
            print(f"  Skipping registration for this object.\n")
            continue

        if object_name not in yoloe_prompts:
            print(f"⚠️  YOLOE prompt not found for '{object_name}', using object name")
            yoloe_prompt = object_name
        else:
            yoloe_prompt = yoloe_prompts[object_name]

        mesh_path = object_meshes[object_name]

        # Call registration
        print(f"🔧 Running registration...")
        if use_visual_ref:
            print(f"  Mode: Visual Reference")
            print(f"  Visual ref object: '{object_name}'")
        else:
            print(f"  Mode: Text Prompt")
            print(f"  YOLOE prompt: '{yoloe_prompt}'")
        print(f"  Mesh: {mesh_path}\n")

        result = pose_client.register(
            rgb=rgb,
            depth=depth,
            K=K,
            object_name=object_name,
            yoloe_prompt=yoloe_prompt,
            mesh_path=mesh_path,
            debug_dir=str(debug_dir),
            conf=reg_config.get("registration_conf", 0.1),
            iteration=reg_config.get("registration_iteration", 1),
            debug=0,
            use_visual_ref=use_visual_ref,
            visual_ref_object_name=object_name
        )

        if result is None or not result["success"]:
            print(f"✗ Registration failed for '{object_name}'!")
            print(f"  Message: {result['message'] if result else 'No response from server'}")
            print(f"  You may need to re-run the pipeline.\n")
            raise RuntimeError(f"Registration failed for {object_name}")

        # Registration successful
        print(f"✓ Registration successful!")
        print(f"  Mask pixels: {result['mask_pixels']}")
        print(f"  Confidence: {result['confidence']:.3f}")

        # Transform pose from camera frame to base frame
        pose_cam = np.array(result["pose"])
        pose_base = transform_pose_camera_to_base(pose_cam, 'left', robot_config)
        print(f"  Object position (base frame): [{pose_base[0,3]:.3f}, {pose_base[1,3]:.3f}, {pose_base[2,3]:.3f}]")

        # Store registration results (both camera and base frames)
        registration_dict[object_name] = {
            "pose_camera": pose_cam,      # For FoundationPose tracking
            "pose_base": pose_base,        # For motion planning
            "mesh_path": mesh_path,
            "timestamp": datetime.now().isoformat()
        }

        # Show debug image paths
        print(f"\n📁 Debug images saved:")
        print(f"  Mask: {result['debug_images']['mask_path']}")
        print(f"  Pose: {result['debug_images']['pose_path']}\n")

        # Ask user to check results
        print(f"{'='*60}")
        print(f"⚠️  USER APPROVAL REQUIRED")
        print(f"{'='*60}")
        print(f"Please check the debug images:")
        print(f"  1. Mask visualization: {result['debug_images']['mask_path']}")
        print(f"  2. Pose visualization: {result['debug_images']['pose_path']}")
        print(f"\nVerify that:")
        print(f"  - The object is correctly segmented (green overlay)")
        print(f"  - The coordinate axes align with the object")
        print(f"  - The Z-axis (blue) points forward from the object\n")

        user_input = input(f"Is the registration for '{object_name}' correct? (y/n): ").strip().lower()

        if user_input != 'y':
            print(f"\n✗ Registration rejected by user for '{object_name}'")
            print(f"  Please adjust YOLOE prompt or object position and re-run pipeline.\n")
            raise RuntimeError(f"Registration rejected by user for {object_name}")

        print(f"✓ Registration approved by user\n")

    print(f"\n{'='*60}")
    print(f"✅ All {len(registration_dict)} objects registered successfully!")
    print(f"{'='*60}\n")

    return registration_dict


def _get_scene_objects_for_task(
    task_config: Dict,
    skill_sequence: List[str],
    get_skill_info_func
) -> Set[str]:
    """
    Get all scene objects for current task.

    Args:
        task_config: Task configuration dict
        skill_sequence: List of skill names
        get_skill_info_func: Function to get skill info

    Returns:
        Set of scene object names
    """
    # Get task info to find scene_objects
    scene_objects = set()

    # Try to get from first skill's target object task
    if skill_sequence:
        skill_info = get_skill_info_func(skill_sequence[0], task_config)
        if skill_info and "target_object" in skill_info:
            target = skill_info["target_object"]
            # Find task that contains this target
            for task in task_config.get("long_horizon_tasks", []):
                if target in task.get("scene_objects", []):
                    scene_objects = set(task.get("scene_objects", []))
                    break

    return scene_objects
