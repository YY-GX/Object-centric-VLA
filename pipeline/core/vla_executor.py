#!/usr/bin/env python3
"""
VLA Skill Executor for Real Robot Pipeline.

Handles VLA skill execution with continuous object tracking,
distractor masking, and success checking.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any

from .camera_utils import get_camera_data, transform_pose_camera_to_base
from .distractor_masking import apply_distractor_rectangle_masking

# DROID data collection frequency
DROID_CONTROL_FREQUENCY = 15


def execute_vla_skill(
    skill_info: Dict,
    initial_states: Dict,
    K: np.ndarray,
    robot_env: Any,
    vla_client: Any,
    pose_client: Any,
    success_checker: Any,
    robot_config: Dict,
    registration_dict: Dict,
    yoloe_client: Optional[Any] = None,
    yoloe_text_prompts: Optional[Dict] = None,
    csv_logger: Optional[Any] = None,
    get_tracked_objects_func=None,
    get_distractor_objects_func=None
) -> Dict:
    """
    Execute VLA skill with continuous object tracking and success checking.

    Args:
        skill_info: Skill information from task config
        initial_states: Initial object poses for success checking
        K: Camera intrinsics
        robot_env: Robot environment
        vla_client: VLA client for predictions
        pose_client: FoundationPose client for tracking
        success_checker: Success checker instance
        robot_config: Robot configuration dict
        registration_dict: Dict of registered objects
        yoloe_client: YOLOE client for distractor detection (optional)
        yoloe_text_prompts: Text prompts for YOLOE text mode (optional)
        csv_logger: CSV logger instance (optional)
        get_tracked_objects_func: Function to get objects to track for skill
        get_distractor_objects_func: Function to get distractor objects for skill

    Returns:
        Dict with "success" and "reason"
    """
    language = skill_info["language"]
    success_check_type = skill_info.get("success_check", "object_lifted")
    objects_to_track = get_tracked_objects_func(skill_info) if get_tracked_objects_func else []
    mesh_paths = {name: info["mesh_path"] for name, info in registration_dict.items()}

    if robot_env is None:
        print(f"   ⚠️  Robot environment not available")
        return {"success": False, "reason": "Robot environment not available"}

    vla_config = robot_config["vla_execution"]
    max_timesteps = vla_config["max_timesteps"]
    open_loop_horizon = vla_config["open_loop_horizon"]
    gripper_threshold = vla_config["gripper_threshold"]

    actions_from_chunk_completed = 0
    pred_action_chunk = None

    # Relative pose state tracking (for is_relative_pose)
    is_relative_pose = robot_config.get("is_relative_pose", False)
    is_pick_skill = skill_info.get("skill_type") == "pick"
    target_object = skill_info.get("target_object")
    object_pos = None
    if is_relative_pose and target_object and target_object in registration_dict:
        pose_base = registration_dict[target_object]["pose_base"]
        object_pos = pose_base[:3, 3]  # Extract position from 4x4 matrix

    # Stage tracking for pick skills
    gripper_history = []
    stage1_ended = False
    stage1_end_cart_pos = None

    for t_step in range(max_timesteps):
        print(f"   🔄 Step {t_step} of {max_timesteps}")

        # Start timing at the beginning of loop iteration
        start_time_step = time.time()

        # Get current observation
        obs = robot_env.get_observation()

        # Track all objects and check for success
        current_states = {}
        for obj_name in objects_to_track:
            pose_result = pose_client.get_pose(
                object_name=obj_name,
                rgb=get_camera_data(obs, 'left', robot_config, 'image'),
                depth=get_camera_data(obs, 'left', robot_config, 'depth'),
                K=K,
                iteration=robot_config["tracking"]["tracking_iteration"]
            )
            if pose_result and pose_result["success"]:
                pose_cam = np.array(pose_result["pose"])
                pose_matrix = transform_pose_camera_to_base(pose_cam, 'left', robot_config)
                current_states[obj_name] = {
                    "position": pose_matrix[:3, 3],
                    "pose_matrix": pose_matrix
                }

        # Check skill success
        if len(current_states) == len(objects_to_track):
            success_result = success_checker.check_success(
                success_check_type=success_check_type,
                skill_info=skill_info,
                initial_states=initial_states,
                current_states=current_states,
                mesh_paths=mesh_paths
            )

            if success_result["success"]:
                print(f"   ✅ Success detected at step {t_step}: {success_result['reason']}")
                return {"success": True, "reason": f"Success at step {t_step}"}

        # Detect distractor objects for random erasing (if enabled)
        masked_wrist_image = None
        if yoloe_client is not None and get_distractor_objects_func is not None:
            distractors = get_distractor_objects_func(skill_info)
            if distractors:
                wrist_image = get_camera_data(obs, 'wrist', robot_config, 'image')
                # Convert BGR to RGB for YOLOE
                wrist_image_rgb = wrist_image[..., ::-1] if wrist_image is not None else None
                if wrist_image_rgb is not None:
                    distractor_mask = yoloe_client.detect_and_union(
                        wrist_image_rgb,
                        distractors,
                        text_prompts=yoloe_text_prompts,
                        conf=0.1
                    )
                    # Apply rectangle masking to distractor regions
                    if distractor_mask is not None and distractor_mask.sum() > 0:
                        masked_wrist_image = apply_distractor_rectangle_masking(
                            wrist_image_rgb,
                            distractor_mask,
                            max_rectangles=5
                        )
                        print(f"   🎭 Applied distractor masking ({distractor_mask.sum()} pixels detected)")

        # Stage detection for pick skills (detect stage 1 end by looking at past gripper values)
        current_gripper = obs["robot_state"]["gripper_position"]
        gripper_history.append(current_gripper)

        if is_relative_pose and is_pick_skill and not stage1_ended and len(gripper_history) >= 4:
            # Check if gripper closed and stable for past 3 steps
            g = gripper_history
            gripper_stable_threshold = 1e-4
            # Gripper closed (current value high) and stable
            if (g[-1] > 0.5 and  # Gripper closed
                abs(g[-1] - g[-2]) < gripper_stable_threshold and
                abs(g[-2] - g[-3]) < gripper_stable_threshold and
                abs(g[-3] - g[-4]) < gripper_stable_threshold):
                stage1_ended = True
                # Compute and store the relative pose at this moment
                ee_pos = np.array(obs["robot_state"]["cartesian_position"])
                if object_pos is not None:
                    rel_pos = ee_pos[:3] - object_pos
                    stage1_end_cart_pos = np.concatenate([rel_pos, ee_pos[3:]])
                    print(f"   📍 Stage 1 ended at step {t_step}, freezing relative pose")

        # Prepare observation for VLA
        vla_obs = prepare_vla_observation(
            obs,
            robot_config,
            object_pos=object_pos,
            is_pick_skill=is_pick_skill,
            stage1_ended=stage1_ended,
            stage1_end_cart_pos=stage1_end_cart_pos,
            masked_wrist_image=masked_wrist_image
        )

        # Query VLA server for new action chunk
        if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= open_loop_horizon:
            actions_from_chunk_completed = 0

            try:
                pred_action_chunk = vla_client.predict(
                    vla_obs, language, open_loop_horizon
                )
            except Exception as e:
                print(f"   ❌ VLA prediction failed: {e}")
                return {"success": False, "reason": f"VLA prediction error: {e}"}

        # Select action from chunk
        action = pred_action_chunk[actions_from_chunk_completed]
        actions_from_chunk_completed += 1

        # Binarize gripper action (match evaluate_openvla_oft.py pattern)
        if action[6] > gripper_threshold:
            action = np.concatenate([action[:6], np.zeros((1,))])  # Open
        else:
            action = np.concatenate([action[:6], np.ones((1,))])   # Close

        # Clip action to safe range
        action = np.clip(action, -1, 1)

        # Log actions and state
        if csv_logger is not None:
            csv_logger.log_step(t_step, action, obs["robot_state"])

        # Execute action
        try:
            robot_env.step(action)
        except Exception as e:
            print(f"   ❌ Robot step failed: {e}")
            return {"success": False, "reason": f"Robot step error: {e}"}

        # Sleep to match DROID control frequency
        elapsed_time = time.time() - start_time_step
        if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

    print(f"   ✅ Completed VLA execution with tracking ({max_timesteps} steps)")
    return {"success": True, "reason": "Completed"}


def prepare_vla_observation(
    obs: Dict,
    robot_config: Dict,
    object_pos: np.ndarray = None,
    is_pick_skill: bool = False,
    stage1_ended: bool = False,
    stage1_end_cart_pos: np.ndarray = None,
    masked_wrist_image: np.ndarray = None
) -> Dict:
    """
    Prepare observation for VLA client.

    Args:
        obs: Raw observation from robot_env
        robot_config: Robot configuration dict
        object_pos: Object position for relative pose (None = use absolute)
        is_pick_skill: Whether current skill is a pick skill
        stage1_ended: Whether pick stage 1 has ended
        stage1_end_cart_pos: Frozen relative pose from stage 1 end
        masked_wrist_image: Pre-masked wrist image (RGB), if provided overrides obs wrist image

    Returns:
        Observation dict for VLA
    """
    image_obs = obs["image"]
    robot_state = obs["robot_state"]

    # Extract camera images
    left_camera_id = robot_config["cameras"]["left_camera_id"]
    wrist_camera_id = robot_config["cameras"]["wrist_camera_id"]

    left_image = None
    wrist_image = None

    for key, img in image_obs.items():
        if left_camera_id in key and "left" in key:
            left_image = img[..., :3][..., ::-1]  # Drop alpha, convert BGR to RGB
        elif wrist_camera_id in key and "left" in key:
            wrist_image = img[..., :3][..., ::-1]

    if left_image is None or wrist_image is None:
        raise ValueError(f"Could not find camera images in observation")

    # Use masked wrist image if provided
    if masked_wrist_image is not None:
        wrist_image = masked_wrist_image

    # Compute cartesian_position (relative or absolute)
    abs_cart_pos = np.array(robot_state["cartesian_position"])
    is_relative_pose = robot_config.get("is_relative_pose", False)

    if is_relative_pose and object_pos is not None:
        if is_pick_skill and stage1_ended and stage1_end_cart_pos is not None:
            # Pick skill stage 2: use frozen relative pose
            cart_pos = stage1_end_cart_pos
        else:
            # Stage 1 or place skill: compute relative position + original orientation
            rel_pos = abs_cart_pos[:3] - object_pos
            cart_pos = np.concatenate([rel_pos, abs_cart_pos[3:]])
    else:
        # Use absolute position
        cart_pos = abs_cart_pos

    return {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "cartesian_position": cart_pos,
        "gripper_position": np.array([robot_state["gripper_position"]])
    }
