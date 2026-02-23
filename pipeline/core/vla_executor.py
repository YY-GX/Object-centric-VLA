#!/usr/bin/env python3
"""
VLA Skill Executor for Real Robot Pipeline.

Handles VLA skill execution with continuous object tracking,
distractor masking, and success checking.

Supports both OpenVLA-OFT (cartesian) and Pi 0.5 (joint) models.
"""

import os
import csv
import time
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional, Any

from .camera_utils import get_camera_data, transform_pose_camera_to_base
from .distractor_masking import apply_distractor_rectangle_masking

# DROID data collection frequency
DROID_CONTROL_FREQUENCY = 15


def _bboxes_overlap(bbox1: list, bbox2: list, iou_threshold: float = 0.3) -> bool:
    """
    Check if two bboxes overlap with IoU above threshold.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        iou_threshold: Minimum IoU to consider as overlap

    Returns:
        True if bboxes overlap with IoU >= threshold
    """
    # Get intersection coordinates
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Check if there's no intersection
    if x2 <= x1 or y2 <= y1:
        return False

    # Compute intersection area
    intersection = (x2 - x1) * (y2 - y1)

    # Compute areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute IoU
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0

    return iou >= iou_threshold

def _save_masked_wrist_video(
    save_dir: str,
    skill_name: str,
    wrist_images: List[np.ndarray]
) -> None:
    """
    Save masked wrist camera video for a skill.

    Args:
        save_dir: Directory to save video
        skill_name: Skill name for filename
        wrist_images: List of wrist camera images (RGB) with random erasing applied
    """
    if not wrist_images:
        print("   Warning: No wrist images to save")
        return

    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        print("   Warning: moviepy not installed, skipping video saving")
        return

    # Clean skill name for filename
    clean_name = skill_name.replace(" ", "_").replace(".", "")
    video_path = os.path.join(save_dir, f"skill_{clean_name}_masked_wrist.mp4")

    try:
        clip = ImageSequenceClip(list(wrist_images), fps=10)
        clip.write_videofile(video_path, codec="libx264", verbose=False, logger=None)
        print(f"   Saved masked wrist video: {video_path}")
    except Exception as e:
        print(f"   Failed to save masked wrist video: {e}")


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
    get_distractor_objects_func=None,
    output_dir: Optional[str] = None,
    save_masked_video: bool = True
) -> Dict:
    """
    Execute VLA skill with continuous object tracking and success checking.

    Supports both Pi 0.5 (joint velocity) and OpenVLA-OFT (cartesian velocity) models.

    Args:
        skill_info: Skill information from task config
        initial_states: Initial object poses for success checking
        K: Camera intrinsics
        robot_env: Robot environment
        vla_client: VLA client for predictions (has model_type attribute)
        pose_client: FoundationPose client for tracking
        success_checker: Success checker instance
        robot_config: Robot configuration dict
        registration_dict: Dict of registered objects
        yoloe_client: YOLOE client for distractor detection (optional)
        yoloe_text_prompts: Text prompts for YOLOE text mode (optional)
        csv_logger: CSV logger instance (optional)
        get_tracked_objects_func: Function to get objects to track for skill
        get_distractor_objects_func: Function to get distractor objects for skill
        output_dir: Directory to save masked wrist video (optional)
        save_masked_video: Whether to save per-skill masked wrist video (default True)

    Returns:
        Dict with "success" and "reason"
    """
    language = skill_info["language"]
    success_check_type = skill_info.get("success_check", "object_lifted")
    objects_to_track = get_tracked_objects_func(skill_info) if get_tracked_objects_func else []
    mesh_paths = {name: info["mesh_path"] for name, info in registration_dict.items()}

    if robot_env is None:
        print(f"   Warning: Robot environment not available")
        return {"success": False, "reason": "Robot environment not available"}

    # Get VLA model type from client
    model_type = getattr(vla_client, 'model_type', 'pi05')
    print(f"   VLA model type: {model_type}")

    vla_config = robot_config["vla_execution"]
    max_timesteps = vla_config["max_timesteps"]
    open_loop_horizon = vla_config["open_loop_horizon"]
    gripper_threshold = vla_config["gripper_threshold"]

    actions_from_chunk_completed = 0
    pred_action_chunk = None

    # Data collection for masked wrist video saving
    rollout_wrist_images = []
    skill_name = skill_info.get("language", "skill")

    # Relative pose state tracking (for is_relative_pose) - only for openvla-oft
    is_relative_pose = robot_config.get("is_relative_pose", False) and model_type == "openvla-oft"
    is_pick_skill = skill_info.get("skill_type") == "pick"
    target_object = skill_info.get("target_object")
    object_pos = None
    if is_relative_pose and target_object and target_object in registration_dict:
        pose_base = registration_dict[target_object]["pose_base"]
        object_pos = pose_base[:3, 3]  # Extract position from 4x4 matrix

    # Stage tracking for pick skills (only for openvla-oft with relative pose)
    gripper_history = []
    stage1_ended = False
    stage1_end_cart_pos = None

    try:
        for t_step in range(max_timesteps):
            print(f"   Step {t_step} of {max_timesteps}")

            # Start timing at the beginning of loop iteration
            start_time_step = time.time()

            # Get current observation
            obs = robot_env.get_observation()

            # Camera returns BGRA, convert to RGB for FoundationPose
            bgra_image = get_camera_data(obs, 'left', robot_config, 'image')
            rgb_image = bgra_image[..., :3][..., ::-1]  # Strip alpha, then BGR to RGB

            # Track all objects and check for success
            current_states = {}
            for obj_name in objects_to_track:
                pose_result = pose_client.get_pose(
                    object_name=obj_name,
                    rgb=rgb_image,
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
            # Auto mode: check every step
            # User decide mode: check every N steps starting from step M
            should_check = False
            if len(current_states) == len(objects_to_track):
                if not success_checker.user_decide_mode:
                    # Auto mode: check every step
                    should_check = True
                else:
                    # User decide mode: check every interval steps starting from start_step
                    success_config = robot_config.get("success_checking", {})
                    start_step = success_config.get("user_decide_start_step", 30)
                    interval = success_config.get("user_decide_interval", 10)
                    if t_step >= start_step and (t_step - start_step) % interval == 0:
                        should_check = True

            if should_check:
                success_result = success_checker.check_success(
                    success_check_type=success_check_type,
                    skill_info=skill_info,
                    initial_states=initial_states,
                    current_states=current_states,
                    mesh_paths=mesh_paths
                )

                if success_result["success"]:
                    print(f"   Success detected at step {t_step}: {success_result['reason']}")
                    # Save masked wrist video before returning
                    if save_masked_video and output_dir:
                        _save_masked_wrist_video(output_dir, skill_name, rollout_wrist_images)
                    return {"success": True, "reason": f"Success at step {t_step}"}

            # Detect distractor objects for random erasing (if enabled)
            masked_wrist_image = None
            if yoloe_client is not None and get_distractor_objects_func is not None:
                distractors = get_distractor_objects_func(skill_info)
                if distractors:
                    wrist_image = get_camera_data(obs, 'wrist', robot_config, 'image')
                    # Convert BGRA to RGB for YOLOE (strip alpha first, then reverse BGR to RGB)
                    wrist_image_rgb = wrist_image[..., :3][..., ::-1] if wrist_image is not None else None
                    if wrist_image_rgb is not None:
                        yoloe_re_config = robot_config.get("yoloe_detection", {}).get("random_erasing", {})
                        yoloe_conf = yoloe_re_config.get("conf_threshold", 0.5)
                        yoloe_min_conf = yoloe_re_config.get("min_conf", 0.5)
                        target_unmask_conf = yoloe_re_config.get("target_unmask_conf", 0.01)
                        print(f"   [DEBUG] YOLOE conf={yoloe_conf}, min_conf={yoloe_min_conf}")

                        # Step 1: Detect target object bbox first (to exclude overlapping distractors)
                        target_object = skill_info.get("target_object")
                        target_bbox = None
                        if target_object:
                            target_result = yoloe_client.detect_with_bbox(
                                wrist_image_rgb,
                                target_object,
                                text_prompts=yoloe_text_prompts,
                                conf=target_unmask_conf
                            )
                            if target_result:
                                # Enlarge target bbox by 1.2x for safety margin
                                bbox = target_result['bbox']
                                if bbox:
                                    x1, y1, x2, y2 = bbox
                                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                    w, h = (x2 - x1) * 1.2, (y2 - y1) * 1.2
                                    target_bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                                    print(f"   Target bbox ({target_object}): {[int(v) for v in target_bbox]}")

                        # Step 2: Detect all distractor objects with their bboxes
                        distractor_detections = yoloe_client.detect_objects_with_bboxes(
                            wrist_image_rgb,
                            distractors,
                            text_prompts=yoloe_text_prompts,
                            conf=yoloe_conf,
                            min_conf=yoloe_min_conf
                        )

                        # Step 3: Filter out distractor bboxes that overlap with target bbox
                        filtered_detections = []
                        for det in distractor_detections:
                            if target_bbox is not None and det['bbox'] is not None:
                                # Check IoU overlap
                                if _bboxes_overlap(det['bbox'], target_bbox, iou_threshold=0.1):
                                    print(f"   Excluded distractor '{det['object_name']}' (overlaps with target)")
                                    continue
                            filtered_detections.append(det)

                        # Step 4: Create union mask from remaining distractor detections
                        if filtered_detections:
                            h, w = wrist_image_rgb.shape[:2]
                            distractor_mask = np.zeros((h, w), dtype=np.uint8)
                            for det in filtered_detections:
                                distractor_mask = np.maximum(distractor_mask, det['mask'])
                                print(f"   Including distractor '{det['object_name']}' in mask")

                            # Step 5: Apply rectangle masking only if mask exists
                            if distractor_mask.sum() > 0:
                                # Check if mask area exceeds max allowed ratio
                                max_mask_area_ratio = yoloe_re_config.get("max_mask_area_ratio", 0.25)
                                total_pixels = distractor_mask.shape[0] * distractor_mask.shape[1]
                                mask_area_ratio = distractor_mask.sum() / total_pixels

                                if mask_area_ratio > max_mask_area_ratio:
                                    print(f"   Skipping distractor masking: mask area {mask_area_ratio:.2%} > max {max_mask_area_ratio:.0%}")
                                else:
                                    masked_wrist_image = apply_distractor_rectangle_masking(
                                        wrist_image_rgb,
                                        distractor_mask,
                                        max_rectangles=5
                                    )
                                    print(f"   Applied distractor masking ({distractor_mask.sum()} pixels, {mask_area_ratio:.2%} of image)")

            # Stage detection for pick skills (detect stage 1 end by looking at past gripper values)
            # Only for openvla-oft with relative pose
            if is_relative_pose:
                current_gripper = obs["robot_state"]["gripper_position"]
                gripper_history.append(current_gripper)

                if is_pick_skill and not stage1_ended and len(gripper_history) >= 4:
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
                            print(f"   Stage 1 ended at step {t_step}, freezing relative pose")

            # Prepare observation for VLA
            vla_obs = prepare_vla_observation(
                obs,
                robot_config,
                model_type=model_type,
                object_pos=object_pos,
                is_pick_skill=is_pick_skill,
                stage1_ended=stage1_ended,
                stage1_end_cart_pos=stage1_end_cart_pos,
                masked_wrist_image=masked_wrist_image
            )

            # Collect masked wrist image for video saving
            if save_masked_video and output_dir:
                rollout_wrist_images.append(vla_obs["wrist_image"].copy())

            # Query VLA server for new action chunk
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= open_loop_horizon:
                actions_from_chunk_completed = 0

                try:
                    pred_action_chunk = vla_client.predict(
                        vla_obs, language, open_loop_horizon
                    )
                except Exception as e:
                    print(f"   VLA prediction failed: {e}")
                    # Save masked wrist video before returning
                    if save_masked_video and output_dir:
                        _save_masked_wrist_video(output_dir, skill_name, rollout_wrist_images)
                    return {"success": False, "reason": f"VLA prediction error: {e}"}

            # Select action from chunk
            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1

            # Binarize gripper action based on model type
            # Pi 0.5: action[7] is gripper, >0.5 means open (1), <=0.5 means close (0)
            # OpenVLA-OFT: action[6] is gripper, >0.5 means open (0), <=0.5 means close (1)
            if model_type == "pi05":
                # Pi 0.5: 8D action [j1..j7, gripper]
                # Gripper logic: >0.5 means open (1), <=0.5 means close (0)
                if action[7] > gripper_threshold:
                    action = np.concatenate([action[:7], np.ones((1,))])   # Open
                else:
                    action = np.concatenate([action[:7], np.zeros((1,))])  # Close
            else:
                # OpenVLA-OFT: 7D action [dx..dyaw, gripper]
                # Gripper logic: >0.5 means open (0), <=0.5 means close (1)
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
                print(f"   Robot step failed: {e}")
                # Save masked wrist video before returning
                if save_masked_video and output_dir:
                    _save_masked_wrist_video(output_dir, skill_name, rollout_wrist_images)
                return {"success": False, "reason": f"Robot step error: {e}"}

            # Sleep to match DROID control frequency
            elapsed_time = time.time() - start_time_step
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

    except KeyboardInterrupt:
        print(f"\n   ⚠️  VLA execution interrupted by user (Ctrl+C)")
        # Save video before re-raising
        if save_masked_video and output_dir:
            _save_masked_wrist_video(output_dir, skill_name, rollout_wrist_images)
        raise  # Re-raise to propagate to main_pipeline

    # Save masked wrist video after loop completes
    if save_masked_video and output_dir:
        _save_masked_wrist_video(output_dir, skill_name, rollout_wrist_images)

    print(f"   Completed VLA execution with tracking ({max_timesteps} steps)")
    return {"success": True, "reason": "Completed"}


def prepare_vla_observation(
    obs: Dict,
    robot_config: Dict,
    model_type: str = "pi05",
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
        model_type: VLA model type ("pi05" or "openvla-oft")
        object_pos: Object position for relative pose (None = use absolute)
        is_pick_skill: Whether current skill is a pick skill
        stage1_ended: Whether pick stage 1 has ended
        stage1_end_cart_pos: Frozen relative pose from stage 1 end
        masked_wrist_image: Pre-masked wrist image (RGB), if provided overrides obs wrist image

    Returns:
        Observation dict for VLA with model-specific fields
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

    # Build observation dict based on model type
    vla_obs = {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "gripper_position": np.array([robot_state["gripper_position"]])
    }

    if model_type == "pi05":
        # Pi 0.5 uses joint positions (7D)
        joint_positions = np.array(robot_state["joint_positions"])
        vla_obs["joint_position"] = joint_positions
        # Also include cartesian for logging purposes
        vla_obs["cartesian_position"] = np.array(robot_state["cartesian_position"])
    else:
        # OpenVLA-OFT uses cartesian position (6D)
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

        vla_obs["cartesian_position"] = cart_pos

    return vla_obs
