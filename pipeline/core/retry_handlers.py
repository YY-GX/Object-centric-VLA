#!/usr/bin/env python3
"""
Retry Handlers for Real Robot Pipeline.

Handles retry logic for pick and place skills when initial attempts fail.
"""

import numpy as np
import trimesh
from typing import Dict, Optional, Any

from .camera_utils import get_camera_data, transform_pose_camera_to_base


def retry_pick_skill(
    skill_info: Dict,
    initial_states: Dict,
    robot_env: Any,
    pose_client: Any,
    motion_planner: Any,
    target_pose_calculator: Any,
    success_checker: Any,
    robot_config: Dict,
    registration_dict: Dict,
    execute_vla_skill_func,
    csv_logger: Optional[Any] = None
) -> Dict:
    """
    Retry pick skill with fresh object pose.

    Recovery logic:
    1. Re-obtain object pose (fresh tracking)
    2. Recalculate above pose
    3. MP to above pose (gripper open)
    4. VLA execute pick
    5. Move EE up (gripper closed)
    6. Check success

    Args:
        skill_info: Skill information for pick skill
        initial_states: Initial object poses (will be updated)
        robot_env: Robot environment
        pose_client: FoundationPose client
        motion_planner: Motion planner instance
        target_pose_calculator: Target pose calculator
        success_checker: Success checker instance
        robot_config: Robot configuration dict
        registration_dict: Dict of registered objects
        execute_vla_skill_func: Function to execute VLA skill
        csv_logger: CSV logger instance

    Returns:
        Dict with "success", "reason", "final_object_poses"
    """
    target_object = skill_info["target_object"]
    print(f"\n{'='*60}")
    print(f"🔄 RETRY PICK: Re-obtaining pose for '{target_object}'")
    print(f"{'='*60}\n")

    # Load camera intrinsics
    K_config = robot_config["camera_intrinsics"]
    K = np.array([
        [K_config["fx"], 0, K_config["cx"]],
        [0, K_config["fy"], K_config["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Step 1: Re-obtain object pose
    print(f"1️⃣  Re-obtaining object pose...")
    obs = robot_env.get_observation()
    pose_result = pose_client.get_pose(
        object_name=target_object,
        rgb=get_camera_data(obs, 'left', robot_config, 'image'),
        depth=get_camera_data(obs, 'left', robot_config, 'depth'),
        K=K,
        iteration=robot_config["tracking"]["tracking_iteration"]
    )

    if pose_result is None or not pose_result["success"]:
        return {"success": False, "reason": f"Failed to re-obtain pose for {target_object}"}

    pose_cam = np.array(pose_result["pose"])
    object_pose_matrix = transform_pose_camera_to_base(pose_cam, 'left', robot_config)
    object_position = object_pose_matrix[:3, 3]

    # Update initial_states with fresh pose
    initial_states[target_object] = {
        "position": object_position,
        "pose_matrix": object_pose_matrix
    }
    print(f"   ✅ Fresh pose: [{object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f}]")

    # Step 2: Recalculate above pose
    print(f"\n2️⃣  Recalculating above pose...")
    object_pose_dict = {"position": object_position}
    reg_info = registration_dict[target_object]
    mesh = trimesh.load(reg_info["mesh_path"])
    mesh_vertices = np.array(mesh.vertices)

    above_pose = target_pose_calculator.calculate_above_pose(
        object_pose_dict, target_object, mesh_vertices=mesh_vertices
    )
    # Set gripper open for pick approach
    above_pose["gripper"] = 0.0

    # Step 3: MP to above pose (gripper open)
    print(f"\n3️⃣  Moving to above pose (gripper open)...")
    if motion_planner is not None:
        mp_result = motion_planner.move_to_pose(above_pose, method="linear")
        if not mp_result["success"]:
            return {"success": False, "reason": f"Failed to reach above pose"}
    print()

    # Step 4: VLA execute pick
    print(f"4️⃣  Executing VLA pick skill...")
    vla_result = execute_vla_skill_func(
        skill_info=skill_info,
        initial_states=initial_states,
        K=K,
        csv_logger=csv_logger
    )

    if not vla_result["success"]:
        return {"success": False, "reason": f"VLA pick failed: {vla_result['reason']}"}
    print()

    # Step 5: Move EE up (gripper closed)
    print(f"5️⃣  Moving EE up (gripper closed)...")
    if motion_planner is not None:
        # Get current pose and set gripper closed
        current_pose = motion_planner.get_current_ee_pose()
        lift_pose = {
            "position": current_pose["position"] + np.array([0, 0, 0.05]),
            "orientation_euler": current_pose["orientation_euler"],
            "gripper": 1.0  # Keep gripper closed after pick
        }
        motion_planner.move_to_pose(lift_pose, method="linear")
    print()

    # Step 6: Get final pose and check success
    print(f"6️⃣  Checking pick success...")
    obs_final = robot_env.get_observation()
    final_pose_result = pose_client.get_pose(
        object_name=target_object,
        rgb=get_camera_data(obs_final, 'left', robot_config, 'image'),
        depth=get_camera_data(obs_final, 'left', robot_config, 'depth'),
        K=K,
        iteration=robot_config["tracking"]["tracking_iteration"]
    )

    final_object_poses = {}
    if final_pose_result and final_pose_result["success"]:
        pose_cam = np.array(final_pose_result["pose"])
        final_pose_matrix = transform_pose_camera_to_base(pose_cam, 'left', robot_config)
        final_object_poses[target_object] = {
            "position": final_pose_matrix[:3, 3],
            "pose_matrix": final_pose_matrix
        }

    # Check success
    mesh_paths = {name: info["mesh_path"] for name, info in registration_dict.items()}
    success_result = success_checker.check_success(
        success_check_type=skill_info.get("success_check", "object_lifted"),
        skill_info=skill_info,
        initial_states=initial_states,
        current_states=final_object_poses,
        mesh_paths=mesh_paths
    )

    print(f"   {'✅' if success_result['success'] else '❌'} {success_result['reason']}")

    return {
        "success": success_result["success"],
        "reason": success_result["reason"],
        "final_object_poses": final_object_poses
    }


def retry_place_skill(
    place_skill_info: Dict,
    pick_skill_info: Dict,
    initial_states: Dict,
    robot_env: Any,
    pose_client: Any,
    motion_planner: Any,
    target_pose_calculator: Any,
    success_checker: Any,
    robot_config: Dict,
    registration_dict: Dict,
    execute_vla_skill_func,
    csv_logger: Optional[Any] = None
) -> Dict:
    """
    Retry place skill by first re-doing pick, then place.

    Recovery logic:
    1. Re-do pick skill (re-obtain grasp object, pick it up)
    2. Re-obtain target object pose for place
    3. Recalculate above pose for place target
    4. MP to above pose (gripper closed, holding object)
    5. VLA execute place
    6. Move EE up (gripper open)
    7. Check success

    Args:
        place_skill_info: Skill information for place skill
        pick_skill_info: Skill information for previous pick skill
        initial_states: Initial object poses (will be updated)
        robot_env: Robot environment
        pose_client: FoundationPose client
        motion_planner: Motion planner instance
        target_pose_calculator: Target pose calculator
        success_checker: Success checker instance
        robot_config: Robot configuration dict
        registration_dict: Dict of registered objects
        execute_vla_skill_func: Function to execute VLA skill
        csv_logger: CSV logger instance

    Returns:
        Dict with "success", "reason", "final_object_poses"
    """
    grasp_object = place_skill_info.get("grasp_object")
    target_object = place_skill_info["target_object"]

    print(f"\n{'='*60}")
    print(f"🔄 RETRY PLACE: Re-doing pick for '{grasp_object}', then place on '{target_object}'")
    print(f"{'='*60}\n")

    # Step 1: Re-do pick skill first
    print(f"📦 Step 1: Re-doing PICK skill first...")
    pick_result = retry_pick_skill(
        skill_info=pick_skill_info,
        initial_states=initial_states,
        robot_env=robot_env,
        pose_client=pose_client,
        motion_planner=motion_planner,
        target_pose_calculator=target_pose_calculator,
        success_checker=success_checker,
        robot_config=robot_config,
        registration_dict=registration_dict,
        execute_vla_skill_func=execute_vla_skill_func,
        csv_logger=csv_logger
    )

    if not pick_result["success"]:
        return {"success": False, "reason": f"Re-pick failed: {pick_result['reason']}"}

    # Update initial_states with pick result
    if "final_object_poses" in pick_result:
        initial_states.update(pick_result["final_object_poses"])

    print(f"\n✅ Re-pick successful, now proceeding to PLACE...")

    # Load camera intrinsics
    K_config = robot_config["camera_intrinsics"]
    K = np.array([
        [K_config["fx"], 0, K_config["cx"]],
        [0, K_config["fy"], K_config["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Step 2: Re-obtain target object pose for place
    print(f"\n2️⃣  Re-obtaining pose for place target '{target_object}'...")
    obs = robot_env.get_observation()
    pose_result = pose_client.get_pose(
        object_name=target_object,
        rgb=get_camera_data(obs, 'left', robot_config, 'image'),
        depth=get_camera_data(obs, 'left', robot_config, 'depth'),
        K=K,
        iteration=robot_config["tracking"]["tracking_iteration"]
    )

    if pose_result is None or not pose_result["success"]:
        return {"success": False, "reason": f"Failed to re-obtain pose for {target_object}"}

    pose_cam = np.array(pose_result["pose"])
    object_pose_matrix = transform_pose_camera_to_base(pose_cam, 'left', robot_config)
    object_position = object_pose_matrix[:3, 3]

    # Update initial_states with fresh pose
    initial_states[target_object] = {
        "position": object_position,
        "pose_matrix": object_pose_matrix
    }
    print(f"   ✅ Fresh pose: [{object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f}]")

    # Step 3: Recalculate above pose for place
    print(f"\n3️⃣  Recalculating above pose for place...")
    object_pose_dict = {"position": object_position}
    reg_info = registration_dict[target_object]
    mesh = trimesh.load(reg_info["mesh_path"])
    mesh_vertices = np.array(mesh.vertices)

    above_pose = target_pose_calculator.calculate_above_pose(
        object_pose_dict, target_object, mesh_vertices=mesh_vertices
    )
    # Set gripper closed for place approach (holding object)
    above_pose["gripper"] = 1.0

    # Step 4: MP to above pose (gripper closed)
    print(f"\n4️⃣  Moving to above pose (gripper closed, holding object)...")
    if motion_planner is not None:
        mp_result = motion_planner.move_to_pose(above_pose, method="linear")
        if not mp_result["success"]:
            return {"success": False, "reason": f"Failed to reach above pose for place"}
    print()

    # Step 5: VLA execute place
    print(f"5️⃣  Executing VLA place skill...")
    vla_result = execute_vla_skill_func(
        skill_info=place_skill_info,
        initial_states=initial_states,
        K=K,
        csv_logger=csv_logger
    )

    if not vla_result["success"]:
        return {"success": False, "reason": f"VLA place failed: {vla_result['reason']}"}
    print()

    # Step 6: Move EE up (gripper open)
    print(f"6️⃣  Moving EE up (gripper open)...")
    if motion_planner is not None:
        current_pose = motion_planner.get_current_ee_pose()
        lift_pose = {
            "position": current_pose["position"] + np.array([0, 0, 0.05]),
            "orientation_euler": current_pose["orientation_euler"],
            "gripper": 0.0  # Open gripper after place
        }
        motion_planner.move_to_pose(lift_pose, method="linear")
    print()

    # Step 7: Get final poses and check success
    print(f"7️⃣  Checking place success...")
    obs_final = robot_env.get_observation()
    final_object_poses = {}

    # Get final poses for both objects
    for obj_name in [grasp_object, target_object]:
        if obj_name is None:
            continue
        final_pose_result = pose_client.get_pose(
            object_name=obj_name,
            rgb=get_camera_data(obs_final, 'left', robot_config, 'image'),
            depth=get_camera_data(obs_final, 'left', robot_config, 'depth'),
            K=K,
            iteration=robot_config["tracking"]["tracking_iteration"]
        )

        if final_pose_result and final_pose_result["success"]:
            pose_cam = np.array(final_pose_result["pose"])
            final_pose_matrix = transform_pose_camera_to_base(pose_cam, 'left', robot_config)
            final_object_poses[obj_name] = {
                "position": final_pose_matrix[:3, 3],
                "pose_matrix": final_pose_matrix
            }

    # Check success
    mesh_paths = {name: info["mesh_path"] for name, info in registration_dict.items()}
    success_result = success_checker.check_success(
        success_check_type=place_skill_info.get("success_check", "object_on_target"),
        skill_info=place_skill_info,
        initial_states=initial_states,
        current_states=final_object_poses,
        mesh_paths=mesh_paths
    )

    print(f"   {'✅' if success_result['success'] else '❌'} {success_result['reason']}")

    return {
        "success": success_result["success"],
        "reason": success_result["reason"],
        "final_object_poses": final_object_poses
    }
