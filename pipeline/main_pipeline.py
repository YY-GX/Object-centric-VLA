#!/usr/bin/env python3
"""
Main Pipeline - Real Robot Deployment Pipeline.

This is the main orchestration script that executes long-horizon tasks
on real robot using modular components.

Based on LIBERO's evaluate_above.py but adapted for real robot deployment.

Usage:
    python main_pipeline.py --task_name "pick the black bowl" --max_retries 3
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os
from scipy.spatial.transform import Rotation as R
import trimesh

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15

# Add paths
pipeline_dir = Path(__file__).parent
sys.path.append(str(pipeline_dir))  # For importing pipeline.core, pipeline.utils
sys.path.append(str(pipeline_dir / "../droid"))  # For importing droid.robot_env

# Import pipeline modules
from core import (
    ObjectPoseClient,
    TargetPoseCalculator,
    MotionPlanner,
    SuccessChecker,
    VLAClient
)
from utils import (
    load_task_config,
    plan_task_sequence,
    get_skill_info,
    VideoRecorder,
    CSVLogger
)

try:
    from droid.robot_env import RobotEnv
except ImportError as e:
    raise ValueError(f"RobotEnv not available: {e}")


class RealRobotPipeline:
    """
    Main pipeline for real robot long-horizon task execution.

    Orchestrates all components:
    - Object pose tracking
    - Target pose calculation
    - Motion planning
    - VLA skill execution
    - Success checking
    - Logging and video recording
    """

    def __init__(
        self,
        task_name: str,
        task_config_path: str,
        robot_config_path: str
    ):
        """
        Initialize pipeline.

        Args:
            task_name: Name of long-horizon task
            task_config_path: Path to task_config.json
            robot_config_path: Path to real_robot_config.json
        """
        self.task_name = task_name

        # Load configs
        print(f"{'='*80}")
        print(f"🤖 REAL ROBOT PIPELINE")
        print(f"{'='*80}\n")

        print(f"📋 Loading configuration...")
        self.task_config = load_task_config(task_config_path)
        with open(robot_config_path, 'r') as f:
            self.robot_config = json.load(f)
        print()

        # Plan task sequence
        print(f"🎯 Planning task: '{task_name}'")
        self.skill_sequence = plan_task_sequence(task_name, self.task_config)
        if self.skill_sequence is None:
            raise ValueError(f"Task '{task_name}' not found in config")
        print()

        # Initialize robot environment
        if RobotEnv is not None:
            print(f"🤖 Initializing robot environment...")
            # Enable depth for cameras (required for FoundationPose)
            camera_kwargs = {
                'varied_camera': {'depth': True},  # Left/varying camera
                'hand_camera': {'depth': True}      # Wrist/hand camera
            }
            self.robot_env = RobotEnv(
                action_space="cartesian_velocity",
                gripper_action_space="position",
                camera_kwargs=camera_kwargs
            )
            print(f"✅ Robot environment initialized (depth enabled)")
        else:
            print(f"⚠️  RobotEnv not available")
            raise ValueError("RobotEnv not available")
        print()

        # Setup logging (before initializing modules)
        self.output_dir = Path(self.robot_config["logging"]["save_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_clean = task_name.replace(" ", "_").lower()
        self.run_dir = self.output_dir / f"{task_clean}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Output directory: {self.run_dir}")
        print()

        # Initialize modules
        print(f"🔧 Initializing pipeline modules...")
        self._initialize_modules()
        print()

    def _get_camera_data(self, obs: Dict, camera_role: str, data_type: str = 'image', side: str = 'left') -> np.ndarray:
        """
        Helper to extract camera data by role and convert depth to meters.

        Args:
            obs: Observation dict from robot_env
            camera_role: 'left' or 'wrist'
            data_type: 'image' or 'depth'
            side: 'left' or 'right'

        Returns:
            Image (H,W,3) or depth in meters (H,W) as float32
        """
        serial = self.robot_config['cameras'][f'{camera_role}_camera_id']
        key = f'{serial}_{side}'
        data = obs[data_type][key]

        # Convert depth from millimeters to meters
        if data_type == 'depth':
            data = data.astype(np.float32) / 1000.0

        return data

    def _transform_pose_camera_to_base(self, pose_cam: np.ndarray, camera_role: str = 'left') -> np.ndarray:
        """
        Transform object pose from camera frame to robot base frame.

        Args:
            pose_cam: 4x4 pose matrix in camera frame (from FoundationPose)
            camera_role: 'left' or 'wrist' (default: 'left')

        Returns:
            4x4 pose matrix in robot base frame
        """
        # Get camera extrinsics [x, y, z, roll, pitch, yaw]
        extrinsics_key = f'{camera_role}_camera'
        extrinsics = self.robot_config['camera_extrinsics'][extrinsics_key]

        # Build T_base_from_cam (4x4 transformation matrix)
        T_base_from_cam = np.eye(4, dtype=np.float32)
        T_base_from_cam[:3, :3] = R.from_euler('xyz', extrinsics[3:6]).as_matrix()
        T_base_from_cam[:3, 3] = extrinsics[:3]

        # Transform pose from camera frame to base frame
        pose_base = T_base_from_cam @ pose_cam

        return pose_base

    def _initialize_modules(self):
        """Initialize all pipeline modules and run registration."""
        # Object pose client (FoundationPose server)
        fp_server_config = self.robot_config["foundationpose_server"]
        self.object_pose_client = ObjectPoseClient(
            server_url=fp_server_config["host"],
            port=fp_server_config["port"],
            timeout=30000
        )

        # Target pose calculator
        self.target_pose_calculator = TargetPoseCalculator(
            config=self.robot_config["above_pose"]
        )

        # Motion planner
        if self.robot_env is not None:
            self.motion_planner = MotionPlanner(
                robot_env=self.robot_env,
                config=self.robot_config["motion_planning"]
            )
        else:
            self.motion_planner = None

        # Success checker
        self.success_checker = SuccessChecker(
            config=self.robot_config["success_checking"]
        )

        # VLA client
        vla_server_config = self.robot_config["vla_server"]
        self.vla_client = VLAClient(
            server_host=vla_server_config["host"],
            server_port=vla_server_config["port"]
        )

        # Tracking state management
        self.tracked_objects = set()  # Set of object names currently being tracked

        # Video recorder (if enabled)
        if self.robot_config["logging"]["save_video"]:
            self.video_recorder = VideoRecorder(
                save_dir=self.run_dir / "videos",
                fps=self.robot_config["logging"]["video_fps"]
            )
        else:
            self.video_recorder = None

        # Run registration phase
        print(f"\n{'='*80}")
        print(f"🎯 REGISTRATION PHASE")
        print(f"{'='*80}\n")
        self._run_registration_phase()

        # # Print registration results
        # print(f"\n{'='*80}")
        # print(f"📊 REGISTRATION RESULTS (World/Base Coordinates)")
        # print(f"{'='*80}")
        # for obj_name, obj_data in self.registration_dict.items():
        #     pose_base = obj_data["pose_base"]
        #     position = pose_base[:3, 3]
        #     print(f"\n{obj_name}:")
        #     print(f"  Position (x, y, z): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
        #     print(f"  Full pose matrix (base frame):")
        #     print(f"{pose_base}")
        # print(f"\n{'='*80}\n")

    def _run_registration_phase(self):
        """
        Run registration for all objects in task.

        For each unique object:
        1. Capture current RGB + depth
        2. Call FoundationPose server to register
        3. Save debug images
        4. Ask user to check results
        5. Store registration results in self.registration_dict
        """
        # Get unique objects from all skills
        unique_objects = set()
        for skill_name in self.skill_sequence:
            skill_info = get_skill_info(skill_name, self.task_config)
            if skill_info and "target_object" in skill_info:
                unique_objects.add(skill_info["target_object"])

        print(f"Objects to register: {list(unique_objects)}\n")

        # Initialize registration dict
        self.registration_dict = {}

        # Load configs
        K_config = self.robot_config["camera_intrinsics"]
        K = np.array([
            [K_config["fx"], 0, K_config["cx"]],
            [0, K_config["fy"], K_config["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        object_meshes = self.robot_config["object_meshes"]
        yoloe_prompts = self.robot_config["yoloe_prompts"]
        reg_config = self.robot_config["registration"]

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
            obs = self.robot_env.get_observation()
            rgb = self._get_camera_data(obs, 'left', 'image')
            depth = self._get_camera_data(obs, 'left', 'depth')
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
            print(f"  YOLOE prompt: '{yoloe_prompt}'")
            print(f"  Mesh: {mesh_path}\n")

            result = self.object_pose_client.register(
                rgb=rgb,
                depth=depth,
                K=K,
                object_name=object_name,
                yoloe_prompt=yoloe_prompt,
                mesh_path=mesh_path,
                debug_dir=str(debug_dir),
                conf=reg_config.get("registration_conf", 0.1),
                iteration=reg_config.get("registration_iteration", 1),
                debug=0
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
            pose_base = self._transform_pose_camera_to_base(pose_cam, camera_role='left')
            print(f"  Object position (base frame): [{pose_base[0,3]:.3f}, {pose_base[1,3]:.3f}, {pose_base[2,3]:.3f}]")

            # Store registration results (both camera and base frames)
            self.registration_dict[object_name] = {
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
        print(f"✅ All {len(self.registration_dict)} objects registered successfully!")
        print(f"{'='*60}\n")

    def _get_tracked_objects_for_skill(self, skill_info: Dict) -> list:
        """
        Get list of objects that need to be tracked for this skill.

        Args:
            skill_info: Skill information from task config

        Returns:
            List of object names to track
        """
        objects = []

        # Always track target_object
        if "target_object" in skill_info:
            objects.append(skill_info["target_object"])

        # For place skills, also track grasp_object
        if skill_info.get("skill_type") == "place" and "grasp_object" in skill_info:
            objects.append(skill_info["grasp_object"])

        return objects

    def _should_keep_tracking(self, object_name: str, current_skill_idx: int) -> bool:
        """
        Check if object should continue being tracked after current skill.

        Args:
            object_name: Object to check
            current_skill_idx: Index of current skill in sequence

        Returns:
            True if next skill needs this object
        """
        # Check if there's a next skill
        if current_skill_idx + 1 >= len(self.skill_sequence):
            return False

        # Get next skill info
        next_skill_name = self.skill_sequence[current_skill_idx + 1]
        next_skill_info = get_skill_info(next_skill_name, self.task_config)

        if not next_skill_info:
            return False

        # Check if next skill needs this object
        next_tracked_objects = self._get_tracked_objects_for_skill(next_skill_info)
        return object_name in next_tracked_objects

    def execute_task(self, max_retries: int = 3) -> bool:
        """
        Execute long-horizon task.

        Main execution loop following evaluate_above.py logic:
        For each skill:
            1. Get object pose
            2. Calculate above pose
            3. Move to above pose (motion planner)
            4. Execute VLA skill
            5. Move EE up
            6. Check success
            7. Retry on failure

        Args:
            max_retries: Maximum retries per skill

        Returns:
            True if all skills succeeded
        """
        print(f"{'='*80}")
        print(f"🚀 STARTING TASK EXECUTION")
        print(f"{'='*80}\n")

        # Initialize logging
        if self.robot_config["logging"]["save_csv"]:
            csv_logger = CSVLogger(self.run_dir, prefix=f"task_{self.task_name.replace(' ', '_')}")
        else:
            csv_logger = None

        # Track initial object states
        initial_states = {}

        success_count = 0
        total_skills = len(self.skill_sequence)

        for skill_idx, skill_name in enumerate(self.skill_sequence, 1):
            print(f"\n{'='*80}")
            print(f"📍 SKILL {skill_idx}/{total_skills}: {skill_name}")
            print(f"{'='*80}\n")

            # Get skill info
            skill_info = get_skill_info(skill_name, self.task_config)
            if skill_info is None:
                print(f"❌ Skill info not found. Skipping...")
                raise ValueError(f"Skill info not found for skill: {skill_name}")

            target_object = skill_info["target_object"]
            language = skill_info["language"]
            skill_type = skill_info["skill_type"]

            # Execute skill with retry logic
            skill_success = False
            for retry in range(max_retries):
                if retry > 0:
                    print(f"\n🔄 Retry {retry}/{max_retries-1}")

                result = self._execute_single_skill(
                    skill_info=skill_info,
                    skill_idx=skill_idx - 1,  # Convert to 0-indexed
                    initial_states=initial_states,
                    csv_logger=csv_logger
                )

                if result["success"]:
                    skill_success = True
                    # Update initial states for next skill (handles multiple objects)
                    if "final_object_poses" in result:
                        initial_states.update(result["final_object_poses"])
                    break
                else:
                    print(f"❌ Skill failed: {result['reason']}")
                    if retry < max_retries - 1:
                        print(f"   Retrying in 2 seconds...")
                        time.sleep(2.0)

            if skill_success:
                success_count += 1
                print(f"\n✅ Skill {skill_idx}/{total_skills} completed successfully")
            else:
                print(f"\n❌ Skill {skill_idx}/{total_skills} failed after {max_retries} retries")
                print(f"   Aborting task execution")
                break

        # Close logging
        if csv_logger is not None:
            csv_logger.close()

        # Final summary
        print(f"\n{'='*80}")
        print(f"📊 TASK EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Task: {self.task_name}")
        print(f"Skills completed: {success_count}/{total_skills}")
        print(f"Success rate: {success_count/total_skills*100:.1f}%")
        print(f"Output directory: {self.run_dir}")
        print(f"{'='*80}\n")

        return success_count == total_skills

    def _execute_single_skill(
        self,
        skill_info: Dict,
        skill_idx: int,
        initial_states: Dict,
        csv_logger: Optional[CSVLogger]
    ) -> Dict:
        """
        Execute single atomic skill with multi-object tracking support.

        Args:
            skill_info: Skill information from task config
            skill_idx: Index of skill in sequence (0-indexed)
            initial_states: Initial object poses (updated in-place)
            csv_logger: CSV logger instance

        Returns:
            Dict with "success", "reason", "final_object_poses"
        """
        skill_name = f"{skill_info['skill_type']} {skill_info['target_object']}"
        target_object = skill_info["target_object"]
        language = skill_info["language"]
        skill_type = skill_info["skill_type"]

        # Get list of objects to track for this skill
        objects_to_track = self._get_tracked_objects_for_skill(skill_info)

        # Load camera intrinsics
        K_config = self.robot_config["camera_intrinsics"]
        K = np.array([
            [K_config["fx"], 0, K_config["cx"]],
            [0, K_config["fy"], K_config["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 1: Start tracking for all needed objects
        print(f"1️⃣  Starting tracking for {len(objects_to_track)} object(s): {objects_to_track}...")
        for obj_name in objects_to_track:
            # Skip if already tracking
            if obj_name in self.tracked_objects:
                print(f"   ⏩ {obj_name} already tracking (from previous skill)")
                continue

            if obj_name not in self.registration_dict:
                return {"success": False, "reason": f"Object '{obj_name}' not registered"}

            reg_info = self.registration_dict[obj_name]
            track_result = self.object_pose_client.start_tracking(
                K=K,
                mesh_path=reg_info["mesh_path"],
                initial_pose=reg_info["pose_camera"],
                object_name=obj_name
            )

            if track_result is None or not track_result["success"]:
                return {"success": False, "reason": f"Failed to start tracking {obj_name}"}

            self.tracked_objects.add(obj_name)
            print(f"   ✅ Started tracking: {obj_name}")

        print()

        # Step 2: Collect initial poses for all tracked objects
        print(f"2️⃣  Collecting initial poses for tracked objects...")
        obs = self.robot_env.get_observation()
        for obj_name in objects_to_track:
            # Skip if already have initial state
            if obj_name in initial_states:
                print(f"   ⏩ {obj_name} initial state already collected")
                continue

            pose_result = self.object_pose_client.get_pose(
                object_name=obj_name,
                rgb=self._get_camera_data(obs, 'left', 'image'),
                depth=self._get_camera_data(obs, 'left', 'depth'),
                K=K,
                iteration=self.robot_config["tracking"]["tracking_iteration"]
            )

            if pose_result is None or not pose_result["success"]:
                return {"success": False, "reason": f"Failed to get initial pose for {obj_name}"}

            pose_cam = np.array(pose_result["pose"])
            object_pose_matrix = self._transform_pose_camera_to_base(pose_cam, camera_role='left')
            object_position = object_pose_matrix[:3, 3]
            initial_states[obj_name] = {"position": object_position, "pose_matrix": object_pose_matrix}
            print(f"   ✅ {obj_name}: [{object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f}]")

        print()

        # Step 3: Calculate above pose
        print(f"3️⃣  Calculating above pose...")
        object_pose_dict = {"position": target_obj_pose["position"]}

        # Load mesh to get object size
        reg_info = self.registration_dict[target_object]
        mesh = trimesh.load(reg_info["mesh_path"])
        mesh_vertices = np.array(mesh.vertices)

        above_pose = self.target_pose_calculator.calculate_above_pose(
            object_pose_dict, target_object, mesh_vertices=mesh_vertices
        )
        print()

        # Step 4: Move to above pose
        if self.motion_planner is not None:
            print(f"4️⃣  Moving to above pose...")

            # Print current EE pose before motion
            current_state = self.robot_env.get_observation()["robot_state"]
            current_pos = current_state["cartesian_position"][:3]
            current_euler = current_state["cartesian_position"][3:6]
            print(f"   Current EE pose:")
            print(f"     Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            print(f"     Orientation (euler): [{current_euler[0]:.3f}, {current_euler[1]:.3f}, {current_euler[2]:.3f}]")

            mp_result = self.motion_planner.move_to_pose(above_pose, method="linear")

            # Print final EE pose after motion
            final_state = self.robot_env.get_observation()["robot_state"]
            final_pos = final_state["cartesian_position"][:3]
            final_euler = final_state["cartesian_position"][3:6]
            print(f"   Final EE pose:")
            print(f"     Position: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
            print(f"     Orientation (euler): [{final_euler[0]:.3f}, {final_euler[1]:.3f}, {final_euler[2]:.3f}]")

            if not mp_result["success"]:
                self.object_pose_client.end_tracking()
                return {
                    "success": False,
                    "reason": f"Failed to reach above pose (distance: {mp_result['final_distance']:.3f}m)"
                }
            print()
        else:
            print(f"4️⃣  Motion planner not available (skipping)")
            print()

        # Step 5: Execute VLA skill (with tracking and success checking)
        print(f"5️⃣  Executing VLA skill: '{language}'...")
        vla_result = self._execute_vla_skill(
            skill_info=skill_info,
            initial_states=initial_states,
            K=K,
            csv_logger=csv_logger
        )

        if not vla_result["success"]:
            # Smart end tracking
            for obj_name in list(self.tracked_objects):
                if not self._should_keep_tracking(obj_name, skill_idx):
                    self.object_pose_client.end_tracking(object_name=obj_name)
                    self.tracked_objects.remove(obj_name)
            return {"success": False, "reason": f"VLA execution failed: {vla_result['reason']}"}
        print()

        # Step 6: Move EE up
        if self.motion_planner is not None:
            print(f"6️⃣  Moving EE up...")
            self.motion_planner.move_ee_up(lift_distance=0.05, skill_type=skill_type)
            print()
        else:
            print(f"6️⃣  Motion planner not available (skipping)")
            print()

        # Step 7: Get final poses for all tracked objects
        print(f"7️⃣  Getting final poses for tracked objects...")
        obs_final = self.robot_env.get_observation()
        final_object_poses = {}

        for obj_name in objects_to_track:
            final_pose_result = self.object_pose_client.get_pose(
                object_name=obj_name,
                rgb=self._get_camera_data(obs_final, 'left', 'image'),
                depth=self._get_camera_data(obs_final, 'left', 'depth'),
                K=K,
                iteration=self.robot_config["tracking"]["tracking_iteration"]
            )

            if final_pose_result and final_pose_result["success"]:
                pose_cam = np.array(final_pose_result["pose"])
                final_pose_matrix = self._transform_pose_camera_to_base(pose_cam, camera_role='left')
                final_object_poses[obj_name] = {
                    "position": final_pose_matrix[:3, 3],
                    "pose_matrix": final_pose_matrix
                }
                final_pos = final_pose_matrix[:3, 3]
                print(f"   ✅ {obj_name}: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
        print()

        # Step 8: Smart end tracking
        print(f"8️⃣  Managing tracking sessions...")
        for obj_name in list(self.tracked_objects):
            if not self._should_keep_tracking(obj_name, skill_idx):
                self.object_pose_client.end_tracking(object_name=obj_name)
                self.tracked_objects.remove(obj_name)
                print(f"   ⏹  Ended tracking: {obj_name}")
            else:
                print(f"   ⏩ Keeping tracking: {obj_name} (needed for next skill)")
        print()

        # Step 9: Check skill success using new method
        print(f"9️⃣  Checking skill success...")
        success_check_type = skill_info.get("success_check", "object_lifted")
        mesh_paths = {name: info["mesh_path"] for name, info in self.registration_dict.items()}

        success_result = self.success_checker.check_success(
            success_check_type=success_check_type,
            skill_info=skill_info,
            initial_states=initial_states,
            current_states=final_object_poses,
            mesh_paths=mesh_paths
        )

        print(f"   {'✅' if success_result['success'] else '❌'} {success_result['reason']}")
        print()

        return {
            "success": success_result["success"],
            "reason": success_result["reason"],
            "confidence": success_result["confidence"],
            "final_object_poses": final_object_poses
        }


    def _execute_vla_skill(
        self,
        skill_info: Dict,
        initial_states: Dict,
        K: np.ndarray,
        csv_logger: Optional[CSVLogger]
    ) -> Dict:
        """
        Execute VLA skill with continuous object tracking and success checking.

        Args:
            skill_info: Skill information from task config
            initial_states: Initial object poses for success checking
            K: Camera intrinsics
            csv_logger: CSV logger instance

        Returns:
            Dict with "success" and "reason"
        """
        language = skill_info["language"]
        success_check_type = skill_info.get("success_check", "object_lifted")
        objects_to_track = self._get_tracked_objects_for_skill(skill_info)
        mesh_paths = {name: info["mesh_path"] for name, info in self.registration_dict.items()}
        if self.robot_env is None:
            print(f"   ⚠️  Robot environment not available")
            return {"success": False, "reason": "Robot environment not available"}

        vla_config = self.robot_config["vla_execution"]
        max_timesteps = vla_config["max_timesteps"]
        open_loop_horizon = vla_config["open_loop_horizon"]
        gripper_threshold = vla_config["gripper_threshold"]

        actions_from_chunk_completed = 0
        pred_action_chunk = None

        for t_step in range(max_timesteps):
            # Start timing at the beginning of loop iteration
            start_time_step = time.time()

            # Get current observation
            obs = self.robot_env.get_observation()

            # Track all objects and check for success
            current_states = {}
            for obj_name in objects_to_track:
                pose_result = self.object_pose_client.get_pose(
                    object_name=obj_name,
                    rgb=self._get_camera_data(obs, 'left', 'image'),
                    depth=self._get_camera_data(obs, 'left', 'depth'),
                    K=K,
                    iteration=self.robot_config["tracking"]["tracking_iteration"]
                )
                if pose_result and pose_result["success"]:
                    pose_cam = np.array(pose_result["pose"])
                    pose_matrix = self._transform_pose_camera_to_base(pose_cam, camera_role='left')
                    current_states[obj_name] = {
                        "position": pose_matrix[:3, 3],
                        "pose_matrix": pose_matrix
                    }

            # Check skill success
            if len(current_states) == len(objects_to_track):
                success_result = self.success_checker.check_success(
                    success_check_type=success_check_type,
                    skill_info=skill_info,
                    initial_states=initial_states,
                    current_states=current_states,
                    mesh_paths=mesh_paths
                )

                if success_result["success"]:
                    print(f"   ✅ Success detected at step {t_step}: {success_result['reason']}")
                    return {"success": True, "reason": f"Success at step {t_step}"}

            # Prepare observation for VLA
            vla_obs = self._prepare_vla_observation(obs)

            # Query VLA server for new action chunk
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= open_loop_horizon:
                actions_from_chunk_completed = 0

                try:
                    pred_action_chunk = self.vla_client.predict(
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

            # Log actions
            if csv_logger is not None:
                csv_logger.log_action(t_step, action)

            # Execute action
            try:
                self.robot_env.step(action)
            except Exception as e:
                print(f"   ❌ Robot step failed: {e}")
                return {"success": False, "reason": f"Robot step error: {e}"}

            # Sleep to match DROID control frequency
            elapsed_time = time.time() - start_time_step
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

        print(f"   ✅ Completed VLA execution with tracking ({max_timesteps} steps)")
        return {"success": True, "reason": "Completed"}

    def _prepare_vla_observation(self, obs: Dict) -> Dict:
        """
        Prepare observation for VLA client.

        Args:
            obs: Raw observation from robot_env

        Returns:
            Observation dict for VLA
        """
        image_obs = obs["image"]
        robot_state = obs["robot_state"]

        # Extract camera images
        # Camera IDs from config
        left_camera_id = self.robot_config["cameras"]["left_camera_id"]
        wrist_camera_id = self.robot_config["cameras"]["wrist_camera_id"]

        # Find images by camera ID
        left_image = None
        wrist_image = None

        for key, img in image_obs.items():
            if left_camera_id in key and "left" in key:
                left_image = img[..., :3][..., ::-1]  # Drop alpha, convert BGR to RGB
            elif wrist_camera_id in key and "left" in key:
                wrist_image = img[..., :3][..., ::-1]

        if left_image is None or wrist_image is None:
            raise ValueError(f"Could not find camera images in observation")

        return {
            "left_image": left_image,
            "wrist_image": wrist_image,
            "cartesian_position": np.array(robot_state["cartesian_position"]),
            "gripper_position": np.array([robot_state["gripper_position"]])
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real Robot Deployment Pipeline")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of long-horizon task (e.g., 'pick the black bowl')"
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="config/task_config.json",
        help="Path to task config file"
    )
    parser.add_argument(
        "--robot_config",
        type=str,
        default="config/real_robot_config.json",
        help="Path to robot config file"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=6,
        help="Maximum retries per skill"
    )

    args = parser.parse_args()

    # Resolve config paths relative to pipeline directory
    pipeline_dir = Path(__file__).parent
    task_config_path = pipeline_dir / args.task_config
    robot_config_path = pipeline_dir / args.robot_config

    # Initialize and run pipeline
    pipeline = RealRobotPipeline(
        task_name=args.task_name,
        task_config_path=str(task_config_path),
        robot_config_path=str(robot_config_path)
    )

    # exit(0)

    success = pipeline.execute_task(max_retries=args.max_retries)

    if success:
        print("🎉 Task completed successfully!")
        return 0
    else:
        print("❌ Task failed")
        return 1


if __name__ == "__main__":
    exit(main())
