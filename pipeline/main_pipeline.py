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
import cv2

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
from core.yoloe_obj_detector_client import YOLOEObjectDetectorClient
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

        # YOLOE client for distractor detection (if random erasing enabled)
        self.yoloe_client = None
        self.yoloe_text_prompts = None  # For text mode
        if self.robot_config.get("is_randome_erasing", False):
            use_visual_ref = self.robot_config.get("is_visual_ref_yoloe", False)
            if use_visual_ref:
                # Visual mode: use port 5559
                self.yoloe_client = YOLOEObjectDetectorClient(
                    host="localhost",
                    port=5559,
                    mode="visual"
                )
                print(f"✅ YOLOE client initialized (visual mode, port 5559)")
            else:
                # Text mode: use port 5557
                self.yoloe_client = YOLOEObjectDetectorClient(
                    host="localhost",
                    port=5557,
                    mode="text"
                )
                # Store text prompts for distractor detection
                self.yoloe_text_prompts = self.robot_config.get("yoloe_prompts", {})
                print(f"✅ YOLOE client initialized (text mode, port 5557)")

        # Run registration phase
        print(f"\n{'='*80}")
        print(f"🎯 REGISTRATION PHASE")
        print(f"{'='*80}\n")
        self._run_registration_phase()

        # Print registration results
        print(f"\n{'='*80}")
        print(f"📊 REGISTRATION RESULTS (World/Base Coordinates)")
        print(f"{'='*80}")
        for obj_name, obj_data in self.registration_dict.items():
            pose_base = obj_data["pose_base"]
            position = pose_base[:3, 3]
            print(f"\n{obj_name}:")
            print(f"  Position (x, y, z): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
            print(f"  Full pose matrix (base frame):")
            print(f"{pose_base}")
        print(f"\n{'='*80}\n")
        exit(0)

    def _register_visual_references(self, objects: set):
        """
        Pre-register visual reference images to YOLOE visual server.

        Loads reference images from yoloe_visual_ref_dir and registers them
        with bounding boxes from bboxes.json.

        Args:
            objects: Set of object names to register
        """
        import cv2

        visual_ref_dir = self.robot_config.get("yoloe_visual_ref_dir", "")
        if not visual_ref_dir:
            print("⚠️  yoloe_visual_ref_dir not configured, skipping visual ref registration")
            return

        # Resolve path relative to pipeline root
        if not os.path.isabs(visual_ref_dir):
            visual_ref_dir = str(Path(__file__).parent.parent / visual_ref_dir)

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
        visual_server_config = self.robot_config.get("yoloe_visual_server", {})
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

    def _register_wrist_visual_references(self):
        """
        Register wrist-view visual references for distractor detection.

        Loads reference images from pipeline/data/yoloe_ref_images/ (wrist view)
        and registers them to YOLOE visual server using self.yoloe_client.

        Called when is_visual_ref_yoloe=True and is_randome_erasing=True.
        """
        import cv2

        # Use default wrist ref directory
        wrist_ref_dir = Path(__file__).parent / "data" / "yoloe_ref_images"
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

        # Get all scene objects from current task
        task_info = None
        for task in self.task_config.get("long_horizon_tasks", []):
            if task["name"] == self.task_name:
                task_info = task
                break

        if task_info is None:
            print(f"⚠️  Task not found, skipping wrist ref registration")
            return

        scene_objects = set(task_info.get("scene_objects", []))
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

                response = self.yoloe_client.register(
                    object_name=ref_name,
                    image=img_rgb,
                    bbox=bbox
                )

                if response.get('success', False):
                    print(f"  ✓ Registered '{ref_name}'")
                else:
                    print(f"  ✗ Failed: '{ref_name}': {response.get('error', 'unknown')}")

        print(f"✓ Wrist visual reference registration complete\n")

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

        # Check if using visual reference mode
        use_visual_ref = self.robot_config.get("is_visual_ref_yoloe", False)
        if use_visual_ref:
            print(f"🎯 Visual Reference Mode ENABLED")
            # Pre-register 3rd-view visual references for FoundationPose
            self._register_visual_references(unique_objects)
            # Pre-register wrist-view visual references for distractor detection
            if self.yoloe_client is not None:
                self._register_wrist_visual_references()
        else:
            print(f"📝 Text Prompt Mode (default)")

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
            if use_visual_ref:
                print(f"  Mode: Visual Reference")
                print(f"  Visual ref object: '{object_name}'")
            else:
                print(f"  Mode: Text Prompt")
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

    def _get_distractor_objects(self, skill_info: Dict) -> List[str]:
        """
        Get list of distractor objects for random erasing.

        Distractors are objects in scene_objects that are NOT:
        - The current skill's target_object
        - The current skill's grasp_object (if any)

        Args:
            skill_info: Skill information from task config

        Returns:
            List of distractor object names
        """
        # Get scene_objects from current task
        task_info = None
        for task in self.task_config.get("long_horizon_tasks", []):
            if task["name"] == self.task_name:
                task_info = task
                break

        if task_info is None:
            return []

        scene_objects = set(task_info.get("scene_objects", []))

        # Remove target_object and grasp_object
        exclude = set()
        if "target_object" in skill_info:
            exclude.add(skill_info["target_object"])
        if "grasp_object" in skill_info:
            exclude.add(skill_info["grasp_object"])

        distractors = list(scene_objects - exclude)
        return distractors

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
            7. Retry on failure with recovery logic:
               - Pick failure: re-obtain pose, recalculate, MP, VLA
               - Place failure: re-do pick first, then place

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

        # Track previous pick skill for place recovery
        previous_pick_skill_info = None

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

            # First attempt: use normal execution
            result = self._execute_single_skill(
                skill_info=skill_info,
                skill_idx=skill_idx - 1,  # Convert to 0-indexed
                initial_states=initial_states,
                csv_logger=csv_logger
            )

            if result["success"]:
                skill_success = True
                if "final_object_poses" in result:
                    initial_states.update(result["final_object_poses"])
            else:
                print(f"❌ Skill failed: {result['reason']}")

                # Retry with recovery logic
                for retry in range(1, max_retries):
                    print(f"\n🔄 Recovery Retry {retry}/{max_retries-1}")
                    time.sleep(2.0)

                    if skill_type == "pick":
                        # Pick failure: re-obtain pose, recalculate above, MP, VLA
                        result = self._retry_pick_skill(
                            skill_info=skill_info,
                            initial_states=initial_states,
                            csv_logger=csv_logger
                        )
                    elif skill_type == "place":
                        # Place failure: re-do pick first, then place
                        if previous_pick_skill_info is None:
                            print(f"❌ No previous pick skill found for place recovery")
                            result = {"success": False, "reason": "No previous pick skill"}
                        else:
                            result = self._retry_place_skill(
                                place_skill_info=skill_info,
                                pick_skill_info=previous_pick_skill_info,
                                initial_states=initial_states,
                                csv_logger=csv_logger
                            )
                    else:
                        # Unknown skill type: just retry normal execution
                        result = self._execute_single_skill(
                            skill_info=skill_info,
                            skill_idx=skill_idx - 1,
                            initial_states=initial_states,
                            csv_logger=csv_logger
                        )

                    if result["success"]:
                        skill_success = True
                        if "final_object_poses" in result:
                            initial_states.update(result["final_object_poses"])
                        break
                    else:
                        print(f"❌ Recovery failed: {result['reason']}")

            # Track pick skill for potential place recovery
            if skill_type == "pick":
                previous_pick_skill_info = skill_info

            if skill_success:
                success_count += 1
                print(f"\n✅ Skill {skill_idx}/{total_skills} completed successfully")
            else:
                print(f"\n❌ Skill {skill_idx}/{total_skills} failed after {max_retries} attempts")
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
        target_obj_pose = initial_states[target_object]
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

        # Relative pose state tracking (for is_relative_pose)
        is_relative_pose = self.robot_config.get("is_relative_pose", False)
        is_pick_skill = skill_info.get("skill_type") == "pick"
        target_object = skill_info.get("target_object")
        object_pos = None
        if is_relative_pose and target_object and target_object in self.registration_dict:
            pose_base = self.registration_dict[target_object]["pose_base"]
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

            # Detect distractor objects for random erasing (if enabled)
            masked_wrist_image = None
            if self.yoloe_client is not None:
                distractors = self._get_distractor_objects(skill_info)
                if distractors:
                    wrist_image = self._get_camera_data(obs, 'wrist', 'image')
                    # Convert BGR to RGB for YOLOE
                    wrist_image_rgb = wrist_image[..., ::-1] if wrist_image is not None else None
                    if wrist_image_rgb is not None:
                        distractor_mask = self.yoloe_client.detect_and_union(
                            wrist_image_rgb,
                            distractors,
                            text_prompts=self.yoloe_text_prompts,  # For text mode
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
            vla_obs = self._prepare_vla_observation(
                obs,
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

            # Log actions and state
            if csv_logger is not None:
                csv_logger.log_step(t_step, action, obs["robot_state"])

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

    def _prepare_vla_observation(
        self,
        obs: Dict,
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
        left_camera_id = self.robot_config["cameras"]["left_camera_id"]
        wrist_camera_id = self.robot_config["cameras"]["wrist_camera_id"]

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
        is_relative_pose = self.robot_config.get("is_relative_pose", False)

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

    def _retry_pick_skill(
        self,
        skill_info: Dict,
        initial_states: Dict,
        csv_logger: Optional[CSVLogger]
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
            csv_logger: CSV logger instance

        Returns:
            Dict with "success", "reason", "final_object_poses"
        """
        target_object = skill_info["target_object"]
        print(f"\n{'='*60}")
        print(f"🔄 RETRY PICK: Re-obtaining pose for '{target_object}'")
        print(f"{'='*60}\n")

        # Load camera intrinsics
        K_config = self.robot_config["camera_intrinsics"]
        K = np.array([
            [K_config["fx"], 0, K_config["cx"]],
            [0, K_config["fy"], K_config["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 1: Re-obtain object pose
        print(f"1️⃣  Re-obtaining object pose...")
        obs = self.robot_env.get_observation()
        pose_result = self.object_pose_client.get_pose(
            object_name=target_object,
            rgb=self._get_camera_data(obs, 'left', 'image'),
            depth=self._get_camera_data(obs, 'left', 'depth'),
            K=K,
            iteration=self.robot_config["tracking"]["tracking_iteration"]
        )

        if pose_result is None or not pose_result["success"]:
            return {"success": False, "reason": f"Failed to re-obtain pose for {target_object}"}

        pose_cam = np.array(pose_result["pose"])
        object_pose_matrix = self._transform_pose_camera_to_base(pose_cam, camera_role='left')
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
        reg_info = self.registration_dict[target_object]
        mesh = trimesh.load(reg_info["mesh_path"])
        mesh_vertices = np.array(mesh.vertices)

        above_pose = self.target_pose_calculator.calculate_above_pose(
            object_pose_dict, target_object, mesh_vertices=mesh_vertices
        )
        # Set gripper open for pick approach
        above_pose["gripper"] = 0.0

        # Step 3: MP to above pose (gripper open)
        print(f"\n3️⃣  Moving to above pose (gripper open)...")
        if self.motion_planner is not None:
            mp_result = self.motion_planner.move_to_pose(above_pose, method="linear")
            if not mp_result["success"]:
                return {"success": False, "reason": f"Failed to reach above pose"}
        print()

        # Step 4: VLA execute pick
        print(f"4️⃣  Executing VLA pick skill...")
        vla_result = self._execute_vla_skill(
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
        if self.motion_planner is not None:
            # Get current pose and set gripper closed
            current_pose = self.motion_planner.get_current_ee_pose()
            lift_pose = {
                "position": current_pose["position"] + np.array([0, 0, 0.05]),
                "orientation_euler": current_pose["orientation_euler"],
                "gripper": 1.0  # Keep gripper closed after pick
            }
            self.motion_planner.move_to_pose(lift_pose, method="linear")
        print()

        # Step 6: Get final pose and check success
        print(f"6️⃣  Checking pick success...")
        obs_final = self.robot_env.get_observation()
        final_pose_result = self.object_pose_client.get_pose(
            object_name=target_object,
            rgb=self._get_camera_data(obs_final, 'left', 'image'),
            depth=self._get_camera_data(obs_final, 'left', 'depth'),
            K=K,
            iteration=self.robot_config["tracking"]["tracking_iteration"]
        )

        final_object_poses = {}
        if final_pose_result and final_pose_result["success"]:
            pose_cam = np.array(final_pose_result["pose"])
            final_pose_matrix = self._transform_pose_camera_to_base(pose_cam, camera_role='left')
            final_object_poses[target_object] = {
                "position": final_pose_matrix[:3, 3],
                "pose_matrix": final_pose_matrix
            }

        # Check success
        mesh_paths = {name: info["mesh_path"] for name, info in self.registration_dict.items()}
        success_result = self.success_checker.check_success(
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

    def _retry_place_skill(
        self,
        place_skill_info: Dict,
        pick_skill_info: Dict,
        initial_states: Dict,
        csv_logger: Optional[CSVLogger]
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
        pick_result = self._retry_pick_skill(
            skill_info=pick_skill_info,
            initial_states=initial_states,
            csv_logger=csv_logger
        )

        if not pick_result["success"]:
            return {"success": False, "reason": f"Re-pick failed: {pick_result['reason']}"}

        # Update initial_states with pick result
        if "final_object_poses" in pick_result:
            initial_states.update(pick_result["final_object_poses"])

        print(f"\n✅ Re-pick successful, now proceeding to PLACE...")

        # Load camera intrinsics
        K_config = self.robot_config["camera_intrinsics"]
        K = np.array([
            [K_config["fx"], 0, K_config["cx"]],
            [0, K_config["fy"], K_config["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Step 2: Re-obtain target object pose for place
        print(f"\n2️⃣  Re-obtaining pose for place target '{target_object}'...")
        obs = self.robot_env.get_observation()
        pose_result = self.object_pose_client.get_pose(
            object_name=target_object,
            rgb=self._get_camera_data(obs, 'left', 'image'),
            depth=self._get_camera_data(obs, 'left', 'depth'),
            K=K,
            iteration=self.robot_config["tracking"]["tracking_iteration"]
        )

        if pose_result is None or not pose_result["success"]:
            return {"success": False, "reason": f"Failed to re-obtain pose for {target_object}"}

        pose_cam = np.array(pose_result["pose"])
        object_pose_matrix = self._transform_pose_camera_to_base(pose_cam, camera_role='left')
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
        reg_info = self.registration_dict[target_object]
        mesh = trimesh.load(reg_info["mesh_path"])
        mesh_vertices = np.array(mesh.vertices)

        above_pose = self.target_pose_calculator.calculate_above_pose(
            object_pose_dict, target_object, mesh_vertices=mesh_vertices
        )
        # Set gripper closed for place approach (holding object)
        above_pose["gripper"] = 1.0

        # Step 4: MP to above pose (gripper closed)
        print(f"\n4️⃣  Moving to above pose (gripper closed, holding object)...")
        if self.motion_planner is not None:
            mp_result = self.motion_planner.move_to_pose(above_pose, method="linear")
            if not mp_result["success"]:
                return {"success": False, "reason": f"Failed to reach above pose for place"}
        print()

        # Step 5: VLA execute place
        print(f"5️⃣  Executing VLA place skill...")
        vla_result = self._execute_vla_skill(
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
        if self.motion_planner is not None:
            current_pose = self.motion_planner.get_current_ee_pose()
            lift_pose = {
                "position": current_pose["position"] + np.array([0, 0, 0.05]),
                "orientation_euler": current_pose["orientation_euler"],
                "gripper": 0.0  # Open gripper after place
            }
            self.motion_planner.move_to_pose(lift_pose, method="linear")
        print()

        # Step 7: Get final poses and check success
        print(f"7️⃣  Checking place success...")
        obs_final = self.robot_env.get_observation()
        final_object_poses = {}

        # Get final poses for both objects
        for obj_name in [grasp_object, target_object]:
            if obj_name is None:
                continue
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

        # Check success
        mesh_paths = {name: info["mesh_path"] for name, info in self.registration_dict.items()}
        success_result = self.success_checker.check_success(
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


def apply_distractor_rectangle_masking(
    image: np.ndarray,
    distractor_mask: np.ndarray,
    max_rectangles: int = 5
) -> np.ndarray:
    """
    Apply rectangle masking to distractor regions in image.

    Finds connected components in distractor_mask, sorts by area (largest first),
    and masks up to max_rectangles components with their bounding boxes.

    Args:
        image: Input image (H, W, 3) uint8
        distractor_mask: Binary mask of distractor regions (H, W) uint8
        max_rectangles: Maximum number of rectangles to mask (default 5)

    Returns:
        Masked image with rectangular black regions over distractors
    """
    if distractor_mask is None or distractor_mask.sum() == 0:
        return image

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        distractor_mask, connectivity=8
    )

    if num_labels <= 1:  # Only background
        return image

    # Get component areas (skip background label 0)
    # stats columns: [x, y, width, height, area]
    component_info = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        component_info.append({
            'label': label_id,
            'area': area,
            'bbox': (x, y, x + w, y + h)
        })

    # Sort by area (largest first)
    component_info.sort(key=lambda c: c['area'], reverse=True)

    # Take top N components
    selected = component_info[:max_rectangles]

    # Apply rectangle masking
    masked_image = image.copy()
    for comp in selected:
        x1, y1, x2, y2 = comp['bbox']
        masked_image[y1:y2, x1:x2] = 0

    return masked_image


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
