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
import trimesh

# ========== [1] IMPORTS ==========

# Add paths
pipeline_dir = Path(__file__).parent
sys.path.append(str(pipeline_dir))
sys.path.append(str(pipeline_dir / "../droid"))

# Import core modules
from core import (
    # Original modules
    ObjectPoseClient,
    TargetPoseCalculator,
    MotionPlanner,
    SuccessChecker,
    VLAClient,
    YOLOEObjectDetectorClient,
    # Camera utilities
    get_camera_data,
    transform_pose_camera_to_base,
    # Object registration
    run_registration_phase,
    # VLA execution
    execute_vla_skill,
    # Retry handlers
    retry_pick_skill,
    retry_place_skill,
    # Skill helpers
    get_tracked_objects_for_skill,
    get_distractor_objects,
    should_keep_tracking,
)

# Import utils
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


# ========== [2] PIPELINE CLASS ==========

class RealRobotPipeline:
    """
    Main pipeline for real robot long-horizon task execution.

    Orchestrates all components:
    - Object pose tracking (FoundationPose)
    - Target pose calculation
    - Motion planning
    - VLA skill execution
    - Success checking
    - Logging and video recording
    """

    # ========== [2.1] INITIALIZATION ==========

    def __init__(
        self,
        task_name: str,
        task_config_path: str = "config/task_config.json",
        robot_config_path: str = "config/real_robot_config.json"
    ):
        """
        Initialize pipeline.

        Args:
            task_name: Long-horizon task name (e.g., "pick the red cup")
            task_config_path: Path to task configuration file
            robot_config_path: Path to robot configuration file
        """
        print(f"\n{'='*80}")
        print(f"🤖 REAL ROBOT PIPELINE INITIALIZATION")
        print(f"{'='*80}\n")

        self.task_name = task_name
        self.pipeline_root = Path(__file__).parent

        # ========== [2.1.1] Load configs ==========
        print(f"📁 Loading configurations...")
        self.task_config = load_task_config(task_config_path)
        with open(robot_config_path, 'r') as f:
            self.robot_config = json.load(f)
        print(f"   ✓ Task config: {task_config_path}")
        print(f"   ✓ Robot config: {robot_config_path}\n")

        # ========== [2.1.2] Plan task sequence ==========
        print(f"📋 Planning task sequence...")
        self.skill_sequence = plan_task_sequence(task_name, self.task_config)
        print(f"   Task: '{task_name}'")
        print(f"   Skills: {self.skill_sequence}\n")

        # ========== [2.1.3] Create output directory ==========
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir_name = task_name.replace(" ", "_")
        self.run_dir = Path(self.robot_config["logging"]["output_dir"]) / f"{task_dir_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Output directory: {self.run_dir}\n")

        # ========== [2.1.4] Initialize state tracking ==========
        self.registration_dict = {}
        self.tracked_objects = set()

        # ========== [2.1.5] Initialize YOLOE client (for distractor masking) ==========
        self.yoloe_client = None
        self.yoloe_text_prompts = None
        if self.robot_config.get("is_randome_erasing", False):
            use_visual_ref = self.robot_config.get("is_visual_ref_yoloe", False)
            if use_visual_ref:
                self.yoloe_client = YOLOEObjectDetectorClient(
                    host="localhost", port=5559, mode="visual"
                )
            else:
                self.yoloe_client = YOLOEObjectDetectorClient(
                    host="localhost", port=5557, mode="text"
                )
                self.yoloe_text_prompts = self.robot_config.get("yoloe_prompts", {})
            print(f"🎭 YOLOE distractor masking: {'visual' if use_visual_ref else 'text'} mode\n")

        # ========== [2.1.6] Initialize modules ==========
        self._initialize_modules()

    def _initialize_modules(self):
        """Initialize all pipeline modules and run registration."""
        print(f"{'='*60}")
        print(f"🔧 INITIALIZING MODULES")
        print(f"{'='*60}\n")

        # ========== [2.1.6.1] Robot environment ==========
        print("1️⃣  Initializing robot environment...")
        self.robot_env = RobotEnv()
        print("   ✓ RobotEnv initialized\n")

        # ========== [2.1.6.2] FoundationPose client ==========
        print("2️⃣  Connecting to FoundationPose server...")
        fp_config = self.robot_config["foundation_pose_server"]
        self.object_pose_client = ObjectPoseClient(
            server_url=fp_config["host"],
            port=fp_config["port"],
            timeout=fp_config["timeout"]
        )
        print()

        # ========== [2.1.6.3] Target pose calculator ==========
        print("3️⃣  Initializing target pose calculator...")
        self.target_pose_calculator = TargetPoseCalculator(self.robot_config)
        print("   ✓ TargetPoseCalculator ready\n")

        # ========== [2.1.6.4] Motion planner ==========
        print("4️⃣  Initializing motion planner...")
        self.motion_planner = MotionPlanner(self.robot_env, self.robot_config)
        print()

        # ========== [2.1.6.5] VLA client ==========
        print("5️⃣  Connecting to VLA server...")
        vla_config = self.robot_config["vla_server"]
        self.vla_client = VLAClient(
            host=vla_config["host"],
            port=vla_config["port"]
        )
        print()

        # ========== [2.1.6.6] Success checker ==========
        print("6️⃣  Initializing success checker...")
        self.success_checker = SuccessChecker(self.robot_config)
        print("   ✓ SuccessChecker ready\n")

        # ========== [2.1.6.7] Run registration phase ==========
        print("7️⃣  Running object registration phase...")
        self.registration_dict = run_registration_phase(
            robot_env=self.robot_env,
            pose_client=self.object_pose_client,
            robot_config=self.robot_config,
            task_config=self.task_config,
            skill_sequence=self.skill_sequence,
            yoloe_client=self.yoloe_client,
            pipeline_root=self.pipeline_root,
            get_skill_info_func=get_skill_info
        )

        # Print registered object poses
        print(f"\n{'='*80}")
        print(f"📍 REGISTERED OBJECT POSES (BASE FRAME)")
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

    # ========== [2.2] MAIN EXECUTION ==========

    def execute_task(self, max_retries: int = 3) -> bool:
        """
        Execute long-horizon task.

        Main execution loop:
        For each skill:
            1. Get object pose
            2. Calculate above pose
            3. Move to above pose (motion planner)
            4. Execute VLA skill
            5. Move EE up
            6. Check success
            7. Retry on failure with recovery logic

        Args:
            max_retries: Maximum retries per skill

        Returns:
            True if all skills succeeded
        """
        print(f"{'='*80}")
        print(f"🚀 STARTING TASK EXECUTION")
        print(f"{'='*80}\n")

        # ========== [2.2.1] Initialize logging ==========
        if self.robot_config["logging"]["save_csv"]:
            csv_logger = CSVLogger(self.run_dir, prefix=f"task_{self.task_name.replace(' ', '_')}")
        else:
            csv_logger = None

        # Track initial object states
        initial_states = {}
        previous_pick_skill_info = None
        success_count = 0
        total_skills = len(self.skill_sequence)

        # ========== [2.2.2] Execute each skill ==========
        for skill_idx, skill_name in enumerate(self.skill_sequence, 1):
            print(f"\n{'='*80}")
            print(f"📍 SKILL {skill_idx}/{total_skills}: {skill_name}")
            print(f"{'='*80}\n")

            skill_info = get_skill_info(skill_name, self.task_config)
            if skill_info is None:
                raise ValueError(f"Skill info not found for skill: {skill_name}")

            skill_type = skill_info["skill_type"]
            skill_success = False

            # ========== [2.2.2.1] First attempt ==========
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
            else:
                print(f"❌ Skill failed: {result['reason']}")

                # ========== [2.2.2.2] Retry with recovery logic ==========
                for retry in range(1, max_retries):
                    print(f"\n🔄 Recovery Retry {retry}/{max_retries-1}")
                    time.sleep(2.0)

                    if skill_type == "pick":
                        result = self._retry_pick_skill_wrapper(
                            skill_info, initial_states, csv_logger
                        )
                    elif skill_type == "place":
                        if previous_pick_skill_info is None:
                            result = {"success": False, "reason": "No previous pick skill"}
                        else:
                            result = self._retry_place_skill_wrapper(
                                skill_info, previous_pick_skill_info, initial_states, csv_logger
                            )
                    else:
                        result = self._execute_single_skill(
                            skill_info, skill_idx - 1, initial_states, csv_logger
                        )

                    if result["success"]:
                        skill_success = True
                        if "final_object_poses" in result:
                            initial_states.update(result["final_object_poses"])
                        break
                    else:
                        print(f"❌ Recovery failed: {result['reason']}")

            # Track pick skill for place recovery
            if skill_type == "pick":
                previous_pick_skill_info = skill_info

            if skill_success:
                success_count += 1
                print(f"\n✅ Skill {skill_idx}/{total_skills} completed successfully")
            else:
                print(f"\n❌ Skill {skill_idx}/{total_skills} failed after {max_retries} attempts")
                break

        # ========== [2.2.3] Cleanup and summary ==========
        if csv_logger is not None:
            csv_logger.close()

        print(f"\n{'='*80}")
        print(f"📊 TASK EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Task: {self.task_name}")
        print(f"Skills completed: {success_count}/{total_skills}")
        print(f"Success rate: {success_count/total_skills*100:.1f}%")
        print(f"Output directory: {self.run_dir}")
        print(f"{'='*80}\n")

        return success_count == total_skills

    # ========== [2.3] SINGLE SKILL EXECUTION ==========

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
        target_object = skill_info["target_object"]
        language = skill_info["language"]
        skill_type = skill_info["skill_type"]

        # Get objects to track
        objects_to_track = get_tracked_objects_for_skill(skill_info)

        # Load camera intrinsics
        K_config = self.robot_config["camera_intrinsics"]
        K = np.array([
            [K_config["fx"], 0, K_config["cx"]],
            [0, K_config["fy"], K_config["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        # ========== [2.3.1] Start tracking ==========
        print(f"1️⃣  Starting tracking for {len(objects_to_track)} object(s): {objects_to_track}...")
        for obj_name in objects_to_track:
            if obj_name in self.tracked_objects:
                print(f"   ⏩ {obj_name} already tracking")
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

        # ========== [2.3.2] Collect initial poses ==========
        print(f"2️⃣  Collecting initial poses...")
        obs = self.robot_env.get_observation()
        for obj_name in objects_to_track:
            if obj_name in initial_states:
                continue

            pose_result = self.object_pose_client.get_pose(
                object_name=obj_name,
                rgb=get_camera_data(obs, 'left', self.robot_config, 'image'),
                depth=get_camera_data(obs, 'left', self.robot_config, 'depth'),
                K=K,
                iteration=self.robot_config["tracking"]["tracking_iteration"]
            )

            if pose_result is None or not pose_result["success"]:
                return {"success": False, "reason": f"Failed to get initial pose for {obj_name}"}

            pose_cam = np.array(pose_result["pose"])
            pose_matrix = transform_pose_camera_to_base(pose_cam, 'left', self.robot_config)
            position = pose_matrix[:3, 3]
            initial_states[obj_name] = {"position": position, "pose_matrix": pose_matrix}
            print(f"   ✅ {obj_name}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        print()

        # ========== [2.3.3] Calculate above pose ==========
        print(f"3️⃣  Calculating above pose...")
        target_obj_pose = initial_states[target_object]
        reg_info = self.registration_dict[target_object]
        mesh = trimesh.load(reg_info["mesh_path"])

        above_pose = self.target_pose_calculator.calculate_above_pose(
            {"position": target_obj_pose["position"]},
            target_object,
            mesh_vertices=np.array(mesh.vertices)
        )
        print()

        # ========== [2.3.4] Move to above pose ==========
        if self.motion_planner is not None:
            print(f"4️⃣  Moving to above pose...")
            mp_result = self.motion_planner.move_to_pose(above_pose, method="linear")
            if not mp_result["success"]:
                return {"success": False, "reason": f"Failed to reach above pose"}
            print()

        # ========== [2.3.5] Execute VLA skill ==========
        print(f"5️⃣  Executing VLA skill: '{language}'...")
        vla_result = execute_vla_skill(
            skill_info=skill_info,
            initial_states=initial_states,
            K=K,
            robot_env=self.robot_env,
            vla_client=self.vla_client,
            pose_client=self.object_pose_client,
            success_checker=self.success_checker,
            robot_config=self.robot_config,
            registration_dict=self.registration_dict,
            yoloe_client=self.yoloe_client,
            yoloe_text_prompts=self.yoloe_text_prompts,
            csv_logger=csv_logger,
            get_tracked_objects_func=get_tracked_objects_for_skill,
            get_distractor_objects_func=lambda si: get_distractor_objects(si, self.task_config, self.task_name)
        )

        if not vla_result["success"]:
            self._cleanup_tracking(skill_idx)
            return {"success": False, "reason": f"VLA execution failed: {vla_result['reason']}"}
        print()

        # ========== [2.3.6] Move EE up ==========
        if self.motion_planner is not None:
            print(f"6️⃣  Moving EE up...")
            self.motion_planner.move_ee_up(lift_distance=0.05, skill_type=skill_type)
            print()

        # ========== [2.3.7] Get final poses ==========
        print(f"7️⃣  Getting final poses...")
        obs_final = self.robot_env.get_observation()
        final_object_poses = {}

        for obj_name in objects_to_track:
            final_pose_result = self.object_pose_client.get_pose(
                object_name=obj_name,
                rgb=get_camera_data(obs_final, 'left', self.robot_config, 'image'),
                depth=get_camera_data(obs_final, 'left', self.robot_config, 'depth'),
                K=K,
                iteration=self.robot_config["tracking"]["tracking_iteration"]
            )

            if final_pose_result and final_pose_result["success"]:
                pose_cam = np.array(final_pose_result["pose"])
                pose_matrix = transform_pose_camera_to_base(pose_cam, 'left', self.robot_config)
                final_object_poses[obj_name] = {
                    "position": pose_matrix[:3, 3],
                    "pose_matrix": pose_matrix
                }
        print()

        # ========== [2.3.8] Cleanup tracking ==========
        self._cleanup_tracking(skill_idx)

        # ========== [2.3.9] Check success ==========
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

    # ========== [2.4] HELPER METHODS ==========

    def _cleanup_tracking(self, skill_idx: int):
        """End tracking for objects not needed in next skill."""
        print(f"8️⃣  Managing tracking sessions...")
        for obj_name in list(self.tracked_objects):
            if not should_keep_tracking(
                obj_name, skill_idx, self.skill_sequence,
                self.task_config, get_skill_info
            ):
                self.object_pose_client.end_tracking(object_name=obj_name)
                self.tracked_objects.remove(obj_name)
                print(f"   ⏹  Ended tracking: {obj_name}")
            else:
                print(f"   ⏩ Keeping tracking: {obj_name}")
        print()

    def _retry_pick_skill_wrapper(
        self,
        skill_info: Dict,
        initial_states: Dict,
        csv_logger: Optional[CSVLogger]
    ) -> Dict:
        """Wrapper for retry_pick_skill with pipeline context."""
        K_config = self.robot_config["camera_intrinsics"]
        K = np.array([
            [K_config["fx"], 0, K_config["cx"]],
            [0, K_config["fy"], K_config["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        return retry_pick_skill(
            skill_info=skill_info,
            initial_states=initial_states,
            robot_env=self.robot_env,
            pose_client=self.object_pose_client,
            motion_planner=self.motion_planner,
            target_pose_calculator=self.target_pose_calculator,
            success_checker=self.success_checker,
            robot_config=self.robot_config,
            registration_dict=self.registration_dict,
            execute_vla_skill_func=lambda si, ist, k, csv: execute_vla_skill(
                skill_info=si,
                initial_states=ist,
                K=k,
                robot_env=self.robot_env,
                vla_client=self.vla_client,
                pose_client=self.object_pose_client,
                success_checker=self.success_checker,
                robot_config=self.robot_config,
                registration_dict=self.registration_dict,
                yoloe_client=self.yoloe_client,
                yoloe_text_prompts=self.yoloe_text_prompts,
                csv_logger=csv,
                get_tracked_objects_func=get_tracked_objects_for_skill,
                get_distractor_objects_func=lambda s: get_distractor_objects(s, self.task_config, self.task_name)
            ),
            csv_logger=csv_logger
        )

    def _retry_place_skill_wrapper(
        self,
        place_skill_info: Dict,
        pick_skill_info: Dict,
        initial_states: Dict,
        csv_logger: Optional[CSVLogger]
    ) -> Dict:
        """Wrapper for retry_place_skill with pipeline context."""
        return retry_place_skill(
            place_skill_info=place_skill_info,
            pick_skill_info=pick_skill_info,
            initial_states=initial_states,
            robot_env=self.robot_env,
            pose_client=self.object_pose_client,
            motion_planner=self.motion_planner,
            target_pose_calculator=self.target_pose_calculator,
            success_checker=self.success_checker,
            robot_config=self.robot_config,
            registration_dict=self.registration_dict,
            execute_vla_skill_func=lambda si, ist, k, csv: execute_vla_skill(
                skill_info=si,
                initial_states=ist,
                K=k,
                robot_env=self.robot_env,
                vla_client=self.vla_client,
                pose_client=self.object_pose_client,
                success_checker=self.success_checker,
                robot_config=self.robot_config,
                registration_dict=self.registration_dict,
                yoloe_client=self.yoloe_client,
                yoloe_text_prompts=self.yoloe_text_prompts,
                csv_logger=csv,
                get_tracked_objects_func=get_tracked_objects_for_skill,
                get_distractor_objects_func=lambda s: get_distractor_objects(s, self.task_config, self.task_name)
            ),
            csv_logger=csv_logger
        )


# ========== [3] MAIN ENTRY POINT ==========

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
        default=3,
        help="Maximum retries per skill"
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = RealRobotPipeline(
        task_name=args.task_name,
        task_config_path=args.task_config,
        robot_config_path=args.robot_config
    )

    success = pipeline.execute_task(max_retries=args.max_retries)

    if success:
        print("🎉 Task completed successfully!")
        return 0
    else:
        print("❌ Task failed")
        return 1


if __name__ == "__main__":
    exit(main())
