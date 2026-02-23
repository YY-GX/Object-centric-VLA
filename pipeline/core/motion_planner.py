#!/usr/bin/env python3
"""
Motion Planner - Execute robot motions to target poses.

This module wraps the DROID RobotEnv and provides higher-level motion planning
primitives for moving to poses, lifting the EE, and monitoring execution.

Supports multiple motion planning methods:
- IK_MP: Direct IK-based reaching (blocking)
- CLI_MP: Cartesian Linear Interpolation with closed-loop control
"""

import time
import numpy as np
from typing import Dict, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from scipy.spatial.transform import Rotation as R, Slerp
import sys
import os

# Add path to robot scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts'))

try:
    from robot_env import RobotEnv
except ImportError:
    print("⚠️  Warning: Could not import RobotEnv. Motion planner may not work on this machine.")
    RobotEnv = None


def compute_orientation_error(target_euler: np.ndarray, current_euler: np.ndarray) -> float:
    """
    Compute orientation error handling euler angle wrapping at ±π.

    Args:
        target_euler: Target euler angles [roll, pitch, yaw]
        current_euler: Current euler angles [roll, pitch, yaw]

    Returns:
        L2 norm of wrapped angle differences
    """
    diff = target_euler - current_euler
    # Wrap each angle difference to [-π, π]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.linalg.norm(diff)


class MotionPlanner:
    """
    High-level motion planning interface for real robot.

    Wraps RobotEnv and provides:
    - Multiple motion planning methods (IK, linear interpolation, etc.)
    - Vertical lifting motions
    - Success/failure detection
    """

    def __init__(self, robot_env, config: Dict):
        """
        Initialize motion planner.

        Args:
            robot_env: DROID RobotEnv instance
            config: Motion planning config with:
                - position_threshold: Success threshold (meters)
                - orientation_threshold: Success threshold (radians)
                - max_retries: Maximum retry attempts
        """
        self.robot_env = robot_env
        self.position_threshold = config.get("position_threshold", 0.02)
        self.orientation_threshold = config.get("orientation_threshold", 0.524)
        self.max_retries = config.get("max_retries", 3)

        print(f"🤖 MotionPlanner initialized:")
        print(f"   Position threshold: {self.position_threshold}m")
        print(f"   Orientation threshold: {self.orientation_threshold} rad")
        print(f"   Max retries: {self.max_retries}")

    def move_to_pose(
        self,
        target_pose: Dict,
        method: Literal["ik", "linear"] = "ik",
        **kwargs
    ) -> Dict:
        """
        Execute motion to target Cartesian pose using specified method.

        Args:
            target_pose: Target EE pose with:
                - "position": [x, y, z] (meters)
                - "orientation_euler": [roll, pitch, yaw] (radians)
                - "gripper": float 0-1 (optional, default: current)
            method: Motion planning method:
                - "ik": Direct IK-based reaching (blocking)
                - "linear": Linear Cartesian interpolation (future)
            **kwargs: Method-specific arguments

        Returns:
            Dict with:
                - success: bool
                - final_distance: float (meters)
                - final_orientation_error: float (radians)
                - execution_time: float (seconds)
        """
        if method == "ik":
            return self.IK_MP(target_pose, **kwargs)
        elif method == "linear":
            return self.CLI_MP(target_pose, **kwargs)
        else:
            raise ValueError(f"Unknown motion planning method: {method}")

    def IK_MP(
        self,
        target_pose: Dict,
        max_retries: Optional[int] = None
    ) -> Dict:
        """
        Direct IK-based motion planning using DROID's blocking IK solver.

        Uses robot.update_command() with cartesian_position + blocking=True.
        This internally:
        1. Solves IK for target pose
        2. Executes smooth joint-space motion
        3. Returns when motion completes

        Args:
            target_pose: Target pose dict with:
                - "position": [x, y, z]
                - "orientation_euler": [roll, pitch, yaw]
                - "gripper": float 0-1 (optional)
            max_retries: Number of retry attempts (currently unused)

        Returns:
            Dict with success, final_distance, execution_time
        """
        target_pos = np.array(target_pose["position"])
        target_euler = np.array(target_pose["orientation_euler"])
        target_gripper = target_pose.get("gripper", None)

        print(f"🎯 IK Motion Planning to:")
        print(f"   Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"   Orientation (euler): [{target_euler[0]:.3f}, {target_euler[1]:.3f}, {target_euler[2]:.3f}]")

        start_time = time.time()

        # Get current gripper state if not specified
        if target_gripper is None:
            current_pose = self.get_current_ee_pose()
            target_gripper = current_pose["gripper"]

        # Construct action: [x, y, z, roll, pitch, yaw, gripper]
        action = np.concatenate([target_pos, target_euler, [target_gripper]])

        # Execute blocking motion using DROID's IK solver
        try:
            action_info = self.robot_env.update_robot(
                action,
                action_space="cartesian_position",
                gripper_action_space="position",
                blocking=True
            )
        except Exception as e:
            print(f"❌ IK motion failed: {e}")
            return {
                "success": False,
                "final_distance": float('inf'),
                "final_orientation_error": float('inf'),
                "execution_time": time.time() - start_time,
                "error": str(e)
            }

        execution_time = time.time() - start_time

        # Check final convergence
        final_pose = self.get_current_ee_pose()
        final_distance = np.linalg.norm(target_pos - final_pose["position"])

        # Compute orientation error (with angle wrapping)
        orientation_error = compute_orientation_error(target_euler, final_pose["orientation_euler"])

        success = (final_distance < self.position_threshold and
                   orientation_error < self.orientation_threshold)

        if success:
            print(f"✅ Reached target pose ({execution_time:.2f}s)")
            print(f"   Final distance: {final_distance:.4f}m")
            print(f"   Orientation error: {orientation_error:.4f} rad")
        else:
            print(f"⚠️  Motion completed but not converged:")
            print(f"   Final distance: {final_distance:.4f}m (threshold: {self.position_threshold}m)")
            print(f"   Orientation error: {orientation_error:.4f} rad (threshold: {self.orientation_threshold} rad)")

        return {
            "success": success,
            "final_distance": float(final_distance),
            "final_orientation_error": float(orientation_error),
            "execution_time": execution_time
        }

    def CLI_MP(
        self,
        target_pose: Dict,
        num_steps: int = 10
    ) -> Dict:
        """
        Cartesian Linear Interpolation Motion Planning.

        Uses linear interpolation for position, SLERP for orientation.
        Sends absolute poses via cartesian_position action space.

        Args:
            target_pose: Target pose dict with position, orientation_euler, gripper
            num_steps: Number of interpolation steps

        Returns:
            Dict with success, final_distance, execution_time
        """
        target_pos = np.array(target_pose["position"])
        target_euler = np.array(target_pose["orientation_euler"])
        target_gripper = target_pose.get("gripper", None)

        print(f"🎯 Cartesian Linear Interpolation ({num_steps} steps)")
        print(f"   Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}], euler=[{target_euler[0]:.3f}, {target_euler[1]:.3f}, {target_euler[2]:.3f}]")

        start_time = time.time()

        # Get current pose
        current_pose = self.get_current_ee_pose()
        current_pos = current_pose["position"]
        current_euler = current_pose["orientation_euler"]
        if target_gripper is None:
            target_gripper = current_pose["gripper"]

        # Convert euler to quaternions for SLERP
        current_quat = R.from_euler('xyz', current_euler).as_quat()  # [x,y,z,w]
        target_quat = R.from_euler('xyz', target_euler).as_quat()

        # Create SLERP interpolator
        key_rotations = R.from_quat([current_quat, target_quat])
        slerp = Slerp([0, 1], key_rotations)

        # Execute interpolated motion with absolute poses
        for i in range(num_steps):
            t = (i + 1) / num_steps

            # Interpolate target position and orientation
            interp_pos = current_pos + t * (target_pos - current_pos)
            interp_rot = slerp(t)
            interp_euler = interp_rot.as_euler('xyz')

            # Construct absolute pose action: [x, y, z, roll, pitch, yaw, gripper]
            action = np.concatenate([interp_pos, interp_euler, [target_gripper]])

            # Execute using cartesian_position action space
            try:
                self.robot_env.update_robot(
                    action,
                    action_space="cartesian_position",
                    gripper_action_space="position",
                    blocking=True
                )
            except Exception as e:
                print(f"❌ CLI_MP failed at step {i}: {e}")
                return {
                    "success": False,
                    "final_distance": float('inf'),
                    "final_orientation_error": float('inf'),
                    "execution_time": time.time() - start_time,
                    "error": str(e)
                }

            # Early stopping check
            curr = self.get_current_ee_pose()
            pos_error = np.linalg.norm(target_pos - curr["position"])
            ori_error = compute_orientation_error(target_euler, curr["orientation_euler"])
            if pos_error < self.position_threshold and ori_error < self.orientation_threshold:
                print(f"   ✅ Converged early at step {i+1}/{num_steps}")
                break

        execution_time = time.time() - start_time

        # Final convergence check
        final_pose = self.get_current_ee_pose()
        final_distance = np.linalg.norm(target_pos - final_pose["position"])
        orientation_error = compute_orientation_error(target_euler, final_pose["orientation_euler"])
        success = (final_distance < self.position_threshold and orientation_error < self.orientation_threshold)

        if success:
            print(f"✅ Reached target ({execution_time:.2f}s)")
        else:
            print(f"⚠️  Not converged: dist={final_distance:.4f}m, ori={orientation_error:.4f}rad")

        return {
            "success": success,
            "final_distance": float(final_distance),
            "final_orientation_error": float(orientation_error),
            "execution_time": execution_time
        }

    def move_ee_up(
        self,
        lift_distance: float = 0.05,
        skill_type: str = "pick"
    ) -> bool:
        """
        Lift EE vertically by specified distance with retry logic.

        Used after pick/place skills to clear the object.

        Args:
            lift_distance: Vertical distance to lift (meters)
            skill_type: "pick" or "place" (for logging)

        Returns:
            True if successful
        """
        print(f"⬆️  Lifting EE by {lift_distance}m (after {skill_type})...")

        # Set gripper based on skill type (motion planner convention: 0=open, 1=closed)
        # pick: close gripper to hold object, then lift
        # place: open gripper to release object, then lift
        target_gripper = 1.0 if skill_type == "pick" else 0.0

        # Step 1: Send gripper-only action for 5 timesteps before lifting
        gripper_action_name = "Closing" if skill_type == "pick" else "Opening"
        print(f"   {gripper_action_name} gripper for 5 timesteps...")
        for _ in range(5):
            # Zero velocity action with gripper command: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            action = np.array([0, 0, 0, 0, 0, 0, target_gripper])
            self.robot_env.update_robot(action, action_space="cartesian_velocity", gripper_action_space="position")

        # Step 2: Calculate lift target based on current position (after gripper action)
        initial_pose = self.get_current_ee_pose()
        initial_pos = initial_pose["position"]

        target_pos = initial_pos.copy()
        target_pos[2] += lift_distance

        target_pose = {
            "position": target_pos,
            "orientation_euler": initial_pose["orientation_euler"],
            "gripper": target_gripper
        }

        # Step 3: Execute lift with retries
        for attempt in range(self.max_retries):
            # Execute lift to the SAME target each attempt
            result = self.move_to_pose(target_pose)

            if result["success"]:
                print(f"✅ Lifted EE successfully")
                return True
            else:
                if attempt < self.max_retries - 1:
                    print(f"⚠️  Lift attempt {attempt + 1}/{self.max_retries} failed (distance: {result['final_distance']:.4f}m), retrying...")
                else:
                    print(f"⚠️  Lift failed after {self.max_retries} attempts (distance: {result['final_distance']:.4f}m)")

        return False

    def get_current_ee_pose(self) -> Dict:
        """
        Get current EE pose from robot.

        Returns:
            Dict with:
                - position: np.ndarray [x, y, z] (meters)
                - orientation_euler: np.ndarray [roll, pitch, yaw] (radians)
                - gripper: float (0-1, where 0=open, 1=closed)
        """
        obs = self.robot_env.get_observation()
        robot_state = obs["robot_state"]

        # robot_state["cartesian_position"] is [x, y, z, roll, pitch, yaw]
        cartesian_position = np.array(robot_state["cartesian_position"])
        position = cartesian_position[:3]  # [x, y, z]
        orientation_euler = cartesian_position[3:6]  # [roll, pitch, yaw]
        gripper = robot_state["gripper_position"]

        return {
            "position": position,
            "orientation_euler": orientation_euler,
            "gripper": gripper
        }

    def move_to_home(self) -> bool:
        """
        Move robot to home/rest position.

        Returns:
            True if successful
        """
        print(f"🏠 Moving to home position...")

        # Define home position (safe resting pose)
        # Orientation: pointing downward (common for manipulation)
        home_pose = {
            "position": np.array([0.4, 0.0, 0.3]),  # Above table center
            "orientation_euler": np.array([np.pi, 0.0, 0.0]),  # Pointing down
            "gripper": 0.0  # Open
        }

        result = self.move_to_pose(home_pose)
        return result["success"]


if __name__ == "__main__":
    print("MotionPlanner test")
    print("Note: Requires RobotEnv to be initialized. Skipping actual test.")
    print()

    # Mock config for testing
    config = {
        "position_threshold": 0.02,
        "orientation_threshold": 0.524,
        "max_retries": 3
    }

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n✓ MotionPlanner supports:")
    print("  - IK_MP: Direct IK-based reaching (blocking)")
    print("  - CLI_MP: Cartesian Linear Interpolation (closed-loop)")
    print("\n✓ Target pose format:")
    print("  {")
    print("    'position': [x, y, z],")
    print("    'orientation_euler': [roll, pitch, yaw],")
    print("    'gripper': 0.0-1.0  # optional")
    print("  }")
