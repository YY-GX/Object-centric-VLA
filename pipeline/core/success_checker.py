#!/usr/bin/env python3
"""
Success Checker - Verify atomic skill completion.

Implements success checking methods:
- object_lifted: Check if object's Z position increased by threshold
- object_on_target: Check if grasp_object is placed on target_object (uses mesh for Z threshold)
- object_in_target: Check if grasp_object is in target_object (fixed Z threshold)
- object_hang: Check if object is hung (lifted, near target XY, and stable)
"""

import numpy as np
import trimesh
from typing import Dict, List, Optional
from collections import deque


class SuccessChecker:
    """
    Check skill success using heuristics.

    Supports:
    - object_lifted: Object lifted by min_height_increase
    - object_on_target: Object placed on target (XY + Z with mesh-based threshold)
    - object_in_target: Object in target (XY + fixed Z threshold)
    - object_hang: Object hung (lifted + near target XY + stable position)
    """

    def __init__(self, config: Dict):
        """
        Initialize success checker.

        Args:
            config: Success checking config with:
                - user_decide_mode: bool (if True, skip automatic checks and ask user y/n)
                - object_lifted:
                    - min_height_increase: float (meters, default 0.03)
                - object_on_target:
                    - xy_distance_threshold: float (meters, default 0.05)
                    - z_buffer_distance: float (meters, default 0.03)
                - object_in_target:
                    - xy_distance_threshold: float (meters, default 0.05)
                    - z_distance_threshold: float (meters, default 0.05)
                - object_hang:
                    - min_height_increase: float (meters, default 0.02)
                    - xy_distance_threshold: float (meters, default 0.05)
                    - stability_threshold: float (meters, default 0.02)
                    - stability_window: int (timesteps, default 10)
        """
        self.config = config

        # User decide mode - skip automatic checks and ask user
        self.user_decide_mode = config.get("user_decide_mode", False)
        if self.user_decide_mode:
            print(f"   ⚠️  USER DECIDE MODE ENABLED - will ask user for success confirmation")

        # object_lifted parameters
        lifted_config = config.get("object_lifted", {})
        self.min_height_increase = lifted_config.get("min_height_increase", 0.03)

        # object_on_target parameters
        on_target_config = config.get("object_on_target", {})
        self.on_target_xy_threshold = on_target_config.get("xy_distance_threshold", 0.05)
        self.z_buffer_distance = on_target_config.get("z_buffer_distance", 0.03)

        # object_in_target parameters
        in_target_config = config.get("object_in_target", {})
        self.in_target_xy_threshold = in_target_config.get("xy_distance_threshold", 0.05)
        self.in_target_z_threshold = in_target_config.get("z_distance_threshold", 0.05)

        # object_hang parameters
        hang_config = config.get("object_hang", {})
        self.hang_min_height_increase = hang_config.get("min_height_increase", 0.02)
        self.hang_xy_threshold = hang_config.get("xy_distance_threshold", 0.05)
        self.hang_stability_threshold = hang_config.get("stability_threshold", 0.02)
        self.hang_stability_window = hang_config.get("stability_window", 10)

        # Position history for stability check (object_hang)
        # Key: object_name, Value: deque of positions
        self._position_history: Dict[str, deque] = {}

        print(f"SuccessChecker initialized:")
        print(f"   object_lifted: min_height={self.min_height_increase}m")
        print(f"   object_on_target: xy_threshold={self.on_target_xy_threshold}m, z_buffer={self.z_buffer_distance}m")
        print(f"   object_in_target: xy_threshold={self.in_target_xy_threshold}m, z_threshold={self.in_target_z_threshold}m")
        print(f"   object_hang: min_height={self.hang_min_height_increase}m, xy_threshold={self.hang_xy_threshold}m, stability={self.hang_stability_threshold}m over {self.hang_stability_window} steps")

    def reset_position_history(self, object_name: Optional[str] = None):
        """
        Reset position history for stability tracking.

        Args:
            object_name: Specific object to reset, or None to reset all
        """
        if object_name is None:
            self._position_history.clear()
        elif object_name in self._position_history:
            del self._position_history[object_name]

    def _check_user_decide(self, skill_info: Dict, success_check_type: str) -> Dict:
        """Ask user to decide if skill succeeded."""
        language = skill_info.get("language", "unknown skill")
        target = skill_info.get("target_object", "unknown")

        print(f"\n{'='*60}")
        print(f"⚠️  USER SUCCESS CHECK")
        print(f"{'='*60}")
        print(f"Skill: {language}")
        print(f"Target object: {target}")
        print(f"Check type: {success_check_type}")
        print(f"{'='*60}")

        while True:
            user_input = input("Did the skill succeed? (y/n): ").strip().lower()
            if user_input in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")

        success = user_input == 'y'
        return {
            "success": success,
            "confidence": 1.0,
            "reason": f"User decided: {'SUCCESS' if success else 'FAILURE'}",
            "metrics": {"user_decide_mode": True}
        }

    def check_success(
        self,
        success_check_type: str,
        skill_info: Dict,
        initial_states: Dict,
        current_states: Dict,
        mesh_paths: Dict
    ) -> Dict:
        """
        Check if skill succeeded.

        Args:
            success_check_type: "object_lifted", "object_on_target", "object_in_target", or "object_hang"
            skill_info: From task_config (has target_object, grasp_object, etc.)
            initial_states: {object_name: {"position": [x,y,z]}}
            current_states: {object_name: {"position": [x,y,z]}}
            mesh_paths: {object_name: "/path/to/mesh.obj"}

        Returns:
            Dict with:
                - success: bool
                - confidence: float (0-1)
                - reason: str (explanation)
                - metrics: dict (measurements)
        """
        # In user_decide_mode, skip automatic checks and ask user
        if self.user_decide_mode:
            return self._check_user_decide(skill_info, success_check_type)

        if success_check_type == "object_lifted":
            return self._check_object_lifted(skill_info, initial_states, current_states)
        elif success_check_type == "object_on_target":
            return self._check_object_on_target(skill_info, initial_states, current_states, mesh_paths)
        elif success_check_type == "object_in_target":
            return self._check_object_in_target(skill_info, initial_states, current_states)
        elif success_check_type == "object_hang":
            return self._check_object_hang(skill_info, initial_states, current_states)
        else:
            return {
                "success": False,
                "confidence": 0.0,
                "reason": f"Unknown success check type: {success_check_type}",
                "metrics": {}
            }

    def _check_object_lifted(
        self,
        skill_info: Dict,
        initial_states: Dict,
        current_states: Dict
    ) -> Dict:
        """
        Check if object lifted above initial height.

        Args:
            skill_info: Contains "target_object"
            initial_states: Initial positions
            current_states: Current positions

        Returns:
            Success dict with metrics
        """
        target_object = skill_info["target_object"]

        if target_object not in initial_states or target_object not in current_states:
            return {
                "success": False,
                "confidence": 0.0,
                "reason": f"Missing state data for {target_object}",
                "metrics": {}
            }

        initial_pos = initial_states[target_object]["position"]
        current_pos = current_states[target_object]["position"]
        initial_z = initial_pos[2]
        current_z = current_pos[2]
        height_increase = current_z - initial_z

        # Debug: print object poses
        print(f"   [DEBUG] {target_object} initial pos: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}, {initial_pos[2]:.4f}]")
        print(f"   [DEBUG] {target_object} current pos: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")

        success = height_increase >= self.min_height_increase

        return {
            "success": success,
            "confidence": 1.0 if success else 0.0,
            "reason": f"Height increase: {height_increase:.4f}m (threshold: {self.min_height_increase}m)",
            "metrics": {
                "initial_z": float(initial_z),
                "current_z": float(current_z),
                "height_increase": float(height_increase),
                "threshold": self.min_height_increase
            }
        }

    def _check_object_on_target(
        self,
        skill_info: Dict,
        initial_states: Dict,
        current_states: Dict,
        mesh_paths: Dict
    ) -> Dict:
        """
        Check if grasp_object placed on target_object.

        Checks:
        1. XY distance < xy_distance_threshold (5cm)
        2. Z distance < half_longest_edge + z_buffer_distance

        Args:
            skill_info: Contains "target_object" and "grasp_object"
            initial_states: Initial positions
            current_states: Current positions
            mesh_paths: Mesh file paths

        Returns:
            Success dict with metrics
        """
        target_object = skill_info["target_object"]

        # Validate grasp_object exists
        if "grasp_object" not in skill_info:
            raise ValueError(f"skill_info must contain 'grasp_object' for object_on_target check. Got: {skill_info.keys()}")

        grasp_object = skill_info["grasp_object"]

        # Check state data exists
        if target_object not in current_states or grasp_object not in current_states:
            return {
                "success": False,
                "confidence": 0.0,
                "reason": f"Missing state data for {target_object} or {grasp_object}",
                "metrics": {}
            }

        target_pos = np.array(current_states[target_object]["position"])
        grasp_pos = np.array(current_states[grasp_object]["position"])

        # Debug: print object poses
        print(f"   [DEBUG] {target_object} (target) pos: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        print(f"   [DEBUG] {grasp_object} (grasp) pos: [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}]")

        # Check 1: XY distance
        xy_distance = np.linalg.norm(target_pos[:2] - grasp_pos[:2])
        xy_check = xy_distance < self.on_target_xy_threshold

        # Check 2: Z distance (load mesh to get half_longest_edge)
        if grasp_object not in mesh_paths:
            return {
                "success": False,
                "confidence": 0.0,
                "reason": f"Mesh path not found for {grasp_object}",
                "metrics": {}
            }

        mesh = trimesh.load(mesh_paths[grasp_object])
        bbox_min = mesh.vertices.min(axis=0)
        bbox_max = mesh.vertices.max(axis=0)
        extents = bbox_max - bbox_min
        longest_edge = extents.max()
        half_longest_edge = longest_edge / 2.0

        z_distance = abs(target_pos[2] - grasp_pos[2])
        z_threshold = half_longest_edge + self.z_buffer_distance
        z_check = z_distance < z_threshold

        success = xy_check and z_check

        return {
            "success": success,
            "confidence": 1.0 if success else 0.0,
            "reason": f"XY: {xy_distance:.4f}m (<{self.on_target_xy_threshold}m: {xy_check}), Z: {z_distance:.4f}m (<{z_threshold:.4f}m: {z_check})",
            "metrics": {
                "xy_distance": float(xy_distance),
                "xy_threshold": self.on_target_xy_threshold,
                "xy_check": xy_check,
                "z_distance": float(z_distance),
                "z_threshold": float(z_threshold),
                "z_check": z_check,
                "half_longest_edge": float(half_longest_edge)
            }
        }

    def _check_object_in_target(
        self,
        skill_info: Dict,
        initial_states: Dict,
        current_states: Dict
    ) -> Dict:
        """
        Check if grasp_object is in target_object (e.g., cup in basket).

        Checks:
        1. XY distance < xy_distance_threshold (5cm)
        2. Z distance < z_distance_threshold (5cm) - fixed threshold, no mesh

        Args:
            skill_info: Contains "target_object" and "grasp_object"
            initial_states: Initial positions
            current_states: Current positions

        Returns:
            Success dict with metrics
        """
        target_object = skill_info["target_object"]

        # Validate grasp_object exists
        if "grasp_object" not in skill_info:
            raise ValueError(f"skill_info must contain 'grasp_object' for object_in_target check. Got: {skill_info.keys()}")

        grasp_object = skill_info["grasp_object"]

        # Check state data exists
        if target_object not in current_states or grasp_object not in current_states:
            return {
                "success": False,
                "confidence": 0.0,
                "reason": f"Missing state data for {target_object} or {grasp_object}",
                "metrics": {}
            }

        target_pos = np.array(current_states[target_object]["position"])
        grasp_pos = np.array(current_states[grasp_object]["position"])

        # Debug: print object poses
        print(f"   [DEBUG] {target_object} (target) pos: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        print(f"   [DEBUG] {grasp_object} (grasp) pos: [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}]")

        # Check 1: XY distance
        xy_distance = np.linalg.norm(target_pos[:2] - grasp_pos[:2])
        xy_check = xy_distance < self.in_target_xy_threshold

        # Check 2: Z distance (fixed threshold, no mesh)
        z_distance = abs(target_pos[2] - grasp_pos[2])
        z_check = z_distance < self.in_target_z_threshold

        success = xy_check and z_check

        return {
            "success": success,
            "confidence": 1.0 if success else 0.0,
            "reason": f"XY: {xy_distance:.4f}m (<{self.in_target_xy_threshold}m: {xy_check}), Z: {z_distance:.4f}m (<{self.in_target_z_threshold}m: {z_check})",
            "metrics": {
                "xy_distance": float(xy_distance),
                "xy_threshold": self.in_target_xy_threshold,
                "xy_check": xy_check,
                "z_distance": float(z_distance),
                "z_threshold": self.in_target_z_threshold,
                "z_check": z_check
            }
        }

    def _check_object_hang(
        self,
        skill_info: Dict,
        initial_states: Dict,
        current_states: Dict
    ) -> Dict:
        """
        Check if object is hung on target (e.g., tape on holder tree).

        Three conditions:
        1. Height increase >= min_height_increase (2cm)
        2. XY distance to target < xy_distance_threshold (5cm)
        3. Object position stable for stability_window consecutive timesteps
           (position change < stability_threshold for 10 steps)

        Args:
            skill_info: Contains "target_object" and "grasp_object"
            initial_states: Initial positions
            current_states: Current positions

        Returns:
            Success dict with metrics
        """
        target_object = skill_info["target_object"]

        # Validate grasp_object exists
        if "grasp_object" not in skill_info:
            raise ValueError(f"skill_info must contain 'grasp_object' for object_hang check. Got: {skill_info.keys()}")

        grasp_object = skill_info["grasp_object"]

        # Check state data exists
        if grasp_object not in initial_states or grasp_object not in current_states:
            return {
                "success": False,
                "confidence": 0.0,
                "reason": f"Missing state data for {grasp_object}",
                "metrics": {}
            }

        if target_object not in current_states:
            return {
                "success": False,
                "confidence": 0.0,
                "reason": f"Missing state data for {target_object}",
                "metrics": {}
            }

        initial_pos = np.array(initial_states[grasp_object]["position"])
        current_pos = np.array(current_states[grasp_object]["position"])
        target_pos = np.array(current_states[target_object]["position"])

        # Condition 1: Height increase >= threshold (2cm)
        height_increase = current_pos[2] - initial_pos[2]
        height_check = height_increase >= self.hang_min_height_increase

        # Condition 2: XY distance to target < threshold (5cm)
        xy_distance = np.linalg.norm(target_pos[:2] - current_pos[:2])
        xy_check = xy_distance < self.hang_xy_threshold

        # Condition 3: Position stability check
        # Initialize position history if needed
        if grasp_object not in self._position_history:
            self._position_history[grasp_object] = deque(maxlen=self.hang_stability_window)

        # Add current position to history
        self._position_history[grasp_object].append(current_pos.copy())

        # Check stability: need at least stability_window positions
        position_history = self._position_history[grasp_object]
        stability_check = False
        max_position_change = float('inf')

        if len(position_history) >= self.hang_stability_window:
            # Check if all positions in window are within threshold of each other
            positions = np.array(list(position_history))
            # Compute max distance from first position in window to all others
            reference_pos = positions[0]
            distances = np.linalg.norm(positions - reference_pos, axis=1)
            max_position_change = distances.max()
            stability_check = max_position_change < self.hang_stability_threshold

        success = height_check and xy_check and stability_check

        return {
            "success": success,
            "confidence": 1.0 if success else 0.0,
            "reason": f"Height: {height_increase:.4f}m (>={self.hang_min_height_increase}m: {height_check}), "
                      f"XY: {xy_distance:.4f}m (<{self.hang_xy_threshold}m: {xy_check}), "
                      f"Stable: {max_position_change:.4f}m (<{self.hang_stability_threshold}m over {len(position_history)} steps: {stability_check})",
            "metrics": {
                "height_increase": float(height_increase),
                "height_threshold": self.hang_min_height_increase,
                "height_check": height_check,
                "xy_distance": float(xy_distance),
                "xy_threshold": self.hang_xy_threshold,
                "xy_check": xy_check,
                "max_position_change": float(max_position_change) if max_position_change != float('inf') else None,
                "stability_threshold": self.hang_stability_threshold,
                "stability_window": self.hang_stability_window,
                "history_length": len(position_history),
                "stability_check": stability_check
            }
        }


if __name__ == "__main__":
    # Test success checker
    print("Testing SuccessChecker...\n")

    config = {
        "object_lifted": {"min_height_increase": 0.03},
        "object_on_target": {"xy_distance_threshold": 0.05, "z_buffer_distance": 0.03},
        "object_in_target": {"xy_distance_threshold": 0.05, "z_distance_threshold": 0.05},
        "object_hang": {
            "min_height_increase": 0.02,
            "xy_distance_threshold": 0.05,
            "stability_threshold": 0.02,
            "stability_window": 10
        }
    }

    checker = SuccessChecker(config)

    # Test 1: object_lifted - success
    print("\n=== Test 1: object_lifted (success) ===")
    skill_info = {"target_object": "red_cup"}
    initial_states = {"red_cup": {"position": np.array([0.4, 0.0, 0.10])}}
    current_states = {"red_cup": {"position": np.array([0.4, 0.0, 0.15])}}

    result = checker.check_success("object_lifted", skill_info, initial_states, current_states, {})
    print(f"Result: {result}")

    # Test 2: object_in_target - success
    print("\n=== Test 2: object_in_target (success) ===")
    skill_info = {"target_object": "basket", "grasp_object": "red_cup"}
    initial_states = {"red_cup": {"position": np.array([0.4, 0.0, 0.10])}}
    current_states = {
        "red_cup": {"position": np.array([0.5, 0.02, 0.12])},
        "basket": {"position": np.array([0.5, 0.0, 0.10])}
    }

    result = checker.check_success("object_in_target", skill_info, initial_states, current_states, {})
    print(f"Result: {result}")

    # Test 3: object_hang - need multiple calls to build history
    print("\n=== Test 3: object_hang (building history) ===")
    skill_info = {"target_object": "holder_tree", "grasp_object": "tape"}
    initial_states = {"tape": {"position": np.array([0.4, 0.0, 0.10])}}

    # Simulate 15 timesteps with stable position
    for i in range(15):
        current_states = {
            "tape": {"position": np.array([0.42, 0.02, 0.15]) + np.random.randn(3) * 0.001},  # small noise
            "holder_tree": {"position": np.array([0.4, 0.0, 0.12])}
        }
        result = checker.check_success("object_hang", skill_info, initial_states, current_states, {})
        print(f"Step {i+1}: success={result['success']}, history_len={result['metrics']['history_length']}")

    print(f"\nFinal result: {result}")

    print("\nSuccessChecker test complete")
