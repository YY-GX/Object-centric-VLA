#!/usr/bin/env python3
"""
Success Checker - Verify atomic skill completion.

Implements two success checking methods:
- object_lifted: Check if object's Z position increased by threshold
- object_on_target: Check if grasp_object is placed on target_object
"""

import numpy as np
import trimesh
from typing import Dict


class SuccessChecker:
    """
    Check skill success using heuristics.

    Supports:
    - object_lifted: Object lifted by min_height_increase
    - object_on_target: Object placed on target (XY + Z distance checks)
    """

    def __init__(self, config: Dict):
        """
        Initialize success checker.

        Args:
            config: Success checking config with:
                - object_lifted:
                    - min_height_increase: float (meters, default 0.03)
                - object_on_target:
                    - xy_distance_threshold: float (meters, default 0.05)
                    - z_buffer_distance: float (meters, default 0.03)
        """
        self.config = config

        # object_lifted parameters
        lifted_config = config.get("object_lifted", {})
        self.min_height_increase = lifted_config.get("min_height_increase", 0.03)

        # object_on_target parameters
        on_target_config = config.get("object_on_target", {})
        self.xy_distance_threshold = on_target_config.get("xy_distance_threshold", 0.05)
        self.z_buffer_distance = on_target_config.get("z_buffer_distance", 0.03)

        print(f"✓ SuccessChecker initialized:")
        print(f"   object_lifted: min_height={self.min_height_increase}m")
        print(f"   object_on_target: xy_threshold={self.xy_distance_threshold}m, z_buffer={self.z_buffer_distance}m")

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
            success_check_type: "object_lifted" or "object_on_target"
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
        if success_check_type == "object_lifted":
            return self._check_object_lifted(skill_info, initial_states, current_states)
        elif success_check_type == "object_on_target":
            return self._check_object_on_target(skill_info, initial_states, current_states, mesh_paths)
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

        initial_z = initial_states[target_object]["position"][2]
        current_z = current_states[target_object]["position"][2]
        height_increase = current_z - initial_z

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

        # Check 1: XY distance
        xy_distance = np.linalg.norm(target_pos[:2] - grasp_pos[:2])
        xy_check = xy_distance < self.xy_distance_threshold

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
            "reason": f"XY: {xy_distance:.4f}m (<{self.xy_distance_threshold}m: {xy_check}), Z: {z_distance:.4f}m (<{z_threshold:.4f}m: {z_check})",
            "metrics": {
                "xy_distance": float(xy_distance),
                "xy_threshold": self.xy_distance_threshold,
                "xy_check": xy_check,
                "z_distance": float(z_distance),
                "z_threshold": float(z_threshold),
                "z_check": z_check,
                "half_longest_edge": float(half_longest_edge)
            }
        }


if __name__ == "__main__":
    # Test success checker
    print("Testing SuccessChecker...\n")

    config = {
        "object_lifted": {"min_height_increase": 0.03},
        "object_on_target": {"xy_distance_threshold": 0.05, "z_buffer_distance": 0.03}
    }

    checker = SuccessChecker(config)

    # Test 1: object_lifted - success
    print("\n=== Test 1: object_lifted (success) ===")
    skill_info = {"target_object": "red_cup"}
    initial_states = {"red_cup": {"position": np.array([0.4, 0.0, 0.10])}}
    current_states = {"red_cup": {"position": np.array([0.4, 0.0, 0.15])}}

    result = checker.check_success("object_lifted", skill_info, initial_states, current_states, {})
    print(f"Result: {result}")

    # Test 2: object_lifted - fail
    print("\n=== Test 2: object_lifted (fail) ===")
    current_states = {"red_cup": {"position": np.array([0.4, 0.0, 0.12])}}
    result = checker.check_success("object_lifted", skill_info, initial_states, current_states, {})
    print(f"Result: {result}")

    print("\n✓ SuccessChecker test complete")
