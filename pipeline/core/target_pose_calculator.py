#!/usr/bin/env python3
"""
Target Pose Calculator - Calculate target EE poses from object poses.

This module calculates "above object" poses for manipulation skills.
Simple approach: add height to object position with face-down orientation.
"""

import numpy as np
from typing import Dict, Tuple


class TargetPoseCalculator:
    """
    Calculate target EE poses (above pose) from object pose.

    Simple logic:
    - Extract object position
    - Add height offset (default 10cm)
    - Use face-down orientation
    - Support object-specific handling (future)
    """

    # Default face-down gripper orientation (euler: roll, pitch, yaw)
    # Roll=π means gripper pointing down
    REST_EULER = np.array([np.pi, 0.0, 0.0])

    # Gripper length from panda_link8 (flange) to gripper tip (meters)
    GRIPPER_LENGTH = 0.2

    def __init__(self, config: Dict):
        """
        Initialize target pose calculator.

        Args:
            config: Configuration dict with:
                - default_height: Default above height (meters, default 0.10)
                - object_specific: Object-specific overrides (dict, for future use)
        """
        self.default_height = config.get("default_height", 0.10)
        self.object_specific = config.get("object_specific", {})

        print(f"📐 TargetPoseCalculator initialized:")
        print(f"   Default above height: {self.default_height}m")

    def calculate_above_pose(
        self,
        object_pose: Dict,
        object_name: str,
        above_height: float = None,
        mesh_vertices: np.ndarray = None
    ) -> Dict:
        """
        Calculate "above object" EE pose for skill start.

        Logic:
        - Add height = default_height + half_longest_edge + GRIPPER_LENGTH
        - Use face-down orientation

        Args:
            object_pose: Object pose dict with "position" key
            object_name: Object name (for potential object-specific handling)
            above_height: Height above object (meters). If None, use default.
            mesh_vertices: Object mesh vertices (N, 3) for size calculation. Optional.

        Returns:
            Dictionary containing:
                - position: np.ndarray [x, y, z]
                - orientation_euler: np.ndarray [roll, pitch, yaw]
        """
        obj_pos = np.array(object_pose["position"])

        # Use default height if not provided
        if above_height is None:
            above_height = self.default_height

        # Calculate half of longest edge from mesh vertices
        half_longest_edge = 0.05
        # yy: hardcoded half_longest_edge - cuz it is not robust to extract from mesh vertices
        # if mesh_vertices is not None:
        #     bbox_min = mesh_vertices.min(axis=0)
        #     bbox_max = mesh_vertices.max(axis=0)
        #     extents = bbox_max - bbox_min  # [x_size, y_size, z_size]
        #     longest_edge = extents.max()
        #     half_longest_edge = longest_edge / 2.0

        # Total height offset = default_height + half_longest_edge + gripper_length
        total_height = above_height + half_longest_edge + self.GRIPPER_LENGTH

        # Add extra height for specific objects
        if object_name == "holder_tree":
            total_height += 0.10  # Extra 10cm for holder_tree
            print(f"   Added extra 10cm for holder_tree")
        if object_name == "basket":
            total_height += 0.10  # Extra 10cm for basket
            print(f"   Added extra 10cm for basket")

        # Calculate above position: same XY, add height to Z
        above_pos = obj_pos.copy()
        above_pos[2] += total_height

        # Object-specific XY adjustments
        if object_name == "holder_tree":
            above_pos[1] -= 0.10  # -10cm in Y axis for holder_tree
            print(f"   Added -10cm Y offset for holder_tree")

        # Use face-down orientation
        above_euler = self.REST_EULER.copy()

        # Apply object-specific handling (placeholder for future use)
        above_pos, above_euler = self._apply_object_specific_handling(
            above_pos, above_euler, object_name
        )

        print(f"📏 Calculated above pose for '{object_name}':")
        print(f"   Object pos: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        print(f"   Height breakdown: default={above_height:.3f}m + half_edge={half_longest_edge:.3f}m + gripper={self.GRIPPER_LENGTH:.3f}m = {total_height:.3f}m")
        print(f"   Above pos: [{above_pos[0]:.3f}, {above_pos[1]:.3f}, {above_pos[2]:.3f}]")

        return {
            "position": above_pos,
            "orientation_euler": above_euler
        }

    def _apply_object_specific_handling(
        self,
        pos: np.ndarray,
        euler: np.ndarray,
        object_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply object-specific pose adjustments.

        Placeholder for future object-specific handling.
        Examples:
        - Drawer: approach from front, add XY offset
        - Stove: offset to burner center
        - Handle: align with handle orientation

        Args:
            pos: Above position [x, y, z]
            euler: Above orientation [roll, pitch, yaw]
            object_name: Object name

        Returns:
            (pos, euler) tuple (currently unchanged)
        """
        # Placeholder: currently no object-specific handling
        # Future: check object_name and apply custom logic
        # if "drawer" in object_name.lower():
        #     pos[0] += 0.05  # Example: offset in front of drawer
        # if "stove" in object_name.lower():
        #     pos[0] += 0.10  # Example: offset to burner center

        return pos, euler

    def add_random_perturbation(
        self,
        pose: Dict,
        xy_range: float = 0.02,
        z_range: float = 0.01,
        ori_range_rad: float = 0.1
    ) -> Dict:
        """
        Add random perturbation to pose (for retry logic).

        Utility function that can be used by other modules.

        Args:
            pose: Original pose dict with "position" and "orientation_euler"
            xy_range: XY perturbation range (meters)
            z_range: Z perturbation range (meters)
            ori_range_rad: Orientation perturbation range (radians)

        Returns:
            Perturbed pose dict
        """
        perturbed_pos = pose["position"].copy()
        perturbed_euler = pose["orientation_euler"].copy()

        # Position perturbation
        xy_shift = np.random.uniform(-xy_range, xy_range, size=2)
        z_shift = np.random.uniform(-z_range, z_range)
        perturbed_pos[0] += xy_shift[0]
        perturbed_pos[1] += xy_shift[1]
        perturbed_pos[2] += z_shift

        # Orientation perturbation (add random noise to each euler angle)
        ori_shift = np.random.uniform(-ori_range_rad, ori_range_rad, size=3)
        perturbed_euler += ori_shift

        print(f"🔀 Added perturbation:")
        print(f"   Position: xy=[{xy_shift[0]:.3f}, {xy_shift[1]:.3f}], z={z_shift:.3f}")
        print(f"   Orientation: [{ori_shift[0]:.3f}, {ori_shift[1]:.3f}, {ori_shift[2]:.3f}] rad")

        return {
            "position": perturbed_pos,
            "orientation_euler": perturbed_euler
        }


if __name__ == "__main__":
    # Test target pose calculator
    print("Testing TargetPoseCalculator...\n")

    config = {
        "default_height": 0.10,
        "object_specific": {}
    }

    calculator = TargetPoseCalculator(config)

    # Test 1: Simple above pose
    object_pose = {
        "position": np.array([0.4, 0.0, 0.1])
    }

    above_pose = calculator.calculate_above_pose(object_pose, "red_cup")
    print(f"Above pose position: {above_pose['position']}")
    print(f"Above pose orientation (euler): {above_pose['orientation_euler']}")
    print()

    # Test 2: Custom height
    custom_above = calculator.calculate_above_pose(object_pose, "black_bowl", above_height=0.15)
    print(f"Custom height above pose: {custom_above['position']}")
    print()

    # Test 3: Random perturbation
    perturbed = calculator.add_random_perturbation(
        above_pose,
        xy_range=0.02,
        z_range=0.01,
        ori_range_rad=0.1
    )
    print(f"Perturbed position: {perturbed['position']}")
    print(f"Perturbed orientation: {perturbed['orientation_euler']}")
    print()

    print("✓ TargetPoseCalculator test complete")
