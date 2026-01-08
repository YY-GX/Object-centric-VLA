#!/usr/bin/env python3
"""
Skill Helpers for Real Robot Pipeline.

Provides helper functions for skill-related operations like
determining which objects to track and identifying distractors.
"""

from typing import Dict, List


def get_tracked_objects_for_skill(skill_info: Dict) -> List[str]:
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


def get_distractor_objects(
    skill_info: Dict,
    task_config: Dict,
    task_name: str
) -> List[str]:
    """
    Get list of distractor objects for random erasing.

    Distractors are objects in scene_objects that are NOT:
    - The current skill's target_object
    - The current skill's grasp_object (if any)

    Args:
        skill_info: Skill information from task config
        task_config: Full task configuration dict
        task_name: Name of current task

    Returns:
        List of distractor object names
    """
    # Get scene_objects from current task
    task_info = None
    for task in task_config.get("long_horizon_tasks", []):
        if task["name"] == task_name:
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


def should_keep_tracking(
    object_name: str,
    current_skill_idx: int,
    skill_sequence: List[str],
    task_config: Dict,
    get_skill_info_func
) -> bool:
    """
    Check if object should continue being tracked after current skill.

    Args:
        object_name: Object to check
        current_skill_idx: Index of current skill in sequence
        skill_sequence: List of skill names in execution order
        task_config: Task configuration dict
        get_skill_info_func: Function to get skill info from task config

    Returns:
        True if next skill needs this object
    """
    # Check if there's a next skill
    if current_skill_idx + 1 >= len(skill_sequence):
        return False

    # Get next skill info
    next_skill_name = skill_sequence[current_skill_idx + 1]
    next_skill_info = get_skill_info_func(next_skill_name, task_config)

    if not next_skill_info:
        return False

    # Check if next skill needs this object
    next_tracked_objects = get_tracked_objects_for_skill(next_skill_info)
    return object_name in next_tracked_objects
