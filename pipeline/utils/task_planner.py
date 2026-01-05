#!/usr/bin/env python3
"""
Task Planner - Load task config and plan skill sequences.

This module loads task and skill definitions from JSON config
and returns skill sequences for long-horizon tasks.

Similar to LIBERO's task_planner.py but simplified for real robot.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path


def load_task_config(config_path: str) -> Dict:
    """
    Load task configuration from JSON file.

    Args:
        config_path: Path to task_config.json

    Returns:
        Dictionary with:
            - long_horizon_tasks: List of task definitions
            - skill_mappings: Dict of skill_name -> skill_info
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"📋 Loaded task config from {config_path}")
    print(f"   Tasks: {len(config.get('long_horizon_tasks', []))}")
    print(f"   Skills: {len(config.get('skill_mappings', {}))}")

    return config


def plan_task_sequence(task_name: str, config: Dict) -> Optional[List[str]]:
    """
    Get skill sequence for a long-horizon task.

    Args:
        task_name: Name of task (e.g., "pick the black bowl")
        config: Task config dictionary

    Returns:
        List of skill names in execution order, or None if not found
    """
    tasks = config.get("long_horizon_tasks", [])
    task_name_lower = task_name.lower().strip()

    # Find matching task (case-insensitive)
    for task in tasks:
        if task["name"].lower().strip() == task_name_lower:
            skills = task["skills"]

            print(f"✅ Found task: '{task['name']}'")
            print(f"   Task ID: {task['task_id']}")
            print(f"   Skills ({len(skills)}):")
            for i, skill in enumerate(skills, 1):
                print(f"     {i}. {skill}")

            return skills

    # Task not found
    available_tasks = [task["name"] for task in tasks]
    print(f"❌ Task '{task_name}' not found")
    print(f"📋 Available tasks ({len(available_tasks)}):")
    for task in available_tasks:
        print(f"   - {task}")

    return None


def get_skill_info(skill_name: str, config: Dict) -> Optional[Dict]:
    """
    Get skill metadata (target object, language, skill type).

    Args:
        skill_name: Skill name (e.g., "pick black bowl")
        config: Task config dictionary

    Returns:
        Skill info dict or None if not found
    """
    skill_mappings = config.get("skill_mappings", {})
    skill_name_lower = skill_name.lower().strip()

    # Direct lookup
    if skill_name in skill_mappings:
        return skill_mappings[skill_name]

    # Case-insensitive lookup
    for name, info in skill_mappings.items():
        if name.lower() == skill_name_lower:
            return info

    print(f"⚠️  Skill '{skill_name}' not found in config")
    return None


def get_all_tasks(config: Dict) -> List[str]:
    """
    Get list of all available task names.

    Args:
        config: Task config dictionary

    Returns:
        List of task names
    """
    tasks = config.get("long_horizon_tasks", [])
    return [task["name"] for task in tasks]


def get_all_skills(config: Dict) -> List[str]:
    """
    Get list of all available skill names.

    Args:
        config: Task config dictionary

    Returns:
        List of skill names
    """
    skill_mappings = config.get("skill_mappings", {})
    return list(skill_mappings.keys())


if __name__ == "__main__":
    # Test task planner
    print("Testing TaskPlanner...\n")

    # Load config
    config_path = "../config/task_config.json"

    if Path(config_path).exists():
        config = load_task_config(config_path)

        print()

        # Test plan_task_sequence
        task_name = "pick the black bowl"
        skills = plan_task_sequence(task_name, config)

        print()

        # Test get_skill_info
        if skills:
            skill_name = skills[0]
            skill_info = get_skill_info(skill_name, config)
            print(f"Skill info for '{skill_name}':")
            if skill_info:
                for k, v in skill_info.items():
                    print(f"  {k}: {v}")

        print()

        # List all tasks
        all_tasks = get_all_tasks(config)
        print(f"All tasks: {all_tasks}")

        print()

        # List all skills
        all_skills = get_all_skills(config)
        print(f"All skills ({len(all_skills)}): {all_skills[:5]}...")

    else:
        print(f"Config file not found: {config_path}")
        print("Run from pipeline directory or adjust path")
