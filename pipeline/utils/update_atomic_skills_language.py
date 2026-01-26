#!/usr/bin/env python3
"""
Update language instructions in atomic_skills H5 files to match folder names.

Sets `current_task` and `new_tasks` to the skill folder name with underscores
replaced by spaces.

Usage:
    python update_atomic_skills_language.py --dry_run  # Preview changes
    python update_atomic_skills_language.py            # Apply changes
"""

import argparse
import h5py
from pathlib import Path


def update_h5_language(h5_path: Path, new_instruction: str, dry_run: bool = False) -> tuple:
    """
    Update current_task and new_tasks in H5 file.

    Returns:
        Tuple of (old_value, new_value, changed)
    """
    with h5py.File(h5_path, 'r') as f:
        old_task = f.attrs.get('current_task', 'N/A')
        if isinstance(old_task, bytes):
            old_task = old_task.decode('utf-8')

    changed = old_task != new_instruction

    if changed and not dry_run:
        with h5py.File(h5_path, 'a') as f:
            f.attrs['current_task'] = new_instruction
            f.attrs['new_tasks'] = [new_instruction]

    return (old_task, new_instruction, changed)


def process_atomic_skills(base_dir: Path, dry_run: bool = False) -> dict:
    """
    Process all trajectories in atomic_skills directory.

    Args:
        base_dir: Path to atomic_skills directory
        dry_run: If True, only preview changes

    Returns:
        Stats dict
    """
    stats = {'updated': 0, 'skipped': 0, 'errors': 0}

    # Get all skill folders
    skill_folders = [d for d in base_dir.iterdir() if d.is_dir()]

    for skill_folder in sorted(skill_folders):
        skill_name = skill_folder.name
        expected_instruction = skill_name.replace('_', ' ')

        # Get all trajectory folders (including _random_erased)
        traj_folders = [d for d in skill_folder.iterdir() if d.is_dir()]

        if not traj_folders:
            continue

        print(f"\n{'='*60}")
        print(f"Skill: {skill_name}")
        print(f"Expected instruction: \"{expected_instruction}\"")
        print(f"{'='*60}")

        for traj_folder in sorted(traj_folders):
            h5_path = traj_folder / 'trajectory.h5'

            if not h5_path.exists():
                continue

            old_val, new_val, changed = update_h5_language(h5_path, expected_instruction, dry_run)

            if changed:
                status = "[DRY RUN]" if dry_run else "[UPDATED]"
                print(f"  {status} {traj_folder.name}")
                print(f"           \"{old_val}\" -> \"{new_val}\"")
                stats['updated'] += 1
            else:
                stats['skipped'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Update language instructions in atomic_skills H5 files"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/home/yygx/PANDA/data/yue_dataset/atomic_skills',
        help='Path to atomic_skills directory'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return

    print(f"Base directory: {base_dir}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***")

    stats = process_atomic_skills(base_dir, args.dry_run)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Skipped (already correct): {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")

    if args.dry_run:
        print("\n*** DRY RUN COMPLETE - Run without --dry_run to apply changes ***")


if __name__ == '__main__':
    main()
