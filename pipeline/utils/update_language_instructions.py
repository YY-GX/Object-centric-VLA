#!/usr/bin/env python3
"""
Update language instructions in pick/place H5 files.

Updates the `current_task` and `new_tasks` attributes in trajectory.h5 files
located in /pick and /place subfolders.

Usage:
    # Preview changes without modifying
    python update_language_instructions.py --dry_run

    # Apply changes
    python update_language_instructions.py

    # Custom data directory
    python update_language_instructions.py --data_dir /path/to/data/success/2026-01-24
"""

import argparse
import h5py
from pathlib import Path


# Mapping: task_folder_name -> (pick_instruction, place_instruction)
LANGUAGE_MAPPING = {
    "hang_the_tape_onto_the_holder_tree": ("pick the tape", "hang the tape on the holder tree"),
    "put_the_cup_in_the_basket": ("pick the red cup", "place the red cup into the basket"),
    "put_the_mustard_in_the_basket": ("pick the yellow mustard", "place the yellow mustard into the basket"),
    "stack_the_cups": ("pick the red cup", "stack the red cup on the red cup"),
    "stack_the_white_cup_onto_the_red_cup": ("pick the white cup", "stack the white cup on the red cup"),
}


def update_h5_language(h5_path: Path, new_instruction: str, dry_run: bool = False) -> tuple:
    """
    Update current_task and new_tasks in H5 file.

    Args:
        h5_path: Path to trajectory.h5 file
        new_instruction: New language instruction to set
        dry_run: If True, only read and report, don't modify

    Returns:
        Tuple of (old_value, new_value)
    """
    with h5py.File(h5_path, 'r') as f:
        old_task = f.attrs.get('current_task', 'N/A')
        if isinstance(old_task, bytes):
            old_task = old_task.decode('utf-8')

    if not dry_run:
        with h5py.File(h5_path, 'a') as f:
            f.attrs['current_task'] = new_instruction
            f.attrs['new_tasks'] = [new_instruction]

    return (old_task, new_instruction)


def process_data_dir(data_dir: Path, dry_run: bool = False) -> dict:
    """
    Process all task folders in data_dir.

    Args:
        data_dir: Path to data directory (e.g., .../success/2026-01-24)
        dry_run: If True, only preview changes

    Returns:
        Stats dict with counts
    """
    stats = {'updated': 0, 'skipped': 0, 'warnings': 0}

    # Find all task folders
    task_folders = [d for d in data_dir.iterdir() if d.is_dir()]

    for task_folder in sorted(task_folders):
        task_name = task_folder.name

        # Skip debug folder
        if task_name == 'debug':
            continue

        # Check if task is in mapping
        if task_name not in LANGUAGE_MAPPING:
            print(f"\nWARNING: Task '{task_name}' not in LANGUAGE_MAPPING, skipping")
            stats['warnings'] += 1
            continue

        pick_instruction, place_instruction = LANGUAGE_MAPPING[task_name]

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"  Pick instruction:  '{pick_instruction}'")
        print(f"  Place instruction: '{place_instruction}'")
        print(f"{'='*60}")

        # Find all trajectory folders (timestamps)
        trajectory_folders = [d for d in task_folder.iterdir() if d.is_dir()]

        for traj_folder in sorted(trajectory_folders):
            pick_h5 = traj_folder / "pick" / "trajectory.h5"
            place_h5 = traj_folder / "place" / "trajectory.h5"

            # Process pick
            if pick_h5.exists():
                old, new = update_h5_language(pick_h5, pick_instruction, dry_run)
                status = "[DRY RUN]" if dry_run else "[UPDATED]"
                print(f"  {status} {traj_folder.name}/pick: '{old}' -> '{new}'")
                stats['updated'] += 1
            else:
                print(f"  [SKIP] {traj_folder.name}/pick: not found")
                stats['skipped'] += 1

            # Process place
            if place_h5.exists():
                old, new = update_h5_language(place_h5, place_instruction, dry_run)
                status = "[DRY RUN]" if dry_run else "[UPDATED]"
                print(f"  {status} {traj_folder.name}/place: '{old}' -> '{new}'")
                stats['updated'] += 1
            else:
                print(f"  [SKIP] {traj_folder.name}/place: not found")
                stats['skipped'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Update language instructions in pick/place H5 files"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/yygx/PANDA/droid/data/success/2026-01-24',
        help='Data directory containing task folders'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print(f"Data directory: {data_dir}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***")

    stats = process_data_dir(data_dir, args.dry_run)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Warnings: {stats['warnings']}")

    if args.dry_run:
        print("\n*** DRY RUN COMPLETE - Run without --dry_run to apply changes ***")


if __name__ == '__main__':
    main()
