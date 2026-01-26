#!/usr/bin/env python3
"""
Move pick/place folders from compose_2_skills to skill-specific folders.

Moves:
- compose_2_skills/{task}/{traj}/pick  -> {pick_skill_folder}/{traj}/
- compose_2_skills/{task}/{traj}/place -> {place_skill_folder}/{traj}/

Usage:
    python move_pick_place_folders.py --dry_run  # Preview moves
    python move_pick_place_folders.py            # Execute moves
"""

import argparse
import shutil
from pathlib import Path


# Mapping: compose_2_skills task name -> (pick_destination_folder, place_destination_folder)
MAPPING = {
    "hang_the_tape_onto_the_holder_tree": ("pick_the_tape", "hang_tape_on_the_holder_tree"),
    "put_the_cup_in_the_basket": ("pick_the_red_cup", "place_the_red_cup_into_the_basket"),
    "put_the_mustard_in_the_basket": ("pick_the_yellow_mustard", "place_the_yellow_mustard_into_the_basket"),
    "stack_the_cups": ("pick_the_red_cup", "stack_the_red_cup_on_the_red_cup"),
    "stack_the_white_cup_onto_the_red_cup": ("pick_the_white_cup", "stack_the_white_cup_on_the_red_cup"),
}


def move_folders(base_dir: Path, dry_run: bool = False) -> dict:
    """
    Move pick/place folders from compose_2_skills to skill folders.

    Args:
        base_dir: Base directory containing compose_2_skills and skill folders
        dry_run: If True, only preview moves without executing

    Returns:
        Stats dict with move counts
    """
    compose_dir = base_dir / "compose_2_skills"
    stats = {'moved': 0, 'skipped': 0, 'errors': 0}

    if not compose_dir.exists():
        print(f"Error: compose_2_skills directory not found: {compose_dir}")
        return stats

    for task_name, (pick_dest, place_dest) in MAPPING.items():
        task_dir = compose_dir / task_name

        if not task_dir.exists():
            print(f"\nWARNING: Task folder not found: {task_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"  Pick  -> {pick_dest}")
        print(f"  Place -> {place_dest}")
        print(f"{'='*60}")

        # Verify destination folders exist
        pick_dest_dir = base_dir / pick_dest
        place_dest_dir = base_dir / place_dest

        if not pick_dest_dir.exists():
            print(f"  ERROR: Pick destination not found: {pick_dest_dir}")
            stats['errors'] += 1
            continue

        if not place_dest_dir.exists():
            print(f"  ERROR: Place destination not found: {place_dest_dir}")
            stats['errors'] += 1
            continue

        # Process each trajectory folder
        for traj_folder in sorted(task_dir.iterdir()):
            if not traj_folder.is_dir():
                continue

            traj_name = traj_folder.name  # e.g., "Sat_Jan_24_20:03:05_2026"

            # Move pick folder
            src_pick = traj_folder / "pick"
            dst_pick = pick_dest_dir / traj_name

            if src_pick.exists():
                status = "[DRY RUN]" if dry_run else "[MOVED]"
                print(f"  {status} {traj_name}/pick -> {pick_dest}/{traj_name}")

                if not dry_run:
                    try:
                        shutil.move(str(src_pick), str(dst_pick))
                        stats['moved'] += 1
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        stats['errors'] += 1
                else:
                    stats['moved'] += 1
            else:
                print(f"  [SKIP] {traj_name}/pick - not found")
                stats['skipped'] += 1

            # Move place folder
            src_place = traj_folder / "place"
            dst_place = place_dest_dir / traj_name

            if src_place.exists():
                status = "[DRY RUN]" if dry_run else "[MOVED]"
                print(f"  {status} {traj_name}/place -> {place_dest}/{traj_name}")

                if not dry_run:
                    try:
                        shutil.move(str(src_place), str(dst_place))
                        stats['moved'] += 1
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        stats['errors'] += 1
                else:
                    stats['moved'] += 1
            else:
                print(f"  [SKIP] {traj_name}/place - not found")
                stats['skipped'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Move pick/place folders from compose_2_skills to skill folders"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/home/yygx/PANDA/data/yue_dataset/2026-01-24',
        help='Base directory containing compose_2_skills and skill folders'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview moves without executing'
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    print(f"Base directory: {base_dir}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be moved ***")

    stats = move_folders(base_dir, args.dry_run)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Moved: {stats['moved']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")

    if args.dry_run:
        print("\n*** DRY RUN COMPLETE - Run without --dry_run to execute moves ***")


if __name__ == '__main__':
    main()
