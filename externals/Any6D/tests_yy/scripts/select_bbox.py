#!/usr/bin/env python3
"""
Interactive bounding box selection for SAM2 segmentation.

Usage:
    python select_bbox.py --image ../data/1st_frame/left000000.png
"""

import cv2
import argparse
from pathlib import Path


def select_bbox_interactive(image_path):
    """
    Open image and let user select bounding box with mouse.

    Returns:
        tuple: (xmin, ymin, xmax, ymax)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    window_name = 'Select Object Bounding Box (Press SPACE/ENTER to confirm, C to cancel)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    bbox = cv2.selectROI(window_name, img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if bbox[2] == 0 or bbox[3] == 0:
        print("Selection cancelled or invalid.")
        return None

    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height

    return int(xmin), int(ymin), int(xmax), int(ymax)


def main():
    parser = argparse.ArgumentParser(description="Select bounding box for SAM2 segmentation")
    parser.add_argument('--image', required=True, help='Path to image')
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Opening image: {image_path}")
    print("\nInstructions:")
    print("1. Click and drag to draw a bounding box around your object")
    print("2. Press SPACE or ENTER to confirm")
    print("3. Press C to cancel and retry")
    print()

    bbox = select_bbox_interactive(image_path)

    if bbox:
        xmin, ymin, xmax, ymax = bbox
        print(f"\n✓ Bounding box selected: [{xmin}, {ymin}, {xmax}, {ymax}]")
        print(f"\nUse this command to prepare data with SAM2:")
        print(f"python prepare_data.py --frame <frame_name> --camera left \\")
        print(f"    --bbox {xmin} {ymin} {xmax} {ymax}")
    else:
        print("\n✗ No bounding box selected")


if __name__ == '__main__':
    main()
