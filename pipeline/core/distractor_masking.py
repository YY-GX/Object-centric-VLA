#!/usr/bin/env python3
"""
Distractor Masking for Real Robot Pipeline.

Provides functions for detecting and masking distractor objects
in wrist camera images during VLA execution.
"""

import cv2
import numpy as np


def apply_distractor_rectangle_masking(
    image: np.ndarray,
    distractor_mask: np.ndarray,
    max_rectangles: int = 5
) -> np.ndarray:
    """
    Apply rectangle masking to distractor regions in image.

    Finds connected components in distractor_mask, sorts by area (largest first),
    and masks up to max_rectangles components with their bounding boxes.

    Args:
        image: Input image (H, W, 3) uint8
        distractor_mask: Binary mask of distractor regions (H, W) uint8
        max_rectangles: Maximum number of rectangles to mask (default 5)

    Returns:
        Masked image with rectangular black regions over distractors
    """
    if distractor_mask is None or distractor_mask.sum() == 0:
        return image

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        distractor_mask, connectivity=8
    )

    if num_labels <= 1:  # Only background
        return image

    # Get component areas (skip background label 0)
    # stats columns: [x, y, width, height, area]
    component_info = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        component_info.append({
            'label': label_id,
            'area': area,
            'bbox': (x, y, x + w, y + h)
        })

    # Sort by area (largest first)
    component_info.sort(key=lambda c: c['area'], reverse=True)

    # Take top N components
    selected = component_info[:max_rectangles]

    # Apply rectangle masking
    masked_image = image.copy()
    for comp in selected:
        x1, y1, x2, y2 = comp['bbox']
        masked_image[y1:y2, x1:x2] = 0

    return masked_image
