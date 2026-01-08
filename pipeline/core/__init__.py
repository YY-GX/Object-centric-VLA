"""
Core modules for real robot deployment pipeline.
"""

# Original modules
from .object_pose_client import ObjectPoseClient
from .target_pose_calculator import TargetPoseCalculator
from .motion_planner import MotionPlanner
from .success_checker import SuccessChecker
from .vla_client import VLAClient
from .yoloe_obj_detector_client import YOLOEObjectDetectorClient

# New modular components
from .camera_utils import (
    get_camera_data,
    transform_pose_camera_to_base,
    get_camera_intrinsics
)
from .visual_reference_manager import (
    register_3rd_view_references,
    register_wrist_view_references
)
from .object_registration import run_registration_phase
from .vla_executor import execute_vla_skill, prepare_vla_observation
from .retry_handlers import retry_pick_skill, retry_place_skill
from .distractor_masking import apply_distractor_rectangle_masking
from .skill_helpers import (
    get_tracked_objects_for_skill,
    get_distractor_objects,
    should_keep_tracking
)

__all__ = [
    # Original modules
    "ObjectPoseClient",
    "TargetPoseCalculator",
    "MotionPlanner",
    "SuccessChecker",
    "VLAClient",
    "YOLOEObjectDetectorClient",
    # Camera utilities
    "get_camera_data",
    "transform_pose_camera_to_base",
    "get_camera_intrinsics",
    # Visual reference registration
    "register_3rd_view_references",
    "register_wrist_view_references",
    # Object registration
    "run_registration_phase",
    # VLA execution
    "execute_vla_skill",
    "prepare_vla_observation",
    # Retry handlers
    "retry_pick_skill",
    "retry_place_skill",
    # Distractor masking
    "apply_distractor_rectangle_masking",
    # Skill helpers
    "get_tracked_objects_for_skill",
    "get_distractor_objects",
    "should_keep_tracking",
]
