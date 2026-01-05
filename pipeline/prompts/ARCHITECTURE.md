# Real Robot Deployment Pipeline - Architecture Design

Based on the LIBERO simulation pipeline (`scripts/phase3/pipeline/evaluation/evaluate_above.py`) and the requirements from `prompts/for_ai/real_robot/initial_plan.md`, here's the complete modular architecture for `scripts/phase4_real_robot/local_scripts/pipeline`:

## 📁 File Structure

```
scripts/phase4_real_robot/local_scripts/pipeline/
├── config/
│   ├── task_config.json                 # Task definitions and skill sequences
│   └── real_robot_config.json           # Hardware config (camera IDs, pose server URL, etc.)
│
├── core/
│   ├── __init__.py
│   ├── object_pose_server.py            # Object pose tracking server/client
│   ├── target_pose_calculator.py        # Calculate above/approach poses
│   ├── motion_planner.py                # Motion planning and execution
│   ├── success_checker.py               # Skill success verification
│   └── vla_client.py                    # VLA policy server client
│
├── utils/
│   ├── __init__.py
│   ├── task_planner.py                  # Load task config and plan skill sequences
│   ├── coordinate_transforms.py         # Camera frame ↔ robot base frame transforms
│   └── logging_utils.py                 # Logging and video recording
│
├── main_pipeline.py                     # Main orchestration script
├── prompts/
│   └── ARCHITECTURE.md                  # This file
└── README.md                            # Documentation
```

---

## 📝 Module Specifications

### 1. config/task_config.json

**Purpose**: Define long-horizon tasks and atomic skills with their target objects

**Structure**:
```json
{
  "long_horizon_tasks": [
    {
      "task_id": 1,
      "name": "pick the black bowl",
      "skills": ["pick black bowl"]
    },
    {
      "task_id": 2,
      "name": "pick red cup and place on plate",
      "skills": ["pick red cup", "place red cup on plate"]
    }
  ],
  "skill_mappings": {
    "pick black bowl": {
      "target_object": "black_bowl",
      "language": "pick black bowl",
      "success_check": "object_lifted"
    },
    "place red cup on plate": {
      "target_object": "red_cup",
      "language": "place red cup on plate",
      "success_check": "object_on_target"
    }
  }
}
```

---

### 2. config/real_robot_config.json

**Purpose**: Hardware and deployment configuration

**Structure**:
```json
{
  "cameras": {
    "left_camera_id": "36087771",
    "wrist_camera_id": "16478870"
  },
  "object_pose_server": {
    "host": "localhost",
    "port": 5000
  },
  "vla_server": {
    "host": "152.2.134.113",
    "port": 8008
  },
  "motion_planning": {
    "control_frequency": 15,
    "position_threshold": 0.02,
    "max_retries": 3
  },
  "above_pose": {
    "default_height": 0.10,
    "min_height": 0.05,
    "max_height": 0.20
  }
}
```

---

### 3. core/object_pose_server.py

**Purpose**: Interface to object pose tracking system (placeholder for YOLOE + FoundationPose)

**API**:
```python
class ObjectPoseServer:
    def __init__(self, server_url: str, port: int)
    def get_object_pose(self, object_name: str) -> Dict
    def wait_for_stable_pose(self, object_name: str, num_samples: int, position_threshold: float) -> Dict
```

**Input**: `object_name: str`

**Output**:
```python
{
    "position": np.ndarray([x, y, z]),      # meters, robot base frame
    "quaternion": np.ndarray([w, x, y, z]), # orientation
    "confidence": float,                     # 0-1
    "timestamp": float                       # unix timestamp
}
```

**Logic**:
1. Send HTTP/WebSocket request with object_name
2. Receive detection results (bbox, segmentation, 6D pose)
3. Transform from camera frame to robot base frame
4. Return pose dict

**Note**: Placeholder implementation returns mock poses

---

### 4. core/target_pose_calculator.py

**Purpose**: Calculate target EE poses (above pose, approach pose) from object pose

**API**:
```python
class TargetPoseCalculator:
    def __init__(self, config: Dict)
    def calculate_above_pose(self, object_pose: Dict, object_name: str, above_height: float, skill_type: str) -> Dict
    def calculate_approach_trajectory(self, current_pose: Dict, target_pose: Dict, num_waypoints: int) -> List[Dict]
```

**Input**: `object_pose: Dict`, `above_height: float`, `skill_type: str`

**Output**:
```python
{
    "position": np.ndarray([x, y, z]),
    "orientation": np.ndarray([w, x, y, z]),
    "approach_waypoints": List[np.ndarray]  # optional
}
```

**Logic** (from LIBERO's above_pose_calculator.py):
1. Extract object position [x, y, z]
2. Apply object-specific handling (drawers, stove, etc.)
3. Calculate above position: [x, y, z + above_height]
4. Set orientation to downward-facing
5. Return target_pose dict

---

### 5. core/motion_planner.py

**Purpose**: Execute robot motions to target poses

**API**:
```python
class MotionPlanner:
    def __init__(self, robot_env, config: Dict)
    def move_to_pose(self, target_pose: Dict, blocking: bool, max_retries: int) -> Dict
    def move_ee_up(self, lift_distance: float, skill_type: str) -> bool
    def get_current_ee_pose(self) -> Dict
```

**Input**: `target_pose: Dict`, `blocking: bool`, `max_retries: int`

**Output**:
```python
{
    "success": bool,
    "final_distance": float,           # meters
    "final_orientation_error": float,  # radians
    "execution_time": float            # seconds
}
```

**Logic**:
1. Use robot IK solver to compute joint trajectory
2. Execute trajectory using robot_env.step()
3. Monitor execution (position/orientation error)
4. Retry on failure with perturbation
5. Return success status + final errors

---

### 6. core/success_checker.py

**Purpose**: Verify atomic skill completion

**API**:
```python
class SuccessChecker:
    def __init__(self, config: Dict, object_pose_server: ObjectPoseServer)
    def check_skill_success(self, skill_name: str, skill_type: str, target_object: str, initial_state: Dict) -> Dict
    def wait_for_object_settle(self, object_name: str, timeout: float) -> bool
```

**Input**: `skill_name: str`, `skill_type: str`, `target_object: str`

**Output**:
```python
{
    "success": bool,
    "confidence": float,  # 0-1
    "reason": str,
    "metrics": dict       # e.g., height_change, distance_to_target
}
```

**Logic** (skill-type specific):
- **PICK**: Check z-height increase > 0.03m
- **PLACE**: Check object on surface (XY distance, Z-height match)
- **OPEN/CLOSE**: Check joint angle (requires articulated tracking)
- **TURN_ON/OFF**: Visual detection (placeholder)

---

### 7. core/vla_client.py

**Purpose**: Interface to VLA policy server

**API**:
```python
class VLAClient:
    def __init__(self, server_host: str, server_port: int)
    def predict(self, observations: Dict, language_instruction: str, open_loop_horizon: int) -> np.ndarray
    def close(self)
```

**Input**: `observations: Dict`, `language_instruction: str`

**Output**: `action_chunk: np.ndarray` - Shape [H, 7] where 7 = [dx, dy, dz, droll, dpitch, dyaw, gripper]

**Logic** (from evaluate_openvla_oft.py):
1. Prepare request: resize images, add proprio state, language
2. Send WebSocket request to VLA server
3. Receive action chunk [H, 7]
4. Return action chunk

---

### 8. utils/task_planner.py

**Purpose**: Load task config and return skill sequences

**API**:
```python
def load_task_config(config_path: str) -> Dict
def plan_task_sequence(task_name: str, config: Dict) -> List[str]
def get_skill_info(skill_name: str, config: Dict) -> Dict
```

**Input**: `task_name: str`

**Output**: `skill_sequence: List[str]`, `task_info: Dict`

**Logic**:
1. Load task_config.json
2. Find task by name (case-insensitive)
3. Return skill sequence

---

### 9. utils/coordinate_transforms.py

**Purpose**: Transform coordinates between camera and robot frames

**API**:
```python
def camera_to_robot_frame(point_camera: np.ndarray, camera_pose: Dict) -> np.ndarray
def pixel_to_3d_point(pixel_coords: np.ndarray, depth: float, camera_intrinsics: Dict) -> np.ndarray
```

**Note**: Placeholder - requires camera calibration

---

### 10. utils/logging_utils.py

**Purpose**: Video recording and logging

**API**:
```python
class VideoRecorder:
    def __init__(self, save_dir: Path, fps: int)
    def add_frame(self, left_image: np.ndarray, wrist_image: np.ndarray)
    def save(self, filename: str)

class CSVLogger:
    def __init__(self, save_path: Path)
    def log_step(self, timestep: int, action: np.ndarray, state: Dict)
    def close(self)
```

---

### 11. main_pipeline.py

**Purpose**: Main orchestration script

**API**:
```python
class RealRobotPipeline:
    def __init__(self, task_name: str, config_path: str)
    def execute_task(self, max_retries: int) -> bool
    def execute_single_skill(self, skill_name: str, max_retries: int) -> Dict
    def recovery_from_failure(self, failed_skill: str, previous_skill: str) -> bool
```

**Execution Flow** (similar to evaluate_above.py):
```
For each skill in skill_sequence:
    1. Get object pose (object_pose_server)
    2. Calculate above pose (target_pose_calculator)
    3. Move to above pose (motion_planner)
    4. Execute VLA skill (vla_client + robot_env.step)
    5. Move EE up 5cm (motion_planner)
    6. Check skill success (success_checker)
    7. Handle failure: Recovery or retry
    8. If success: Next skill
```

---

## 🔄 Execution Flow Diagram

```
main_pipeline.py
    │
    ├─► task_planner.py
    │   └─► Load task config → Get skill sequence
    │
    └─► For each skill:
        │
        ├─► object_pose_server.py
        │   └─► Get object pose (real-time tracking)
        │
        ├─► target_pose_calculator.py
        │   └─► Calculate above pose from object pose
        │
        ├─► motion_planner.py
        │   └─► Execute motion to above pose
        │
        ├─► vla_client.py
        │   └─► Execute VLA skill (query policy, execute actions)
        │
        ├─► motion_planner.py
        │   └─► Move EE up 5cm
        │
        ├─► success_checker.py
        │   └─► Verify skill completion
        │
        └─► logging_utils.py
            └─► Record video, save logs
```

---

## 🆚 Key Differences from LIBERO Simulation

| Aspect | LIBERO Simulation | Real Robot |
|--------|------------------|------------|
| **Object Poses** | Ground-truth from simulator | YOLOE + FoundationPose (vision-based) |
| **Motion Planning** | MPlib (collision-aware) | Cartesian velocity control (simple) |
| **Success Checking** | BDDL predicates (exact) | Heuristics (height, position tracking) |
| **VLA Execution** | Local inference | Remote server via WebSocket |
| **Coordinate Frames** | Simulation frame | Camera → Robot base transform needed |
| **Environment** | LIBERO env (MuJoCo) | DROID RobotEnv (Franka hardware) |

---

## 📦 Dependencies

- `numpy`, `scipy` - Math operations
- `opencv-python` - Video recording
- `requests` or `websocket-client` - Server communication
- `openpi_client` - VLA policy client
- `droid` - Robot environment

---

## 🚀 Usage Example

```bash
# On local robot laptop:
python scripts/phase4_real_robot/local_scripts/pipeline/main_pipeline.py \
    --task_name "pick the black bowl" \
    --config scripts/phase4_real_robot/local_scripts/pipeline/config/task_config.json \
    --max_retries 3 \
    --save_video
```

---

## 📋 Implementation Priority

1. **Phase 1 - Core Infrastructure**:
   - task_planner.py, target_pose_calculator.py, motion_planner.py, vla_client.py, main_pipeline.py

2. **Phase 2 - Placeholder Modules**:
   - object_pose_server.py (mock), success_checker.py (heuristics), logging_utils.py

3. **Phase 3 - Integration**:
   - Test end-to-end with mock poses

4. **Phase 4 - Vision Integration** (Future):
   - Integrate YOLOE + FoundationPose
