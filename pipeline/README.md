# Real Robot Deployment Pipeline

Modular pipeline for executing long-horizon manipulation tasks on real robot using VLA (Vision-Language-Action) models.

**Based on**: LIBERO simulation pipeline (`scripts/phase3/pipeline/evaluation/evaluate_above.py`)

**Status**: ✅ Fully implemented with mock object poses (ready for vision integration)

---

## 📁 Directory Structure

```
pipeline/
├── config/
│   ├── task_config.json           # Task and skill definitions
│   └── real_robot_config.json     # Hardware and deployment config
│
├── core/
│   ├── object_pose_server.py      # Object pose tracking (mock + placeholder for real)
│   ├── target_pose_calculator.py  # Calculate above/approach poses
│   ├── motion_planner.py          # Motion planning and execution
│   ├── success_checker.py         # Skill success verification
│   └── vla_client.py              # VLA policy server client
│
├── utils/
│   ├── task_planner.py            # Task planning utilities
│   ├── coordinate_transforms.py   # Camera ↔ robot transforms (placeholder)
│   └── logging_utils.py           # Video recording and CSV logging
│
├── main_pipeline.py               # Main orchestration script
├── prompts/
│   └── ARCHITECTURE.md            # Architecture design document
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Prerequisites

**Required**:
- Python 3.10+
- OpenVLA-OFT trained checkpoint
- VLA policy server running
- DROID robot environment setup

**Python packages**:
```bash
pip install numpy scipy opencv-python
pip install openpi_client  # For VLA WebSocket client
```

### 2. Configuration

**Edit `config/task_config.json`**:
- Define your tasks and skills
- Specify target objects for each skill

**Edit `config/real_robot_config.json`**:
- Set camera IDs
- Configure VLA server address
- Set mock poses (or enable real object pose server)

### 3. Start VLA Server

On GPU server:
```bash
python scripts/phase4_real_robot/server/serve_openvla_oft.py \
    --checkpoint_path runs/your_checkpoint \
    --port 8008
```

### 4. Run Pipeline

On local robot laptop:
```bash
cd scripts/phase4_real_robot/local_scripts/pipeline

python main_pipeline.py \
    --task_name "pick the black bowl" \
    --max_retries 3
```

---

## 📖 Usage Examples

### Example 1: Single Object Pick

**Task**: Pick a black bowl

**Config** (`config/task_config.json`):
```json
{
  "long_horizon_tasks": [
    {
      "task_id": 1,
      "name": "pick the black bowl",
      "skills": ["pick black bowl"]
    }
  ],
  "skill_mappings": {
    "pick black bowl": {
      "target_object": "black_bowl",
      "language": "pick black bowl",
      "skill_type": "pick",
      "success_check": "object_lifted"
    }
  }
}
```

**Run**:
```bash
python main_pipeline.py --task_name "pick the black bowl"
```

### Example 2: Pick and Place

**Task**: Pick red cup and place on plate

**Config**:
```json
{
  "long_horizon_tasks": [
    {
      "task_id": 2,
      "name": "pick red cup and place on plate",
      "skills": [
        "pick red cup",
        "place red cup on plate"
      ]
    }
  ],
  "skill_mappings": {
    "pick red cup": {
      "target_object": "red_cup",
      "language": "pick red cup",
      "skill_type": "pick"
    },
    "place red cup on plate": {
      "target_object": "red_cup",
      "target_surface": "plate",
      "language": "place red cup on plate",
      "skill_type": "place"
    }
  }
}
```

**Run**:
```bash
python main_pipeline.py --task_name "pick red cup and place on plate" --max_retries 5
```

---

## 🔧 Configuration Guide

### Task Configuration (`config/task_config.json`)

**Structure**:
```json
{
  "long_horizon_tasks": [
    {
      "task_id": <int>,
      "name": "<task_name>",
      "skills": ["<skill_1>", "<skill_2>", ...]
    }
  ],
  "skill_mappings": {
    "<skill_name>": {
      "target_object": "<object_name>",
      "language": "<vla_instruction>",
      "skill_type": "pick|place|open|close",
      "success_check": "<check_type>"
    }
  }
}
```

**Fields**:
- `task_id`: Unique integer ID
- `name`: Task name (used in `--task_name` argument)
- `skills`: Ordered list of skill names
- `target_object`: Object name for pose tracking
- `language`: VLA language instruction
- `skill_type`: Skill category (determines success checking)

### Robot Configuration (`config/real_robot_config.json`)

**Key sections**:

1. **Cameras**: Camera serial IDs
   ```json
   "cameras": {
     "left_camera_id": "36087771",
     "wrist_camera_id": "16478870"
   }
   ```

2. **VLA Server**: Remote server address
   ```json
   "vla_server": {
     "host": "192.168.1.100",
     "port": 8008
   }
   ```

3. **Mock Mode**: Use mock poses for testing
   ```json
   "mock_mode": {
     "enabled": true,
     "mock_poses": {
       "black_bowl": {
         "position": [0.4, 0.0, 0.1],
         "quaternion": [1.0, 0.0, 0.0, 0.0]
       }
     }
   }
   ```

4. **Motion Planning**: Control parameters
   ```json
   "motion_planning": {
     "control_frequency": 15,
     "position_threshold": 0.02,
     "max_retries": 3
   }
   ```

5. **Above Pose**: Height parameters
   ```json
   "above_pose": {
     "default_height": 0.10,
     "object_specific": {
       "drawer": {"above_height": 0.15}
     }
   }
   ```

---

## 🏗️ Pipeline Execution Flow

```
For each skill in task:
    │
    ├─► 1. Get object pose
    │   └─► ObjectPoseServer.wait_for_stable_pose()
    │
    ├─► 2. Calculate above pose
    │   └─► TargetPoseCalculator.calculate_above_pose()
    │
    ├─► 3. Move to above pose
    │   └─► MotionPlanner.move_to_pose()
    │
    ├─► 4. Execute VLA skill
    │   ├─► VLAClient.predict() [query server]
    │   └─► RobotEnv.step() [execute actions]
    │
    ├─► 5. Move EE up
    │   └─► MotionPlanner.move_ee_up()
    │
    ├─► 6. Check skill success
    │   └─► SuccessChecker.check_skill_success()
    │
    └─► 7. Retry on failure (up to max_retries)
```

---

## 🔍 Module Details

### ObjectPoseServer

**Purpose**: Track object poses in real-time

**Modes**:
- **Mock mode** (current): Returns pre-defined poses from config
- **Real mode** (future): Queries YOLOE + FoundationPose server

**API**:
```python
# Get object pose
pose = server.get_object_pose("black_bowl")
# Returns: {"position": [x,y,z], "quaternion": [w,x,y,z], "confidence": float}

# Wait for stable pose
stable_pose = server.wait_for_stable_pose("red_cup", num_samples=5)
```

**Integration TODO**: Replace `_query_real_server()` with actual YOLOE + FoundationPose API calls

### TargetPoseCalculator

**Purpose**: Calculate target EE poses from object poses

**Features**:
- Object-specific handling (drawers, stove, etc.)
- Configurable above height
- Random perturbation for retries
- Approach trajectory generation

**API**:
```python
# Calculate above pose
above_pose = calculator.calculate_above_pose(
    object_pose, "black_bowl", above_height=0.10, skill_type="pick"
)

# Add perturbation for retry
perturbed = calculator.add_random_perturbation(above_pose, xy_range=0.02)
```

### MotionPlanner

**Purpose**: Execute robot motions

**Features**:
- Cartesian position control
- Convergence monitoring
- Vertical lifting motions
- Current pose queries

**API**:
```python
# Move to pose
result = planner.move_to_pose(target_pose, blocking=True)
# Returns: {"success": bool, "final_distance": float, ...}

# Lift EE
success = planner.move_ee_up(lift_distance=0.05)
```

### SuccessChecker

**Purpose**: Verify skill completion

**Methods**:
- **Pick**: Height increase > threshold
- **Place**: Object lowered and on surface
- **Open/Close**: Joint angle change (placeholder)
- **Turn on/off**: Visual detection (placeholder)

**API**:
```python
result = checker.check_skill_success(
    skill_name="pick black bowl",
    skill_type="pick",
    target_object="black_bowl",
    initial_state=initial_pose
)
# Returns: {"success": bool, "confidence": float, "reason": str, "metrics": dict}
```

### VLAClient

**Purpose**: Query VLA policy server

**Features**:
- WebSocket communication
- Image resizing (224x224)
- Action chunk prediction

**API**:
```python
# Predict action chunk
action_chunk = client.predict(
    observations={"left_image": ..., "wrist_image": ..., ...},
    language_instruction="pick black bowl",
    open_loop_horizon=8
)
# Returns: np.ndarray shape (8, 7) = [dx, dy, dz, droll, dpitch, dyaw, gripper]
```

---

## 📊 Output Files

**After execution**, find outputs in `outputs/<task_name>_<timestamp>/`:

```
outputs/pick_the_black_bowl_20260103_123045/
├── videos/
│   └── task_pick_the_black_bowl.mp4         # Side-by-side camera views
├── task_pick_the_black_bowl_actions.csv     # Action log
└── task_pick_the_black_bowl_states.csv      # State log
```

**CSV Format**:

`actions.csv`:
```
timestep,dx,dy,dz,droll,dpitch,dyaw,gripper
0,0.1,0.0,-0.05,0.0,0.0,0.0,0.0
1,0.08,0.01,-0.04,0.0,0.0,0.0,0.0
...
```

`states.csv`:
```
timestep,x,y,z,roll,pitch,yaw,gripper
0,0.40,0.00,0.20,0.0,0.0,0.0,0.0
1,0.41,0.00,0.19,0.0,0.0,0.0,0.0
...
```

---

## 🐛 Troubleshooting

### Issue: VLA server connection failed

**Solution**:
1. Check VLA server is running: `ps aux | grep serve_openvla_oft`
2. Verify host/port in `config/real_robot_config.json`
3. Test connection: `ping <vla_server_host>`

### Issue: Object not detected

**Solution**:
1. If using mock mode: Check object name matches `mock_poses` in config
2. If using real server: Verify object pose server is running
3. Check object is in camera view

### Issue: Motion planner fails to reach pose

**Solution**:
1. Check target pose is within robot workspace
2. Adjust `position_threshold` in config (increase for looser tolerance)
3. Increase `max_steps_per_motion` in config
4. Enable retries: `--max_retries 5`

### Issue: Success checker always fails

**Solution**:
1. Check initial object state is stored (pick requires initial state)
2. Adjust success thresholds in `config/real_robot_config.json`:
   - `pick.min_height_increase` (default 0.03m)
   - `place.xy_distance_threshold` (default 0.05m)
3. Enable debug mode: Add `print()` statements in `core/success_checker.py`

---

## 🔮 Future Integration

### YOLOE + FoundationPose Server

**Location**: `core/object_pose_server.py` → `_query_real_server()`

**Steps**:
1. Start YOLOE + FoundationPose server on local machine
2. Implement HTTP/WebSocket API client in `_query_real_server()`
3. Add coordinate transform from camera to robot base
4. Set `mock_mode.enabled = false` in config
5. Test with real object detection

**Example integration**:
```python
def _query_real_server(self, object_name: str) -> Optional[Dict]:
    import requests

    url = f"http://{self.server_url}:{self.port}/detect_object"
    response = requests.post(url, json={"object_name": object_name})

    if response.status_code == 200:
        data = response.json()

        # Transform from camera to robot frame
        pose_camera = np.array(data["position"])
        pose_robot = camera_to_robot_frame(pose_camera, CAMERA_POSE)

        return {
            "position": pose_robot,
            "quaternion": np.array(data["quaternion"]),
            "confidence": data["confidence"],
            "timestamp": time.time()
        }
    else:
        return None
```

### Camera Calibration

**Location**: `utils/coordinate_transforms.py`

**Steps**:
1. Calibrate camera intrinsics (focal length, principal point)
2. Calibrate camera extrinsics (pose relative to robot base)
3. Update `get_default_camera_pose()` and `get_default_camera_intrinsics()`
4. Test transforms with known 3D points

---

## 📚 References

- **LIBERO simulation pipeline**: `scripts/phase3/pipeline/evaluation/evaluate_above.py`
- **DROID data collection**: `externals/droid/droid/trajectory_utils/misc.py`
- **OpenVLA-OFT evaluation**: `scripts/phase4_real_robot/local_scripts/evaluation/evaluate_openvla_oft.py`
- **Architecture design**: `prompts/ARCHITECTURE.md`

---

## 🤝 Contributing

To add a new task:
1. Define task in `config/task_config.json`
2. Add skills to `skill_mappings` (if new)
3. Run pipeline: `python main_pipeline.py --task_name "your task"`

To add a new skill type:
1. Add skill to `config/task_config.json`
2. Implement success checking in `core/success_checker.py`
3. (Optional) Add object-specific handling in `core/target_pose_calculator.py`

---

## 📄 License

This pipeline is part of the OpenVLA-OFT project.

---

## ✉️ Contact

For questions or issues, please refer to the main project documentation.
