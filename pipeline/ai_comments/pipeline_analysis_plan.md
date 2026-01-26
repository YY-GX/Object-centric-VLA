# PANDA Pipeline Analysis & Answers

## Question 1: Pipeline Logic Flow

When you run `python main_pipeline.py --task_name "pick the red cup"`:

### Initialization Phase
1. **Load configs** - `task_config.json` and `real_robot_config.json`
2. **Plan task sequence** - Maps "pick the red cup" → `["pick red cup"]`, extracts `scene_objects: ["red_cup"]`
3. **Create output directory** - `outputs/pick_the_red_cup_YYYYMMDD_HHMMSS/`
4. **Initialize YOLOE client** (if `is_randome_erasing=true`) - Visual mode (port 5559) or text mode (port 5557)
5. **Initialize modules**:
   - `RobotEnv` - DROID robot interface with depth enabled
   - `ObjectPoseClient` - FoundationPose server (localhost:5558)
   - `TargetPoseCalculator` - Computes above-pose positions
   - `MotionPlanner` - Linear motion to target poses
   - `VLAClient` - VLA server (localhost:8009)
   - `SuccessChecker` - object_lifted / object_on_target checks

### Registration Phase
6. **For each scene object (e.g., "red_cup")**:
   - Capture RGB+depth from left camera
   - Call FoundationPose `register()` with mesh + YOLOE prompt/visual ref
   - Transform pose: camera frame → base frame
   - Show debug images, wait for user approval (y/n)
   - Store in `registration_dict`

### Skill Execution Phase (per skill)
7. **Start tracking** - `pose_client.start_tracking()` with initial pose
8. **Collect initial poses** - Track all objects needed for this skill
9. **Calculate above pose** - Position above target object (Z + mesh_height + 0.2m)
10. **Motion plan to above pose** - Linear motion with gripper open
11. **Execute VLA skill** (main loop, up to 200 steps):
    - Get observation from `robot_env`
    - **Track objects** with FoundationPose
    - **Check success** (early exit if object lifted > 3cm)
    - **Distractor masking** (if enabled) - YOLOE detects distractors, masks them
    - **Prepare VLA observation** - images + cartesian_position (relative or absolute)
    - **Query VLA** every 8 steps → get action chunk (8, 7)
    - **Execute action** - binarize gripper, clip, step robot
    - Sleep to match 15Hz control frequency
12. **Move EE up** - Lift 5cm after VLA execution
13. **Get final poses** - Track all objects post-skill
14. **Check skill success** - `object_lifted` (Z increase > 3cm) or `object_on_target`
15. **Retry on failure** - Up to `max_retries` with recovery logic

### Multi-skill Tasks
For "pick the red cup and place on plate":
- Skills: `["pick red cup", "place red cup on plate"]`
- After pick succeeds, `red_cup` stays tracked
- Place skill tracks both `red_cup` (what's held) and `plate` (target)
- Success: `red_cup` XY distance < 5cm from `plate`

---

## Question 2: Steps After Training New Atomic Skills

After training new skills, you need to add/update:

### A. Configuration Files

**1. `pipeline/config/task_config.json`**
```json
{
  "skill_mappings": {
    "pick new object": {
      "target_object": "new_object",
      "language": "pick the new object.",
      "skill_type": "pick",
      "success_check": "object_lifted"
    }
  },
  "long_horizon_tasks": [
    {
      "task_id": N,
      "name": "pick the new object",
      "skills": ["pick new object"],
      "scene_objects": ["new_object"]
    }
  ]
}
```

**2. `pipeline/config/real_robot_config.json`**
```json
{
  "object_meshes": {
    "new_object": "/home/yygx/PANDA/pipeline/data/meshes/new_object/mesh.obj"
  },
  "yoloe_prompts": {
    "new_object": "description of the new object for YOLOE text mode"
  }
}
```

### B. Mesh Files (if new object)

**Location**: `pipeline/data/meshes/new_object/`
- `mesh.obj` - 3D mesh for FoundationPose (Wavefront OBJ format)
- Optional: `mesh.mtl`, `textures/*.jpg`

### C. Visual Reference Images (if new object)

**Wrist-view refs**: `pipeline/data/yoloe_ref_images/`
- `new_object_0.jpg` through `new_object_4.jpg` (3-5 images, different angles)
- Update `bboxes.json` with bounding boxes: `"new_object_0": [x1, y1, x2, y2]`

**3rd-person refs**: `pipeline/data/yoloe_ref_images/yoloe_ref_images_3rd_view/`
- Same naming convention: `new_object_0.jpg` ... `new_object_4.jpg`
- Update `bboxes.json` in this directory too

### D. Training Data Structure (for reference)

The training data you collected is at `data/yue_dataset/atomic_skills/[skill_name]/`:
```
skill_name/
├── [TIMESTAMP]/
│   ├── trajectory.h5          # HDF5 with robot states/actions
│   ├── segment_info.json      # Skill frame ranges
│   ├── metadata_*.json        # Full metadata
│   └── recordings/MP4/*.mp4   # Camera videos
└── [TIMESTAMP]_random_erased/ # Data augmentation variant
```

### Summary Checklist for New Skill
- [ ] Add mesh: `pipeline/data/meshes/[object]/mesh.obj`
- [ ] Add wrist-view refs: `pipeline/data/yoloe_ref_images/[object]_[0-4].jpg`
- [ ] Add 3rd-view refs: `pipeline/data/yoloe_ref_images/yoloe_ref_images_3rd_view/[object]_[0-4].jpg`
- [ ] Update `bboxes.json` in both ref image directories
- [ ] Add to `real_robot_config.json`: `object_meshes` and `yoloe_prompts`
- [ ] Add to `task_config.json`: `skill_mappings` and `long_horizon_tasks`

---

## Question 3: Adding Pi 0.5 as VLA Option

### Key Differences: OpenVLA-OFT vs Pi 0.5

| Aspect | OpenVLA-OFT (Current) | Pi 0.5 |
|--------|----------------------|--------|
| **Proprioceptive State** | `cartesian_position` (6D) | `joint_position` (7D) |
| **Action Space** | `cartesian_velocity` | `joint_velocity` |
| **Action Dimension** | 7 (6 delta pose + gripper) | 8 (7 joint vel + gripper) |
| **Gripper Logic** | `>0.5 → open (0)` | `>0.5 → open (1)` (inverted!) |
| **Relative Pose** | Supported (ee_pos - obj_pos) | **Not needed** (uses joint angles) |

### Files to Modify

**1. `pipeline/config/real_robot_config.json`**
Add VLA model selection:
```json
{
  "vla_model": "openvla-oft",  // or "pi05"
  "vla_server": {
    "host": "localhost",
    "port": 8009
  }
}
```

**2. `pipeline/core/vla_client.py`**
- Add `model_type` parameter to `VLAClient.__init__`
- Make observation dict construction model-aware:
  - OpenVLA-OFT: `observation/cartesian_position`
  - Pi 0.5: `observation/joint_position`
- Update action shape validation: `7 for OFT, 8 for Pi 0.5`

**3. `pipeline/core/vla_executor.py`**
- Modify `prepare_vla_observation()`:
  - Extract `joint_positions` from `obs["robot_state"]` for Pi 0.5
  - Skip relative pose computation for Pi 0.5 (not needed)
- Update action execution:
  - OFT: `action[6] > 0.5 → gripper=0 (open)`
  - Pi 0.5: `action[7] > 0.5 → gripper=1 (open)` **inverted!**

**4. `pipeline/main_pipeline.py`**
- Pass `action_space` to `RobotEnv`:
  - OFT: `RobotEnv(action_space="cartesian_velocity")`
  - Pi 0.5: `RobotEnv(action_space="joint_velocity")`

### Key Code Changes

**vla_client.py** - Model-aware observation:
```python
def __init__(self, host, port, model_type="openvla-oft"):
    self.model_type = model_type
    # ...

def predict(self, obs, language, horizon):
    if self.model_type == "pi05":
        request_data = {
            "observation/joint_position": obs["joint_position"],
            # ...
        }
    else:  # openvla-oft
        request_data = {
            "observation/cartesian_position": obs["cartesian_position"],
            # ...
        }
```

**vla_executor.py** - Model-aware action handling:
```python
# Gripper binarization
if model_type == "pi05":
    if action[7] > threshold:
        action = np.concatenate([action[:7], np.ones((1,))])   # Open
    else:
        action = np.concatenate([action[:7], np.zeros((1,))])  # Close
else:  # openvla-oft
    if action[6] > threshold:
        action = np.concatenate([action[:6], np.zeros((1,))])  # Open
    else:
        action = np.concatenate([action[:6], np.ones((1,))])   # Close
```

---

## Question 4: Improving Video Saving

### Current Limitation
- Videos only saved during VLA execution (`vla_executor.py` lines 26-109)
- Motion planner phases not recorded
- Only saves what VLA sees (processed images, not raw stream)

### Option A: Separate ZED Recording Thread (Recommended)

**Problem**: ZED SDK allows only ONE process to open each camera at a time.

**Solution**: Use `RobotEnv` as the single camera source, but run a separate thread that continuously captures frames:

```python
# In main_pipeline.py
import threading
import queue

class PipelineVideoRecorder:
    def __init__(self, robot_env, save_dir):
        self.robot_env = robot_env
        self.save_dir = save_dir
        self.frame_queue = queue.Queue()
        self.recording = False
        self._thread = None

    def start_recording(self):
        self.recording = True
        self.frames_3rd = []
        self.frames_wrist = []
        self._thread = threading.Thread(target=self._capture_loop)
        self._thread.start()

    def _capture_loop(self):
        while self.recording:
            obs = self.robot_env.get_observation()
            # Extract and store frames
            left_img = self._extract_left_image(obs)
            wrist_img = self._extract_wrist_image(obs)
            self.frames_3rd.append(left_img)
            self.frames_wrist.append(wrist_img)
            time.sleep(1/15)  # 15 FPS

    def stop_and_save(self, filename_prefix):
        self.recording = False
        self._thread.join()
        # Save two videos
        self._save_video(self.frames_3rd, f"{filename_prefix}_3rd_person.mp4")
        self._save_video(self.frames_wrist, f"{filename_prefix}_wrist.mp4")
```

**Integration in pipeline**:
```python
# At start of execute_task()
video_recorder = PipelineVideoRecorder(self.robot_env, self.run_dir)
video_recorder.start_recording()

# ... all pipeline execution (registration, motion planning, VLA) ...

# At end of execute_task()
video_recorder.stop_and_save(f"full_rollout_{timestamp}")
```

### Option B: Hook into Existing Observation Flow

Since `robot_env.get_observation()` is already called throughout the pipeline, you can:

1. Create a wrapper that logs every observation:
```python
class ObservationLogger:
    def __init__(self, robot_env):
        self._env = robot_env
        self.frames = []

    def get_observation(self):
        obs = self._env.get_observation()
        # Store frames
        self.frames.append({
            'left': self._extract_image(obs, 'left'),
            'wrist': self._extract_image(obs, 'wrist'),
            'timestamp': time.time()
        })
        return obs
```

2. Replace `robot_env` with wrapper in pipeline initialization

### Option C: SVO Recording (Raw ZED Format)

If you need raw data, ZED cameras support SVO recording:
```python
# Start SVO recording before pipeline
cam.enable_recording(sl.RecordingParameters("output.svo"))

# Stop after pipeline
cam.disable_recording()
```

But this requires modifying `droid/camera_utils/camera_readers/zed_camera.py`.

### Recommended Approach

**Use Option A** (separate thread with RobotEnv as source):
- Doesn't require ZED SDK changes
- Records entire pipeline (registration + motion planning + VLA)
- Produces 2 videos: `full_rollout_3rd_person.mp4` and `full_rollout_wrist.mp4`
- Works because `RobotEnv.get_observation()` is thread-safe (just reading camera buffers)

### Files to Create/Modify

1. **Create**: `pipeline/utils/full_video_recorder.py` - Thread-based recorder class
2. **Modify**: `pipeline/main_pipeline.py` - Initialize recorder at task start, stop at end
3. **Modify**: `pipeline/config/real_robot_config.json` - Add video recording settings:
```json
{
  "video_recording": {
    "enabled": true,
    "fps": 15,
    "save_3rd_person": true,
    "save_wrist": true
  }
}
```

---

## Summary

| Question | Key Points |
|----------|------------|
| **1. Pipeline Flow** | Init → Registration (FoundationPose) → Per-skill: Track → Above pose → Motion plan → VLA loop → Lift → Success check |
| **2. New Skills** | Add mesh + ref images + config entries in task_config.json and real_robot_config.json |
| **3. Pi 0.5** | Change to joint_position obs, joint_velocity actions (8D), invert gripper logic |
| **4. Video Recording** | Use separate thread reading from RobotEnv to record full pipeline (both cameras) |
