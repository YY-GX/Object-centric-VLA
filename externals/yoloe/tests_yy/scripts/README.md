# YOLOE ZMQ Server for FoundationPose

ZMQ-based YOLOE server for object segmentation in FoundationPose pipeline.

## Architecture

```
┌─────────────────┐         ZMQ (port 5557)        ┌──────────────────┐
│  FoundationPose │ ◄──────────────────────────────► │  YOLOE Server    │
│  (Docker)       │   Request: RGB + bbox/text      │  (conda env)     │
│                 │   Response: Mask + bbox         │                  │
└─────────────────┘                                 └──────────────────┘
```

## Installation

### 1. Install YOLOE dependencies (in yoloe conda env)

```bash
# Activate yoloe conda environment
conda activate yoloe

# Install ZMQ
pip install pyzmq

# Verify YOLOE is installed
python -c "from ultralytics import YOLOE; print('YOLOE OK')"
```

### 2. Install ZMQ in FoundationPose docker

```bash
# Inside FoundationPose docker container
pip install pyzmq
```

## Usage

### Start YOLOE Server

**Terminal 1** (outside docker, in yoloe conda env):

```bash
cd /home/yygx/PANDA/externals/yoloe/tests_yy/scripts
conda activate yoloe
python yoloe_server.py
```

**Options:**
```bash
# Use different model
python yoloe_server.py --model jameslahm/yoloe-v8s-seg

# Use different port
python yoloe_server.py --port 5558

# Use CPU instead of GPU
python yoloe_server.py --device cpu
```

### Run FoundationPose Registration

**Terminal 2** (inside FoundationPose docker):

```bash
cd /home/yygx/PANDA/externals/FoundationPose
python tests_yy/scripts/run_registration.py --text_prompt "red cup"
```

**Modes:**

1. **Text prompt** (recommended):
```bash
python tests_yy/scripts/run_registration.py --text_prompt "red cup"
```

2. **Bounding box**:
```bash
python tests_yy/scripts/run_registration.py --bbox 789 455 883 573
```

3. **Interactive bbox selection**:
```bash
python tests_yy/scripts/run_registration.py
# Then click two corners in the window
```

**Advanced options:**
```bash
# Use different YOLOE server address
python tests_yy/scripts/run_registration.py --text_prompt "cup" --yoloe_server tcp://192.168.1.100:5557

# Adjust confidence threshold
python tests_yy/scripts/run_registration.py --text_prompt "cup" --conf 0.05
```

## Server API

### Request Format

```python
request = {
    'image': np.ndarray,      # RGB image (H, W, 3) uint8
    'mode': str,              # 'text', 'bbox', or 'prompt_free'
    'text_prompt': str,       # For mode='text'
    'bbox': [x1,y1,x2,y2],   # For mode='bbox'
    'conf': float             # Confidence threshold (default: 0.1)
}
```

### Response Format

```python
response = {
    'masks': List[np.ndarray],      # List of (H, W) uint8 masks
    'bboxes': List[List[float]],    # List of [x1, y1, x2, y2]
    'confidences': List[float],     # Confidence scores
    'classes': List[int]            # Class IDs (for prompt_free)
}
```

## Troubleshooting

### Server not responding

**Check if server is running:**
```bash
# On host machine
netstat -tulpn | grep 5557
```

**Check ZMQ connection:**
```python
import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5557")
print("Connected!")
```

### No objects detected

- **Lower confidence threshold**: `--conf 0.05` or `--conf 0.01`
- **Try different text prompts**: "red cup" vs "cup" vs "mug"
- **Use bbox mode** instead of text mode
- **Check server logs** for error messages

### Docker can't connect to host server

If FoundationPose docker can't reach the host YOLOE server:

```bash
# Use host.docker.internal instead of localhost
python tests_yy/scripts/run_registration.py --text_prompt "cup" \
    --yoloe_server tcp://host.docker.internal:5557
```

Or use host IP:
```bash
# Get host IP
hostname -I

# Use host IP in docker
python tests_yy/scripts/run_registration.py --text_prompt "cup" \
    --yoloe_server tcp://172.16.0.20:5557
```

## Performance

- **Text prompt**: ~100-300ms per image (depends on model size)
- **Visual prompt (bbox)**: ~100-300ms per image
- **Prompt-free**: ~100-300ms per image

Faster models:
- `yoloe-v8s-seg`: Fastest (~300 FPS on T4)
- `yoloe-v8m-seg`: Medium (~150 FPS on T4)
- `yoloe-v8l-seg`: Best quality (~100 FPS on T4)

## Examples

### Example 1: Text prompt for red cup

```bash
# Terminal 1: Start server
python yoloe_server.py

# Terminal 2: Run registration
python tests_yy/scripts/run_registration.py --text_prompt "red cup"
```

### Example 2: Known bounding box

```bash
# Terminal 1: Start server
python yoloe_server.py

# Terminal 2: Run with bbox
python tests_yy/scripts/run_registration.py --bbox 789 455 883 573
```

### Example 3: Interactive selection

```bash
# Terminal 1: Start server
python yoloe_server.py

# Terminal 2: Interactive mode
python tests_yy/scripts/run_registration.py
# Click two corners in the popup window
```
