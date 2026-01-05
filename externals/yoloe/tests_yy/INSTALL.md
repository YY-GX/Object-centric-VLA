# Installation Instructions

## What You Need to Install

### 1. In YOLOE Conda Environment (Host Machine)

```bash
# Activate yoloe conda environment
conda activate yoloe

# Install ZMQ for communication
pip install pyzmq

# Verify installation
python -c "import zmq; print(f'pyzmq {zmq.pyzmq_version()}')"
python -c "from ultralytics import YOLOE; print('YOLOE OK')"
```

### 2. In FoundationPose Docker Container

```bash
# Enter the docker container
sudo docker exec -it foundationpose bash

# Install ZMQ client
pip install pyzmq

# Verify installation
python -c "import zmq; print(f'pyzmq {zmq.pyzmq_version()}')"
```

## That's It!

No other dependencies needed. The setup keeps YOLOE and FoundationPose completely isolated.

## Quick Test

### Terminal 1: Start YOLOE Server

```bash
cd /home/yygx/PANDA/externals/yoloe/tests_yy/scripts
conda activate yoloe
python yoloe_server.py
```

You should see:
```
Loading YOLOE model: jameslahm/yoloe-v8l-seg
YOLOE server listening on port 5557
Device: cuda:0
Ready to receive requests...
```

### Terminal 2: Test from FoundationPose

```bash
# Inside docker
cd /home/yygx/PANDA/externals/FoundationPose
python tests_yy/scripts/run_registration.py --text_prompt "cup"
```

You should see:
```
Connecting to YOLOE server...
Connected to YOLOE server at tcp://localhost:5557
Running YOLOE segmentation with text='cup'...
✓ YOLOE segmentation complete. Mask shape: (720, 1280), pixels: 12345
```

## Dependencies Summary

| Component | Environment | Dependency | Version |
|-----------|-------------|------------|---------|
| **YOLOE Server** | conda (yoloe) | pyzmq | Latest |
| **YOLOE Server** | conda (yoloe) | ultralytics | (already installed) |
| **FoundationPose Client** | docker (foundationpose) | pyzmq | Latest |

Total additional packages: **1** (pyzmq in each environment)
