# Object-centric-VLA

Object-centric Vision-Language-Action pipeline for real robot manipulation.

## Overview

This repository contains a modular pipeline for long-horizon manipulation tasks on real robots, integrating:
- **FoundationPose** for 6D object pose estimation and tracking
- **YOLOE** for open-vocabulary segmentation
- **VLA** (Vision-Language-Action) model for action generation
- **DROID** robot control interface

## Directory Structure

```
PANDA/
├── externals/
│   ├── FoundationPose/    # 6D pose estimation
│   ├── Any6D/             # 6D pose utilities
│   └── yoloe/             # Open-vocabulary segmentation
└── pipeline/              # Main execution pipeline
    ├── core/              # Core modules (pose, motion, success checking)
    ├── utils/             # Utilities
    └── config/            # Task and robot configurations
```

## Components

- **Object Pose Estimation**: FoundationPose server for registration and tracking
- **Motion Planning**: IK-based and Cartesian Linear Interpolation planners
- **Success Checking**: Heuristic-based verification (object_lifted, object_on_target)
- **VLA Integration**: Language-conditioned action generation

## Setup

See individual component READMEs for setup instructions.
