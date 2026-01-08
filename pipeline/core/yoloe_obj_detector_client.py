"""
YOLOE Object Detector Client

ZMQ client for YOLOE server (text or visual mode).
Used for detecting distractor objects during VLA execution.
"""

import zmq
import pickle
import numpy as np
from typing import List, Dict, Optional


class YOLOEObjectDetectorClient:
    """
    ZMQ client for YOLOE detection server.

    Supports two modes:
    - "text": Connect to yoloe_server.py (port 5557), use text prompts
    - "visual": Connect to yoloe_visual_server.py (port 5559), use visual references
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5559,
        timeout: int = 60000,
        mode: str = "visual"
    ):
        """
        Initialize client.

        Args:
            host: Server host
            port: Server port (5557 for text, 5559 for visual)
            timeout: ZMQ timeout in milliseconds
            mode: "text" or "visual"
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.mode = mode
        self._socket = None
        self._context = None

    def _ensure_connected(self):
        """Lazy connection - connect on first use."""
        if self._socket is None:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self._socket.setsockopt(zmq.SNDTIMEO, self.timeout)
            self._socket.connect(f"tcp://{self.host}:{self.port}")
            print(f"[YOLOEClient] Connected to {self.host}:{self.port} (mode={self.mode})")

    # ==================== Visual Mode APIs ====================

    def register(
        self,
        object_name: str,
        image: np.ndarray,
        bbox: List[int] = None
    ) -> Dict:
        """
        Register reference image for an object (visual mode only).

        Args:
            object_name: Name of object (e.g., 'red_cup_0')
            image: RGB image (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2] or None for full image

        Returns:
            Server response dict
        """
        self._ensure_connected()
        request = {
            'api': 'register',
            'object_name': object_name,
            'image': image,
            'bbox': bbox
        }
        self._socket.send(pickle.dumps(request))
        return pickle.loads(self._socket.recv())

    def detect_visual(
        self,
        object_name: str,
        image: np.ndarray,
        conf: float = 0.1
    ) -> Dict:
        """
        Detect object using visual reference (visual mode).

        Args:
            object_name: Base name of object (e.g., 'red_cup')
            image: RGB image (H, W, 3)
            conf: Confidence threshold

        Returns:
            Dict with 'masks', 'bboxes', 'confidences'
        """
        self._ensure_connected()
        request = {
            'api': 'detect',
            'object_name': object_name,
            'image': image,
            'conf': conf
        }
        self._socket.send(pickle.dumps(request))
        return pickle.loads(self._socket.recv())

    # ==================== Text Mode APIs ====================

    def detect_text(
        self,
        text_prompt: str,
        image: np.ndarray,
        conf: float = 0.1
    ) -> Dict:
        """
        Detect object using text prompt (text mode).

        Args:
            text_prompt: Text description (e.g., 'red plastic cup')
            image: RGB image (H, W, 3)
            conf: Confidence threshold

        Returns:
            Dict with 'masks', 'bboxes', 'confidences'
        """
        self._ensure_connected()
        request = {
            'api': 'segment',
            'image': image,
            'text_prompt': text_prompt,
            'conf': conf
        }
        self._socket.send(pickle.dumps(request))
        return pickle.loads(self._socket.recv())

    # ==================== Unified APIs ====================

    def detect_and_union(
        self,
        image: np.ndarray,
        object_names: List[str],
        text_prompts: Dict[str, str] = None,
        conf: float = 0.1,
        min_conf: float = 0.5
    ) -> Optional[np.ndarray]:
        """
        Detect multiple objects and return union of their masks.

        Args:
            image: RGB image (H, W, 3)
            object_names: List of object names to detect
            text_prompts: Dict mapping object_name -> text_prompt (for text mode)
            conf: Confidence threshold for YOLOE detection
            min_conf: Minimum confidence to include mask in union (default 0.5)

        Returns:
            Union mask (H, W) as uint8, or None if no detections
        """
        if not object_names:
            return None

        h, w = image.shape[:2]
        union_mask = np.zeros((h, w), dtype=np.uint8)
        any_detection = False

        for obj_name in object_names:
            try:
                if self.mode == "visual":
                    response = self.detect_visual(obj_name, image, conf)
                else:
                    # Text mode: need text prompt
                    if text_prompts and obj_name in text_prompts:
                        prompt = text_prompts[obj_name]
                    else:
                        prompt = obj_name  # Fallback to object name
                    response = self.detect_text(prompt, image, conf)

                if 'error' not in response and len(response.get('masks', [])) > 0:
                    # Filter by confidence threshold
                    confidences = response.get('confidences', [1.0] * len(response['masks']))
                    for i, (mask_data, mask_conf) in enumerate(zip(response['masks'], confidences)):
                        if mask_conf >= min_conf:
                            mask = np.array(mask_data, dtype=np.uint8)
                            if mask.shape == (h, w):
                                union_mask = np.maximum(union_mask, mask)
                                any_detection = True
                                print(f"[YOLOEClient] {obj_name} mask {i}: conf={mask_conf:.3f} (included)")
                        else:
                            print(f"[YOLOEClient] {obj_name} mask {i}: conf={mask_conf:.3f} (filtered out)")
            except Exception as e:
                print(f"[YOLOEClient] Detection failed for {obj_name}: {e}")
                continue

        return union_mask if any_detection else None

    def list_objects(self) -> Dict:
        """List all registered objects on server (visual mode only)."""
        self._ensure_connected()
        request = {'api': 'list'}
        self._socket.send(pickle.dumps(request))
        return pickle.loads(self._socket.recv())

    def close(self):
        """Close connection."""
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
