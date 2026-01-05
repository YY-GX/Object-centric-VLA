#!/usr/bin/env python3
"""
YOLOE ZMQ Server - Provides object detection and segmentation via ZMQ.

This server runs YOLOE model and responds to segmentation requests via ZMQ REP socket.
It supports both text prompts and visual prompts (bounding boxes).

Usage:
    python yoloe_server.py --model yoloe-v8l-seg --port 5557
"""

import argparse
import zmq
import numpy as np
import cv2
import pickle
from pathlib import Path
import sys
from PIL import Image

# Add YOLOE to path
YOLOE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(YOLOE_ROOT))

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor


class YOLOEServer:
    def __init__(self, model_name="jameslahm/yoloe-v8l-seg", port=5557, device="cuda:0"):
        """
        Initialize YOLOE server.

        Args:
            model_name: Model to load (e.g., "jameslahm/yoloe-v8l-seg")
            port: ZMQ port to bind to
            device: Device to run model on
        """
        print(f"Loading YOLOE model: {model_name}")
        if "/" in model_name:
            self.model = YOLOE.from_pretrained(model_name)
        else:
            self.model = YOLOE(model_name)

        self.model.to(device)
        self.model.eval()
        self.device = device

        # Setup ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        print(f"YOLOE server listening on port {port}")
        print(f"Device: {device}")
        print("Ready to receive requests...")

    def process_request(self, request):
        """
        Process a segmentation request.

        Request format:
        {
            'image': np.ndarray (H, W, 3) RGB uint8,
            'mode': 'text' or 'bbox' or 'prompt_free',
            'text_prompt': str (if mode='text'),
            'bbox': [x1, y1, x2, y2] (if mode='bbox'),
            'conf': float (confidence threshold, default 0.1)
        }

        Response format:
        {
            'masks': List[np.ndarray] (N, H, W) bool,
            'bboxes': List[List[float]] (N, 4) [x1, y1, x2, y2],
            'confidences': List[float] (N,),
            'classes': List[int] (N,)  # For prompt-free mode
        }
        """
        try:
            image = request['image']
            orig_h, orig_w = image.shape[:2]  # Store original dimensions
            mode = request.get('mode', 'text')
            conf = request.get('conf', 0.1)

            print(f"[DEBUG] Received image shape: {image.shape}, dtype: {image.dtype}")
            print(f"[DEBUG] Original dimensions: {orig_w}x{orig_h}")

            # Convert numpy array to PIL Image (YOLOE expects PIL Image for correct preprocessing)
            # image is RGB numpy array (H, W, 3) uint8
            pil_image = Image.fromarray(image)
            print(f"[DEBUG] PIL image size: {pil_image.size}")  # (width, height)

            # Run YOLOE
            if mode == 'text':
                text_prompt = request.get('text_prompt', 'object')
                # Set text classes
                text_pe = self.model.get_text_pe([text_prompt])
                self.model.set_classes([text_prompt], text_pe)

                # Run prediction
                results = self.model.predict(pil_image, conf=conf, verbose=False)

            elif mode == 'bbox':
                bbox = request.get('bbox')
                # Visual prompt format
                prompts = {
                    'bboxes': [np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])],
                    'cls': [np.array([0])]
                }

                # Reset predictor to use visual prompt predictor
                self.model.predictor = None
                results = self.model.predict(pil_image, prompts=prompts,
                                            predictor=YOLOEVPSegPredictor,
                                            conf=conf, verbose=False)

            elif mode == 'prompt_free':
                results = self.model.predict(pil_image, conf=conf, verbose=False)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Extract results
            result = results[0]

            response = {
                'masks': [],
                'bboxes': [],
                'confidences': [],
                'classes': []
            }

            if result.masks is not None and len(result.masks) > 0:
                # Convert masks to numpy arrays, then to lists (avoid numpy pickle issues)
                masks = result.masks.data.cpu().numpy()  # (N, H_model, W_model) - model inference size
                inference_shape = tuple(masks.shape[1:])  # (H_inference, W_inference)
                orig_shape = (orig_h, orig_w)

                print(f"[DEBUG] Raw mask shape from model: {inference_shape}")
                print(f"[DEBUG] Original image shape: {orig_shape}")

                # Calculate padding (YOLOE adds letterboxing during inference)
                pad = (0, 0)
                if inference_shape != orig_shape:
                    gain = min(
                        inference_shape[0] / orig_shape[0],
                        inference_shape[1] / orig_shape[1],
                    )
                    pad = (
                        (inference_shape[1] - orig_shape[1] * gain) / 2,
                        (inference_shape[0] - orig_shape[0] * gain) / 2,
                    )

                top, left = int(pad[1]), int(pad[0])
                bottom, right = int(inference_shape[0] - pad[1]), int(inference_shape[1] - pad[0])
                print(f"[DEBUG] Padding: top={top}, left={left}, bottom={bottom}, right={right}")

                # Resize masks back to original image dimensions
                resized_masks = []
                for mask in masks:
                    # Crop to remove padding
                    mask_cropped = mask[top:bottom, left:right]

                    # Resize to original image size
                    if mask_cropped.shape != orig_shape:
                        resized_mask = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        resized_mask = mask_cropped

                    # Threshold to get binary mask (since resize may create interpolated values)
                    resized_mask = (resized_mask > 0.5).astype(np.uint8)
                    resized_masks.append(resized_mask)
                    print(f"[DEBUG] Resized mask shape: {resized_mask.shape}, non-zero pixels: {resized_mask.sum()}")

                # Convert each mask to list to avoid numpy version issues with pickle
                response['masks'] = [mask.tolist() for mask in resized_masks]

                # Extract bounding boxes
                bboxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                response['bboxes'] = [bbox.tolist() for bbox in bboxes]
                print(f"[DEBUG] Bounding boxes from model: {bboxes}")

                # Extract confidences
                confidences = result.boxes.conf.cpu().numpy()  # (N,)
                response['confidences'] = confidences.tolist()
                print(f"[DEBUG] Confidences: {confidences}")

                # Extract class IDs (for prompt-free mode)
                if result.boxes.cls is not None:
                    classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
                    response['classes'] = classes.tolist()

            return response

        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def run(self):
        """Main server loop."""
        try:
            while True:
                # Wait for request
                message = self.socket.recv()
                request = pickle.loads(message)

                print(f"Received request: mode={request.get('mode', 'text')}", end='')
                if 'text_prompt' in request:
                    print(f", prompt='{request['text_prompt']}'", end='')
                print()

                # Process request
                response = self.process_request(request)

                # Send response
                self.socket.send(pickle.dumps(response))

                num_detections = len(response.get('masks', []))
                print(f"Sent response: {num_detections} detections")

        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.socket.close()
            self.context.term()


def main():
    parser = argparse.ArgumentParser(description="YOLOE ZMQ Server")
    parser.add_argument(
        '--model',
        type=str,
        default='jameslahm/yoloe-v8l-seg',
        help='YOLOE model name (default: jameslahm/yoloe-v8l-seg)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5557,
        help='ZMQ port (default: 5557)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run model (default: cuda:0)'
    )

    args = parser.parse_args()

    server = YOLOEServer(
        model_name=args.model,
        port=args.port,
        device=args.device
    )

    server.run()


if __name__ == '__main__':
    main()
