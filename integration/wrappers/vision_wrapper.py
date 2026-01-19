"""
Vision Module Wrapper

Provides a standardized interface to the YOLOv8-based vehicle and pedestrian
detection module (module1_vision).
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys

# Add module path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "module1_vision"))


@dataclass
class DetectionResult:
    """Result from vehicle/pedestrian detection."""
    image_name: str
    vehicle_count: int
    pedestrian_count: int
    congestion: str
    raw_detections: Optional[List[dict]] = None
    
    def to_dict(self) -> dict:
        return {
            'image_name': self.image_name,
            'vehicle_count': self.vehicle_count,
            'pedestrian_count': self.pedestrian_count,
            'congestion': self.congestion
        }


class VisionWrapper:
    """
    Wrapper for YOLOv8-based vehicle/pedestrian detection.
    
    This wrapper provides a clean interface to the detection functionality
    in module1_vision, handling model loading and inference.
    
    Example:
        wrapper = VisionWrapper()
        result = wrapper.detect(image)
        print(f"Vehicles: {result.vehicle_count}")
    """
    
    # COCO class IDs
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PEDESTRIAN_CLASS = 0
    
    def __init__(self, model_path: str = None):
        """
        Initialize the vision wrapper.
        
        Args:
            model_path: Path to YOLOv8 model weights. Uses default if not provided.
        """
        self._model = None
        self._model_path = model_path
        
    def _load_model(self):
        """Lazy load the YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                
                if self._model_path:
                    self._model = YOLO(self._model_path)
                else:
                    # Default path relative to module
                    default_path = Path(__file__).parent.parent.parent / "module1_vision" / "yolov8n.pt"
                    if default_path.exists():
                        self._model = YOLO(str(default_path))
                    else:
                        # Fall back to downloading
                        self._model = YOLO("yolov8n.pt")
            except ImportError:
                raise ImportError("ultralytics package required. Install with: pip install ultralytics")
    
    def detect(self, image: np.ndarray, image_name: str = "unknown.jpg") -> DetectionResult:
        """
        Detect vehicles and pedestrians in an image.
        
        Args:
            image: Input image as numpy array (BGR format).
            image_name: Name/identifier for the image.
            
        Returns:
            DetectionResult with counts and congestion level.
        """
        self._load_model()
        
        results = self._model(image, verbose=False)
        
        vehicle_count = 0
        pedestrian_count = 0
        raw_detections = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detection = {
                    'class_id': cls_id,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].tolist() if box.xyxy is not None else None
                }
                raw_detections.append(detection)
                
                if cls_id in self.VEHICLE_CLASSES:
                    vehicle_count += 1
                elif cls_id == self.PEDESTRIAN_CLASS:
                    pedestrian_count += 1
        
        congestion = self._estimate_congestion(vehicle_count)
        
        return DetectionResult(
            image_name=image_name,
            vehicle_count=vehicle_count,
            pedestrian_count=pedestrian_count,
            congestion=congestion,
            raw_detections=raw_detections
        )
    
    def process_batch(self, images: List[Tuple[np.ndarray, str]]) -> List[DetectionResult]:
        """
        Process a batch of images.
        
        Args:
            images: List of (image, image_name) tuples.
            
        Returns:
            List of DetectionResults.
        """
        return [self.detect(img, name) for img, name in images]
    
    def detect_from_file(self, image_path: str) -> DetectionResult:
        """
        Detect from an image file.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            DetectionResult with counts and congestion level.
        """
        import cv2
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        return self.detect(image, image_path.name)
    
    @staticmethod
    def _estimate_congestion(vehicle_count: int) -> str:
        """Estimate congestion level from vehicle count."""
        if vehicle_count < 10:
            return "LOW"
        elif vehicle_count < 25:
            return "Medium"
        else:
            return "High"
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
