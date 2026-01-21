"""
Vehicle Detector using YOLOv8
Optimized for AMD GPU via DirectML
"""

from ultralytics import YOLO
import cv2
import numpy as np


class VehicleDetector:
    """Detects and classifies vehicles using YOLOv8"""
    
    # Vehicle classes we care about (COCO dataset class IDs)
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    # Display names with emojis
    VEHICLE_DISPLAY = {
        'car': '🚗 Car',
        'motorcycle': '🏍️ Motorcycle',
        'bus': '🚌 Bus',
        'truck': '🚚 Truck'
    }
    
    # Colors for each vehicle type (BGR format)
    VEHICLE_COLORS = {
        'car': (255, 150, 50),      # Blue
        'motorcycle': (50, 255, 50), # Green
        'bus': (50, 150, 255),       # Orange
        'truck': (255, 50, 150)      # Purple
    }
    
    def __init__(self, model_size='n', confidence=0.5):
        """
        Initialize the detector
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            confidence: Minimum confidence threshold
        """
        self.confidence = confidence
        self.model = None
        self.model_size = model_size
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        print(f"🔄 Loading YOLOv8{self.model_size} model...")
        try:
            self.model = YOLO(f'yolov8{self.model_size}.pt')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect vehicles in a frame
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of detections: [{'bbox': (x1,y1,x2,y2), 'class': str, 'confidence': float, 'center': (cx, cy)}]
        """
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Only process vehicle classes
            if class_id in self.VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                vehicle_class = self.VEHICLE_CLASSES[class_id]
                
                # Calculate center point
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'class': vehicle_class,
                    'confidence': confidence,
                    'center': (cx, cy)
                })
        
        return detections
    
    def draw_detections(self, frame, detections, gate_line_y=None, gate_start=None, gate_end=None):
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: BGR image
            detections: List of detection dictionaries
            gate_line_y: Y position of virtual gate line (for horizontal lines)
            gate_start: (x, y) start point of gate line (for angled lines)
            gate_end: (x, y) end point of gate line (for angled lines)
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw gate line (supports angled lines)
        if gate_start is not None and gate_end is not None:
            cv2.line(annotated, gate_start, gate_end, (0, 100, 255), 3)
            label_pos = (gate_start[0] + 10, gate_start[1] - 10)
            cv2.putText(annotated, "GATE LINE", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        elif gate_line_y is not None:
            cv2.line(annotated, (0, gate_line_y), (frame.shape[1], gate_line_y), 
                     (0, 100, 255), 3)
            cv2.putText(annotated, "GATE LINE", (10, gate_line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            vehicle_class = det['class']
            confidence = det['confidence']
            cx, cy = det['center']
            
            color = self.VEHICLE_COLORS.get(vehicle_class, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            
            # Draw label background
            label = f"{vehicle_class.upper()} {confidence:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated


# Test if run directly
if __name__ == "__main__":
    print("Testing Vehicle Detector...")
    detector = VehicleDetector(model_size='n', confidence=0.4)
    
    # Create a test frame (black image)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_frame)
    print(f"Test complete. Found {len(detections)} vehicles in test frame.")
