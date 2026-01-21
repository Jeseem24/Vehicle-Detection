"""
Demo Video Generator
Creates a synthetic video with animated vehicles for testing
Run this once to generate a demo video
"""

import cv2
import numpy as np
import random
import os


class Vehicle:
    """Represents an animated vehicle in the demo"""
    
    COLORS = {
        'car': (255, 150, 50),
        'motorcycle': (50, 255, 50),
        'bus': (50, 150, 255),
        'truck': (255, 50, 150)
    }
    
    SIZES = {
        'car': (80, 50),
        'motorcycle': (40, 25),
        'bus': (120, 60),
        'truck': (100, 55)
    }
    
    def __init__(self, x, y, vehicle_type, direction):
        self.x = x
        self.y = y
        self.vehicle_type = vehicle_type
        self.direction = direction  # 1 = down (entry), -1 = up (exit)
        self.width, self.height = self.SIZES[vehicle_type]
        self.color = self.COLORS[vehicle_type]
        self.speed = random.uniform(3, 7)
    
    def update(self):
        self.y += self.speed * self.direction
    
    def draw(self, frame):
        x1 = int(self.x - self.width // 2)
        y1 = int(self.y - self.height // 2)
        x2 = int(self.x + self.width // 2)
        y2 = int(self.y + self.height // 2)
        
        # Draw vehicle body
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Draw wheels
        wheel_radius = 8 if self.vehicle_type != 'motorcycle' else 5
        cv2.circle(frame, (x1 + 15, y2), wheel_radius, (40, 40, 40), -1)
        cv2.circle(frame, (x2 - 15, y2), wheel_radius, (40, 40, 40), -1)
        
        # Label
        label = self.vehicle_type.upper()[0]
        cv2.putText(frame, label, (int(self.x) - 8, int(self.y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def is_offscreen(self, height):
        return self.y < -50 or self.y > height + 50


def generate_demo_video(output_path, duration=60, fps=30, width=1280, height=720):
    """
    Generate a demo video with animated vehicles
    
    Args:
        output_path: Where to save the video
        duration: Video duration in seconds
        fps: Frames per second
        width, height: Video dimensions
    """
    print(f"🎬 Generating demo video: {output_path}")
    print(f"   Duration: {duration}s, Size: {width}x{height}, FPS: {fps}")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Gate line position
    gate_y = height // 2
    
    # Lanes
    lanes = [width * 0.3, width * 0.45, width * 0.55, width * 0.7]
    
    # Vehicles
    vehicles = []
    
    # Vehicle spawn settings
    vehicle_types = ['car', 'car', 'car', 'motorcycle', 'motorcycle', 'bus', 'truck']
    spawn_interval = fps // 2  # Spawn every 0.5 seconds on average
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create background (road)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw road
        cv2.rectangle(frame, (int(width * 0.2), 0), (int(width * 0.8), height), (60, 60, 60), -1)
        
        # Draw lane markings
        for lane_x in [width * 0.35, width * 0.5, width * 0.65]:
            for y in range(0, height, 40):
                cv2.rectangle(frame, (int(lane_x) - 3, y), (int(lane_x) + 3, y + 20), (200, 200, 200), -1)
        
        # Draw gate line
        cv2.line(frame, (int(width * 0.2), gate_y), (int(width * 0.8), gate_y), (0, 100, 255), 4)
        cv2.putText(frame, "GATE LINE", (int(width * 0.2) + 10, gate_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        
        # Spawn new vehicles
        if random.random() < (1.0 / spawn_interval):
            lane = random.choice(lanes)
            vehicle_type = random.choice(vehicle_types)
            
            # Decide direction based on lane
            if lane < width * 0.5:
                # Left lanes go down (entry)
                direction = 1
                start_y = -50
            else:
                # Right lanes go up (exit)
                direction = -1
                start_y = height + 50
            
            vehicles.append(Vehicle(lane, start_y, vehicle_type, direction))
        
        # Update and draw vehicles
        for vehicle in vehicles[:]:
            vehicle.update()
            vehicle.draw(frame)
            
            if vehicle.is_offscreen(height):
                vehicles.remove(vehicle)
        
        # Add title overlay
        cv2.rectangle(frame, (10, 10), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, "DEMO VIDEO", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        cv2.putText(frame, "Smart Campus Vehicle Management", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Write frame
        out.write(frame)
        
        # Progress
        if frame_num % (fps * 5) == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   Progress: {progress:.0f}%")
    
    out.release()
    print(f"✅ Demo video saved: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'demo')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'demo_traffic.mp4')
    generate_demo_video(output_path, duration=60, fps=30)
