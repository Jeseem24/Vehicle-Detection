"""
Smart Campus Vehicle Management System
Main Entry Point - With Simulation Mode for Demo
"""

import argparse
import base64
import cv2
import os
import sys
import time
import threading
import random
import webbrowser
from flask import Flask, send_from_directory, jsonify, Response
from flask_cors import CORS

from detector import VehicleDetector
from tracker import VehicleTracker
from logger import VehicleLogger


# Flask app setup
app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Global state
detector = None
tracker = None
logger = None
video_capture = None
processing = False
gate_line_y = 300
current_frame = None
frame_lock = threading.Lock()
simulation_mode = False


def get_demo_video_path():
    """Get path to demo video"""
    demo_dir = os.path.join(os.path.dirname(__file__), '..', 'demo')
    if os.path.exists(demo_dir):
        for f in os.listdir(demo_dir):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return os.path.join(demo_dir, f)
    return None


class SimulatedVehicle:
    """Simulated vehicle for demo mode"""
    TYPES = ['car', 'car', 'car', 'motorcycle', 'motorcycle', 'bus', 'truck']
    COLORS = {
        'car': (255, 150, 50),
        'motorcycle': (50, 255, 50),
        'bus': (50, 150, 255),
        'truck': (255, 50, 150)
    }
    
    def __init__(self, x, y, vehicle_type, direction, width, height):
        self.x = x
        self.y = y
        self.vehicle_type = vehicle_type
        self.direction = direction  # 1 = down, -1 = up
        self.frame_width = width
        self.frame_height = height
        self.speed = random.uniform(4, 8)
        self.box_w = 80 if vehicle_type in ['car', 'truck'] else (120 if vehicle_type == 'bus' else 40)
        self.box_h = 50 if vehicle_type in ['car', 'truck'] else (60 if vehicle_type == 'bus' else 25)
        
    def update(self):
        self.y += self.speed * self.direction
        
    def is_offscreen(self):
        return self.y < -100 or self.y > self.frame_height + 100
    
    def get_detection(self):
        return {
            'bbox': (int(self.x - self.box_w//2), int(self.y - self.box_h//2), 
                     int(self.x + self.box_w//2), int(self.y + self.box_h//2)),
            'class': self.vehicle_type,
            'confidence': 0.85 + random.random() * 0.1,
            'center': (int(self.x), int(self.y))
        }


# Simulation state
simulated_vehicles = []
sim_lock = threading.Lock()


def simulation_thread(width, height, gate_y):
    """Generate simulated vehicle crossings"""
    global simulated_vehicles, processing, tracker, logger
    
    lanes = [width * 0.3, width * 0.42, width * 0.58, width * 0.7]
    
    while processing:
        # Spawn new vehicles randomly
        if random.random() < 0.08:  # ~8% chance per frame
            lane = random.choice(lanes)
            vehicle_type = random.choice(SimulatedVehicle.TYPES)
            
            if lane < width * 0.5:
                direction = 1
                start_y = -50
            else:
                direction = -1
                start_y = height + 50
            
            with sim_lock:
                simulated_vehicles.append(SimulatedVehicle(lane, start_y, vehicle_type, direction, width, height))
        
        # Update and cleanup vehicles
        with sim_lock:
            for v in simulated_vehicles[:]:
                v.update()
                if v.is_offscreen():
                    simulated_vehicles.remove(v)
        
        time.sleep(0.066)  # ~15 FPS


def process_frame_simulation(frame):
    """Process frame with simulated detections"""
    global tracker, logger, gate_line_y, simulated_vehicles, sim_lock
    
    # Get simulated detections
    with sim_lock:
        detections = [v.get_detection() for v in simulated_vehicles]
    
    # Update tracker
    events = tracker.update(detections)
    stats = tracker.get_stats()
    
    # Log events
    for event_type, vehicle_type in events:
        logger.log_event(
            vehicle_type=vehicle_type,
            direction=event_type,
            vehicle_id=tracker.next_object_id - 1,
            total_in=stats['total_in'],
            total_out=stats['total_out']
        )
        print(f"🚗 {vehicle_type.upper()} {event_type.upper()} - Total In: {stats['total_in']}, Out: {stats['total_out']}")
    
    # Draw on frame
    annotated = frame.copy()
    
    # Draw gate line (supports angled lines)
    cv2.line(annotated, tracker.gate_start, tracker.gate_end, (0, 100, 255), 3)
    label_pos = (tracker.gate_start[0] + 10, tracker.gate_start[1] - 10)
    cv2.putText(annotated, "GATE LINE", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        vehicle_class = det['class']
        color = SimulatedVehicle.COLORS.get(vehicle_class, (255, 255, 255))
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, det['center'], 5, color, -1)
        
        label = f"{vehicle_class.upper()} {det['confidence']:.0%}"
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Stats overlay
    cv2.rectangle(annotated, (10, 10), (280, 110), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (280, 110), (255, 150, 50), 2)
    cv2.putText(annotated, f"ENTRY: {stats['total_in']}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    cv2.putText(annotated, f"EXIT: {stats['total_out']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
    cv2.putText(annotated, f"ON CAMPUS: {stats['on_campus']}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Simulation badge
    cv2.rectangle(annotated, (frame.shape[1] - 180, 10), (frame.shape[1] - 10, 45), (0, 100, 255), -1)
    cv2.putText(annotated, "SIMULATION", (frame.shape[1] - 170, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated, stats


def process_frame_detection(frame):
    """Process frame with real YOLO detection"""
    global detector, tracker, logger, gate_line_y
    
    detections = detector.detect(frame)
    events = tracker.update(detections)
    stats = tracker.get_stats()
    
    for event_type, vehicle_type in events:
        logger.log_event(
            vehicle_type=vehicle_type,
            direction=event_type,
            vehicle_id=tracker.next_object_id - 1,
            total_in=stats['total_in'],
            total_out=stats['total_out']
        )
        print(f"🚗 {vehicle_type.upper()} {event_type.upper()} - Total In: {stats['total_in']}, Out: {stats['total_out']}")
    
    annotated = detector.draw_detections(frame, detections, gate_line_y, 
                                          gate_start=tracker.gate_start, 
                                          gate_end=tracker.gate_end)
    
    # Stats overlay
    cv2.rectangle(annotated, (10, 10), (280, 110), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (280, 110), (255, 150, 50), 2)
    cv2.putText(annotated, f"ENTRY: {stats['total_in']}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    cv2.putText(annotated, f"EXIT: {stats['total_out']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
    cv2.putText(annotated, f"ON CAMPUS: {stats['on_campus']}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return annotated, stats


def video_processing_thread(source, demo_mode=False, use_simulation=False, gate_pos=0.75,
                             gate_start=None, gate_end=None):
    """
    Main video processing loop
    Supports:
    - Webcam/File/RTSP
    - Auto-reconnection for streams
    - Demo mode looping
    - Simulation mode
    - Angled gate lines
    """
    global video_capture, processing, detector, tracker, logger, gate_line_y, current_frame, frame_lock
    
    processing = True
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    while processing:
        # Open video source
        print(f"🔄 Connecting to source: {source}...")
        if isinstance(source, int):
            video_capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        elif 'rtsp' in str(source).lower():
            # Use FFMPEG backend with optimized settings for unstable RTSP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "buffer_size;2000000|"  # 2MB buffer
                "max_delay;500000|"     # 500ms max delay
                "stimeout;5000000|"     # 5 second timeout
                "fflags;nobuffer|"      # Reduce buffering
                "flags;low_delay"       # Low latency
            )
            video_capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        else:
            video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            print(f"❌ Failed to open video source: {source}")
            if demo_mode or 'rtsp' in str(source):
                print("⏳ Retrying in 5 seconds...")
                time.sleep(5)
                continue
            else:
                processing = False
                return
        
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine gate line position (angled or horizontal)
        if gate_start is not None and gate_end is not None:
            # Scale gate line coordinates based on video resolution
            # Default coordinates are for 1920x1080, scale proportionally for other resolutions
            base_width, base_height = 1920, 1080
            scale_x = width / base_width
            scale_y = height / base_height
            
            scaled_start = (int(gate_start[0] * scale_x), int(gate_start[1] * scale_y))
            scaled_end = (int(gate_end[0] * scale_x), int(gate_end[1] * scale_y))
            
            tracker.gate_start = scaled_start
            tracker.gate_end = scaled_end
            gate_line_y = (scaled_start[1] + scaled_end[1]) // 2
            tracker.gate_line_y = gate_line_y
            print(f"📹 Video: {width}x{height}, Gate Line: {scaled_start} -> {scaled_end} (ANGLED)")
        else:
            gate_line_y = int(height * gate_pos)
            tracker.gate_line_y = gate_line_y
            tracker.gate_start = (0, gate_line_y)
            tracker.gate_end = (width, gate_line_y)
            print(f"📹 Video: {width}x{height}, Gate Line: y={gate_line_y} ({gate_pos:.0%})")
        
        print(f"🚀 Starting {'SIMULATION' if use_simulation else 'DETECTION'} mode...")

        # Start simulation thread if needed (only once)
        if use_simulation and not any(t.name == 'SimThread' for t in threading.enumerate()):
            sim_thread = threading.Thread(target=simulation_thread, args=(width, height, gate_line_y), daemon=True, name='SimThread')
            sim_thread.start()
        
        frame_time = 1.0 / 60  # Target 60 FPS for smooth playback
        
        while processing and video_capture.isOpened():
            start_time = time.time()
            
            ret, frame = video_capture.read()
            
            if not ret:
                if demo_mode:
                    print("🔄 Loop video...")
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("⚠️ Video stream ended or signal lost.")
                    break
            
            # Check for corrupted/gray frames (low variance = likely corrupted)
            gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = gray_check.var()
            if variance < 100:  # Very low variance = mostly gray/corrupted
                # Skip this frame, use last good frame if available
                if current_frame is not None:
                    continue
            
            # Process frame (with frame skipping for RTSP to reduce CPU load)
            try:
                if use_simulation:
                    annotated, stats = process_frame_simulation(frame)
                else:
                    # Frame skipping: run YOLO every 3rd frame for RTSP streams
                    is_rtsp = 'rtsp' in str(source).lower()
                    skip_interval = 3 if is_rtsp else 1
                    
                    if frame_count % skip_interval == 0:
                        # Full detection on this frame
                        annotated, stats = process_frame_detection(frame)
                        last_annotated = annotated  # Save for skipped frames
                    else:
                        # Use last detection result, just copy current frame with overlay
                        if 'last_annotated' in dir() and last_annotated is not None:
                            annotated = last_annotated
                            stats = tracker.get_stats()
                        else:
                            annotated, stats = process_frame_detection(frame)
                            last_annotated = annotated
                    
                cv2.putText(annotated, f"FPS: {fps}", (width - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                continue
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                print(f"⚡ Processing: {fps} FPS | Total In: {tracker.entry_count} | Out: {tracker.exit_count}   ", end='\r')
                frame_count = 0
                fps_time = time.time()
            
            with frame_lock:
                current_frame = annotated.copy()
            
            # Frame rate control
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        video_capture.release()
        
        # logic for retry
        if processing and ('rtsp' in str(source) or 'http' in str(source)):
             print("\n⚠️ Connection lost. Reconnecting in 2s...")
             time.sleep(2)
        elif processing and demo_mode:
             pass # Loop handle inside
        else:
             print("\n👋 Source ended.")
             break
    
    print("📹 Video processing stopped")


def generate_mjpeg():
    """Generate MJPEG stream"""
    global current_frame, frame_lock
    
    while True:
        with frame_lock:
            if current_frame is not None:
                _, jpeg = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.066)


# Flask Routes
@app.route('/')
def serve_dashboard():
    return send_from_directory('../frontend', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    if tracker:
        return jsonify(tracker.get_stats())
    return jsonify({'total_in': 0, 'total_out': 0, 'on_campus': 0, 'by_type': {}, 'recent_events': []})


@app.route('/api/reset', methods=['POST'])
def reset_stats():
    if tracker:
        tracker.reset()
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'Tracker not initialized'})


def main():
    global detector, tracker, logger, processing, simulation_mode
    
    parser = argparse.ArgumentParser(description='Smart Campus Vehicle Management System')
    parser.add_argument('--source', type=str, default='demo',
                        help='Video source: "demo", "0" for webcam, or path to video file')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port (default: 5000)')
    parser.add_argument('--confidence', type=float, default=0.4,
                        help='Detection confidence threshold (default: 0.4)')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l'],
                        help='YOLOv8 model size (default: n for nano)')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (loops video)')
    parser.add_argument('--simulate', action='store_true',
                        help='Use simulated vehicle detection (for expo demo)')
    parser.add_argument('--gate-line', type=float, default=0.75,
                        help='Gate line position (0.0 - 1.0, default: 0.75). Ignored if --gate-start/end used.')
    parser.add_argument('--gate-start', type=str, default=None,
                        help='Gate line start point as "x,y" (e.g., "100,400")')
    parser.add_argument('--gate-end', type=str, default=None,
                        help='Gate line end point as "x,y" (e.g., "1180,500")')
    parser.add_argument('--initial-count', type=int, default=0,
                        help='Initial vehicles on campus (default: 0)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("🏫 Smart Campus Vehicle Management System")
    print("=" * 50)
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    use_simulation = args.simulate
    
    if not use_simulation:
        detector = VehicleDetector(model_size=args.model, confidence=args.confidence)
    else:
        print("🎮 Simulation mode enabled - no YOLO detection")
    
    # Initial dummy init (will be updated in thread)
    tracker = VehicleTracker(gate_line_y=300, initial_count=args.initial_count)
    logger = VehicleLogger(log_dir='../logs')
    
    # Determine video source
    source = args.source
    demo_mode = args.demo
    use_simulation = args.simulate

    # Handle "demo" keyword or auto-demo mode
    if source == 'demo':
        demo_video = get_demo_video_path()
        if demo_video:
            source = demo_video
            demo_mode = True
            print(f"🎬 Using demo video: {demo_video}")
            # Only auto-enable simulation for the synthetic video
            if 'demo_traffic.mp4' in demo_video and not args.simulate:
                use_simulation = True
                print("⚡ Auto-enabling simulation for synthetic demo video")
        else:
            print("⚠️ No demo video found.")
            source = 0
    # Handle numeric source (webcam)
    elif source.isdigit():
        source = int(source)
        print(f"📹 Using webcam: {source}")
    # Handle file path or URL
    else:
        print(f"🎥 Using video source: {source}")
        if not os.path.exists(source) and 'rtsp' not in source and 'http' not in source:
             print(f"⚠️ Warning: File not found: {source}")

    # Parse gate line points (if provided)
    # Default: optimized for college gate camera (1920x1080) - scales automatically for other resolutions
    # These create a diagonal line across the center-right area of the frame
    gate_start = (800, 1080)    # Top-left of line (for 1920x1080)
    gate_end = (1460, 0)     # Bottom-right of line (for 1920x1080)
    if args.gate_start and args.gate_end:
        try:
            gate_start = tuple(map(int, args.gate_start.split(',')))
            gate_end = tuple(map(int, args.gate_end.split(',')))
            print(f"📐 Custom angled gate line: {gate_start} -> {gate_end}")
        except ValueError:
            print("⚠️ Warning: Invalid gate-start/gate-end format. Using default horizontal line.")
            gate_start = None
            gate_end = None

    # Start video processing
    video_thread = threading.Thread(
        target=video_processing_thread,
        args=(source, demo_mode, use_simulation, args.gate_line, gate_start, gate_end),
        daemon=True
    )
    video_thread.start()
    
    # Start Flask server
    print(f"\n🌐 Dashboard: http://localhost:{args.port}")
    print(f"📺 Video Feed: http://localhost:{args.port}/video_feed")
    print("   Press Ctrl+C to stop\n")
    
    # Auto-open browser with delay to ensure server is ready
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open(f'http://localhost:{args.port}')
    
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        processing = False


if __name__ == '__main__':
    main()
