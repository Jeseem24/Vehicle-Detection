"""
Smart Campus Vehicle Management System
Main Entry Point - With RTSP Fix for Grey Screen Issue
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
from queue import Queue, Empty
from flask import Flask, send_from_directory, jsonify, Response
from flask_cors import CORS

from detector import VehicleDetector
from tracker import VehicleTracker
from logger import VehicleLogger

# ============== NEW: RTSP STREAM HANDLER ==============
class RTSPStream:
    """
    Threaded RTSP stream handler to prevent grey screens and lag.
    Reads frames in background thread, always provides latest frame.
    """
    
    def __init__(self, rtsp_url, reconnect_delay=2):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.frame_queue = Queue(maxsize=2)
        self.stopped = False
        self.cap = None
        self.last_valid_frame = None
        self.connected = False
        self.lock = threading.Lock()
        self.width = 0
        self.height = 0
        self.consecutive_failures = 0
        self.max_failures = 50
        
    def start(self):
        """Start the background frame reading thread"""
        self._connect()
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        # Wait a bit for first frame
        time.sleep(1)
        return self
    
    def _connect(self):
        """Connect to RTSP stream with optimized settings"""
        print(f"🔄 Connecting to RTSP: {self.rtsp_url[:50]}...")
        
        if self.cap is not None:
            self.cap.release()
        
        # Method 1: TCP transport via URL parameter
        url_with_tcp = self.rtsp_url
        if '?' not in self.rtsp_url:
            url_with_tcp = self.rtsp_url + "?rtsp_transport=tcp"
        elif 'rtsp_transport' not in self.rtsp_url:
            url_with_tcp = self.rtsp_url + "&rtsp_transport=tcp"
        
        # Set environment variables for FFMPEG
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "buffer_size;1024000|"
            "max_delay;500000|"
            "stimeout;5000000|"
            "reorder_queue_size;0|"
            "fflags;nobuffer+discardcorrupt|"
            "flags;low_delay"
        )
        
        self.cap = cv2.VideoCapture(url_with_tcp, cv2.CAP_FFMPEG)
        
        if self.cap.isOpened():
            # Minimal buffer to get latest frames
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.connected = True
            self.consecutive_failures = 0
            print(f"✅ Connected! Resolution: {self.width}x{self.height}")
            return True
        else:
            print(f"❌ Failed to connect")
            self.connected = False
            return False
    
    def _is_valid_frame(self, frame):
        """Check if frame is valid (not grey/corrupted)"""
        if frame is None:
            return False
        if frame.size == 0:
            return False
        
        # Check for grey/blank frame using multiple methods
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Standard deviation (grey frames have very low std)
        std_dev = gray.std()
        if std_dev < 8:  # Lowered threshold
            return False
        
        # Method 2: Check if frame is mostly one color
        mean_val = gray.mean()
        if mean_val < 5 or mean_val > 250:  # Too dark or too bright
            # Additional check - could be valid night footage
            if std_dev < 15:
                return False
        
        return True
    
    def _update(self):
        """Background thread: continuously read frames"""
        while not self.stopped:
            if not self.connected or self.cap is None or not self.cap.isOpened():
                print("⚠️ Stream disconnected, reconnecting...")
                time.sleep(self.reconnect_delay)
                self._connect()
                continue
            
            try:
                # Grab frame (non-blocking)
                grabbed = self.cap.grab()
                
                if not grabbed:
                    self.consecutive_failures += 1
                    if self.consecutive_failures > self.max_failures:
                        print("⚠️ Too many grab failures, reconnecting...")
                        self.connected = False
                    continue
                
                # Retrieve frame
                ret, frame = self.cap.retrieve()
                
                if not ret or frame is None:
                    self.consecutive_failures += 1
                    continue
                
                # Validate frame
                if not self._is_valid_frame(frame):
                    self.consecutive_failures += 1
                    # Use last valid frame if available
                    if self.consecutive_failures > 10 and self.last_valid_frame is not None:
                        frame = self.last_valid_frame.copy()
                    else:
                        continue
                else:
                    self.consecutive_failures = 0
                    self.last_valid_frame = frame.copy()
                
                # Update queue (drop old frame if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                print(f"❌ Frame read error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.1)
    
    def read(self):
        """Get the latest frame"""
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except Empty:
            # Return last valid frame if available
            if self.last_valid_frame is not None:
                return True, self.last_valid_frame.copy()
            return False, None
    
    def get_dimensions(self):
        """Get video dimensions"""
        return self.width, self.height
    
    def isOpened(self):
        """Check if stream is open"""
        return self.connected
    
    def release(self):
        """Release the stream"""
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
    
    def stop(self):
        """Stop the stream"""
        self.release()


# ============== Flask app setup ==============
app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Global state
detector = None
tracker = None
logger = None
video_capture = None
rtsp_stream = None  # NEW: For RTSP streams
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
        self.direction = direction
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
        if random.random() < 0.08:
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
        
        with sim_lock:
            for v in simulated_vehicles[:]:
                v.update()
                if v.is_offscreen():
                    simulated_vehicles.remove(v)
        
        time.sleep(0.066)


def process_frame_simulation(frame):
    """Process frame with simulated detections"""
    global tracker, logger, gate_line_y, simulated_vehicles, sim_lock
    
    with sim_lock:
        detections = [v.get_detection() for v in simulated_vehicles]
    
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
    
    annotated = frame.copy()
    
    cv2.line(annotated, tracker.gate_start, tracker.gate_end, (0, 100, 255), 3)
    label_pos = (tracker.gate_start[0] + 10, tracker.gate_start[1] - 10)
    cv2.putText(annotated, "GATE LINE", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        vehicle_class = det['class']
        color = SimulatedVehicle.COLORS.get(vehicle_class, (255, 255, 255))
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, det['center'], 5, color, -1)
        
        label = f"{vehicle_class.upper()} {det['confidence']:.0%}"
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.rectangle(annotated, (10, 10), (280, 110), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (280, 110), (255, 150, 50), 2)
    cv2.putText(annotated, f"ENTRY: {stats['total_in']}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    cv2.putText(annotated, f"EXIT: {stats['total_out']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
    cv2.putText(annotated, f"ON CAMPUS: {stats['on_campus']}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
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
    
    cv2.rectangle(annotated, (10, 10), (280, 110), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (280, 110), (255, 150, 50), 2)
    cv2.putText(annotated, f"ENTRY: {stats['total_in']}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
    cv2.putText(annotated, f"EXIT: {stats['total_out']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
    cv2.putText(annotated, f"ON CAMPUS: {stats['on_campus']}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return annotated, stats


# ============== FIXED VIDEO PROCESSING THREAD ==============
def video_processing_thread(source, demo_mode=False, use_simulation=False, gate_pos=0.75,
                           gate_start=None, gate_end=None):
    """
    Main video processing loop - FIXED for RTSP streams
    """
    global video_capture, rtsp_stream, processing, detector, tracker, logger
    global gate_line_y, current_frame, frame_lock
    
    processing = True
    frame_count = 0
    fps_time = time.time()
    fps = 0
    last_annotated = None
    is_rtsp = 'rtsp' in str(source).lower() or 'http' in str(source).lower()
    
    while processing:
        print(f"🔄 Opening source: {source}...")
        
        # ===== USE THREADED RTSP READER FOR RTSP STREAMS =====
        if is_rtsp:
            rtsp_stream = RTSPStream(source).start()
            
            if not rtsp_stream.isOpened():
                print("❌ Failed to open RTSP stream, retrying...")
                time.sleep(5)
                continue
            
            width, height = rtsp_stream.get_dimensions()
            
            # Use rtsp_stream.read() instead of video_capture.read()
            video_source = rtsp_stream
        else:
            # Regular video file or webcam
            if isinstance(source, int):
                video_capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                video_capture = cv2.VideoCapture(source)
            
            if not video_capture.isOpened():
                print(f"❌ Failed to open source: {source}")
                if demo_mode:
                    time.sleep(5)
                    continue
                else:
                    processing = False
                    return
            
            width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_source = video_capture
        
        # Configure gate line
        if gate_start is not None and gate_end is not None:
            base_width, base_height = 1920, 1080
            scale_x = width / base_width
            scale_y = height / base_height
            
            scaled_start = (int(gate_start[0] * scale_x), int(gate_start[1] * scale_y))
            scaled_end = (int(gate_end[0] * scale_x), int(gate_end[1] * scale_y))
            
            tracker.gate_start = scaled_start
            tracker.gate_end = scaled_end
            gate_line_y = (scaled_start[1] + scaled_end[1]) // 2
            tracker.gate_line_y = gate_line_y
            print(f"📹 Video: {width}x{height}, Gate Line: {scaled_start} -> {scaled_end}")
        else:
            gate_line_y = int(height * gate_pos)
            tracker.gate_line_y = gate_line_y
            tracker.gate_start = (0, gate_line_y)
            tracker.gate_end = (width, gate_line_y)
            print(f"📹 Video: {width}x{height}, Gate Line: y={gate_line_y}")
        
        print(f"🚀 Starting {'SIMULATION' if use_simulation else 'DETECTION'} mode...")
        
        # Start simulation thread if needed
        if use_simulation and not any(t.name == 'SimThread' for t in threading.enumerate()):
            sim_thread = threading.Thread(
                target=simulation_thread, 
                args=(width, height, gate_line_y), 
                daemon=True, 
                name='SimThread'
            )
            sim_thread.start()
        
        frame_time = 1.0 / 30  # Target 30 FPS
        
        # ===== MAIN FRAME PROCESSING LOOP =====
        while processing:
            start_time = time.time()
            
            # Read frame (works for both RTSP and regular sources)
            if is_rtsp:
                ret, frame = rtsp_stream.read()
            else:
                ret, frame = video_capture.read()
            
            if not ret or frame is None:
                if demo_mode and not is_rtsp:
                    print("🔄 Looping video...")
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                elif is_rtsp:
                    # RTSP stream handles reconnection internally
                    time.sleep(0.1)
                    continue
                else:
                    print("⚠️ Video ended")
                    break
            
            # Process frame
            try:
                if use_simulation:
                    annotated, stats = process_frame_simulation(frame)
                else:
                    # Frame skipping for RTSP (process every 2nd frame)
                    skip_interval = 2 if is_rtsp else 1
                    
                    if frame_count % skip_interval == 0:
                        annotated, stats = process_frame_detection(frame)
                        last_annotated = annotated
                    else:
                        if last_annotated is not None:
                            annotated = last_annotated
                            stats = tracker.get_stats()
                        else:
                            annotated, stats = process_frame_detection(frame)
                            last_annotated = annotated
                
                # Add FPS counter
                cv2.putText(annotated, f"FPS: {fps}", (width - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add LIVE indicator for RTSP
                if is_rtsp:
                    cv2.circle(annotated, (width - 150, 25), 8, (0, 0, 255), -1)
                    cv2.putText(annotated, "LIVE", (width - 135, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                continue
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                print(f"⚡ FPS: {fps} | In: {tracker.entry_count} | Out: {tracker.exit_count}   ", end='\r')
                frame_count = 0
                fps_time = time.time()
            
            # Update current frame for web streaming
            with frame_lock:
                current_frame = annotated.copy()
            
            # Frame rate control
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        # Cleanup
        if is_rtsp and rtsp_stream:
            rtsp_stream.stop()
        elif video_capture:
            video_capture.release()
        
        # Reconnect logic
        if processing and is_rtsp:
            print("\n⚠️ Stream ended, reconnecting in 3s...")
            time.sleep(3)
        elif processing and demo_mode:
            pass
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
        time.sleep(0.033)  # ~30 FPS


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
                        help='Video source: "demo", "0" for webcam, RTSP URL, or path to video')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port (default: 5000)')
    parser.add_argument('--confidence', type=float, default=0.4,
                        help='Detection confidence threshold (default: 0.4)')
    parser.add_argument('--model', type=str, default='n',
                        choices=['n', 's', 'm', 'l'],
                        help='YOLOv8 model size (default: n)')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (loops video)')
    parser.add_argument('--simulate', action='store_true',
                        help='Use simulated detection')
    parser.add_argument('--gate-line', type=float, default=0.75,
                        help='Gate line position (0.0-1.0)')
    parser.add_argument('--gate-start', type=str, default=None,
                        help='Gate line start "x,y"')
    parser.add_argument('--gate-end', type=str, default=None,
                        help='Gate line end "x,y"')
    parser.add_argument('--initial-count', type=int, default=0,
                        help='Initial vehicles on campus')
    args = parser.parse_args()
    
    print("=" * 50)
    print("🏫 Smart Campus Vehicle Management System")
    print("   RTSP Grey Screen Fix Applied ✅")
    print("=" * 50)
    
    # Initialize components
    print("\n📦 Initializing components...")
    
    use_simulation = args.simulate
    
    if not use_simulation:
        detector = VehicleDetector(model_size=args.model, confidence=args.confidence)
    else:
        print("🎮 Simulation mode - no YOLO detection")
    
    tracker = VehicleTracker(gate_line_y=300, initial_count=args.initial_count)
    logger = VehicleLogger(log_dir='../logs')
    
    # Determine video source
    source = args.source
    demo_mode = args.demo
    
    if source == 'demo':
        demo_video = get_demo_video_path()
        if demo_video:
            source = demo_video
            demo_mode = True
            print(f"🎬 Using demo video: {demo_video}")
            if 'demo_traffic.mp4' in demo_video and not args.simulate:
                use_simulation = True
                print("⚡ Auto-enabling simulation")
        else:
            print("⚠️ No demo video found, using webcam")
            source = 0
    elif source.isdigit():
        source = int(source)
        print(f"📹 Using webcam: {source}")
    else:
        print(f"🎥 Using source: {source}")
        if 'rtsp' in source.lower():
            print("📡 RTSP stream detected - using threaded reader")
    
    # Parse gate line
    gate_start = (800, 1080)
    gate_end = (1460, 0)
    if args.gate_start and args.gate_end:
        try:
            gate_start = tuple(map(int, args.gate_start.split(',')))
            gate_end = tuple(map(int, args.gate_end.split(',')))
            print(f"📐 Custom gate line: {gate_start} -> {gate_end}")
        except ValueError:
            print("⚠️ Invalid gate format, using defaults")
            gate_start = None
            gate_end = None
    
    # Start video processing
    video_thread = threading.Thread(
        target=video_processing_thread,
        args=(source, demo_mode, use_simulation, args.gate_line, gate_start, gate_end),
        daemon=True
    )
    video_thread.start()
    
    # Start Flask
    print(f"\n🌐 Dashboard: http://localhost:{args.port}")
    print(f"📺 Video Feed: http://localhost:{args.port}/video_feed")
    print("   Press Ctrl+C to stop\n")
    
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f'http://localhost:{args.port}')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        processing = False


if __name__ == '__main__':
    main()