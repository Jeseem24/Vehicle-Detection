"""
Vehicle Tracker with Entry/Exit Detection
Uses centroid tracking to follow vehicles across frames
"""

import time
from collections import OrderedDict
import numpy as np


class VehicleTracker:
    """Tracks vehicles and detects gate line crossings"""
    
    def __init__(self, gate_line_y=None, max_disappeared=30, direction_threshold=20, initial_count=0,
                 gate_start=None, gate_end=None, frame_width=1280):
        """
        Initialize tracker
        
        Args:
            gate_line_y: Y position of the virtual gate line (for horizontal lines)
            max_disappeared: Max frames before losing track of object
            direction_threshold: Min pixels moved to determine direction
            initial_count: Initial number of vehicles on campus
            gate_start: (x, y) start point of gate line
            gate_end: (x, y) end point of gate line
            frame_width: Width of video frame (for auto-generating horizontal line)
        """
        self.max_disappeared = max_disappeared
        self.direction_threshold = direction_threshold
        self.initial_count = initial_count
        self.frame_width = frame_width
        
        # Gate line as two points (supports angled lines)
        if gate_start is not None and gate_end is not None:
            self.gate_start = gate_start
            self.gate_end = gate_end
            self.gate_line_y = (gate_start[1] + gate_end[1]) // 2  # Average for backwards compat
        else:
            # Default horizontal line
            self.gate_line_y = gate_line_y if gate_line_y else 300
            self.gate_start = (0, self.gate_line_y)
            self.gate_end = (frame_width, self.gate_line_y)
        
        # Tracking state
        self.next_object_id = 0
        self.objects = OrderedDict()        # ID -> current centroid
        self.previous = OrderedDict()       # ID -> previous centroid
        self.disappeared = OrderedDict()    # ID -> frames since last seen
        self.vehicle_types = OrderedDict()  # ID -> vehicle class
        self.crossed = set()                # IDs that have already crossed the line
        
        # Counts
        self.entry_count = 0
        self.exit_count = 0
        self.counts_by_type = {
            'car': {'in': 0, 'out': 0},
            'motorcycle': {'in': 0, 'out': 0},
            'bus': {'in': 0, 'out': 0},
            'truck': {'in': 0, 'out': 0}
        }
        
        # Recent events for activity log
        self.recent_events = []
    
    def _register(self, centroid, vehicle_type):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.previous[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.vehicle_types[self.next_object_id] = vehicle_type
        self.next_object_id += 1
    
    def _deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.previous[object_id]
        del self.disappeared[object_id]
        del self.vehicle_types[object_id]
    
    def _point_side_of_line(self, px, py):
        """
        Determine which side of the gate line a point is on.
        Returns positive if on one side, negative if on other, 0 if on line.
        Uses cross product of vectors.
        """
        x1, y1 = self.gate_start
        x2, y2 = self.gate_end
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    
    def _check_crossing(self, object_id, prev_pos, curr_pos):
        """Check if object crossed the gate line (works for angled lines)"""
        if object_id in self.crossed:
            return None
        
        vehicle_type = self.vehicle_types.get(object_id, 'car')
        
        # Check if the point crossed from one side of the line to the other
        prev_side = self._point_side_of_line(prev_pos[0], prev_pos[1])
        curr_side = self._point_side_of_line(curr_pos[0], curr_pos[1])
        
        # No crossing if both on same side
        if prev_side * curr_side > 0:
            return None
        
        # Determine direction based on which way the crossing happened
        # Positive to negative = ENTRY (crossing downward/rightward)
        # Negative to positive = EXIT (crossing upward/leftward)
        if prev_side > 0 and curr_side <= 0:
            self.entry_count += 1
            self.counts_by_type[vehicle_type]['in'] += 1
            self.crossed.add(object_id)
            
            event = {
                'time': time.strftime('%I:%M:%S %p'),
                'type': vehicle_type,
                'direction': 'ENTRY',
                'id': object_id
            }
            self.recent_events.insert(0, event)
            self.recent_events = self.recent_events[:50]
            
            return ('entry', vehicle_type)
        
        elif prev_side < 0 and curr_side >= 0:
            self.exit_count += 1
            self.counts_by_type[vehicle_type]['out'] += 1
            self.crossed.add(object_id)
            
            event = {
                'time': time.strftime('%I:%M:%S %p'),
                'type': vehicle_type,
                'direction': 'EXIT',
                'id': object_id
            }
            self.recent_events.insert(0, event)
            self.recent_events = self.recent_events[:50]
            
            return ('exit', vehicle_type)
        
        return None
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with 'center' and 'class'
            
        Returns:
            List of crossing events: [('entry'/'exit', vehicle_type), ...]
        """
        events = []
        
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return events
        
        # Get input centroids and types
        input_centroids = np.array([d['center'] for d in detections])
        input_types = [d['class'] for d in detections]
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self._register(tuple(centroid), input_types[i])
            return events
        
        # Match existing objects to new detections using distance
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        
        # Calculate distances between all pairs
        distances = np.zeros((len(object_centroids), len(input_centroids)))
        for i, obj_cent in enumerate(object_centroids):
            for j, inp_cent in enumerate(input_centroids):
                distances[i, j] = np.sqrt(np.sum((obj_cent - inp_cent) ** 2))
        
        # Greedy matching
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            
            if distances[row, col] > 150:  # Max distance threshold
                continue
            
            object_id = object_ids[row]
            prev_centroid = self.objects[object_id]
            new_centroid = tuple(input_centroids[col])
            
            # Check for gate crossing (pass full centroids for angled line support)
            crossing = self._check_crossing(object_id, prev_centroid, new_centroid)
            if crossing:
                events.append(crossing)
            
            # Update object
            self.previous[object_id] = prev_centroid
            self.objects[object_id] = new_centroid
            self.disappeared[object_id] = 0
            
            used_rows.add(row)
            used_cols.add(col)
        
        # Handle disappeared objects
        for row in range(len(object_centroids)):
            if row not in used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
        
        # Register new objects
        for col in range(len(input_centroids)):
            if col not in used_cols:
                self._register(tuple(input_centroids[col]), input_types[col])
        
        return events
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'total_in': self.entry_count,
            'total_out': self.exit_count,
            'on_campus': max(0, self.initial_count + self.entry_count - self.exit_count),
            'by_type': self.counts_by_type,
            'recent_events': self.recent_events[:10]
        }
    
    def reset(self):
        """Reset all counts"""
        self.entry_count = 0
        self.exit_count = 0
        self.counts_by_type = {
            'car': {'in': 0, 'out': 0},
            'motorcycle': {'in': 0, 'out': 0},
            'bus': {'in': 0, 'out': 0},
            'truck': {'in': 0, 'out': 0}
        }
        self.recent_events = []
        self.crossed.clear()


# Test if run directly
if __name__ == "__main__":
    print("Testing Vehicle Tracker...")
    tracker = VehicleTracker(gate_line_y=300)
    
    # Simulate vehicle moving down (entry)
    for y in range(250, 350, 10):
        detections = [{'center': (320, y), 'class': 'car'}]
        events = tracker.update(detections)
        if events:
            print(f"Event: {events}")
    
    print(f"Stats: {tracker.get_stats()}")
