"""
CSV Logger for Vehicle Events
Creates timestamped logs of all vehicle entries and exits
"""

import csv
import os
from datetime import datetime


class VehicleLogger:
    """Logs vehicle events to CSV files"""
    
    def __init__(self, log_dir='../logs'):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        self._ensure_dir()
        self.current_file = None
        self.current_date = None
        self._open_log_file()
    
    def _ensure_dir(self):
        """Ensure log directory exists"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"📁 Created log directory: {self.log_dir}")
    
    def _open_log_file(self):
        """Open or create today's log file"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today != self.current_date:
            self.current_date = today
            filename = f"vehicle_log_{today}.csv"
            filepath = os.path.join(self.log_dir, filename)
            
            file_exists = os.path.exists(filepath)
            self.current_file = filepath
            
            # Write header if new file
            if not file_exists:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp',
                        'date',
                        'time',
                        'vehicle_type',
                        'direction',
                        'vehicle_id',
                        'total_in',
                        'total_out'
                    ])
                print(f"📝 Created new log file: {filename}")
    
    def log_event(self, vehicle_type, direction, vehicle_id, total_in, total_out):
        """
        Log a vehicle event
        
        Args:
            vehicle_type: Type of vehicle (car, motorcycle, bus, truck)
            direction: 'entry' or 'exit'
            vehicle_id: Unique tracking ID
            total_in: Current total entries
            total_out: Current total exits
        """
        self._open_log_file()  # Check if we need a new file
        
        now = datetime.now()
        
        with open(self.current_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.isoformat(),
                now.strftime('%Y-%m-%d'),
                now.strftime('%H:%M:%S'),
                vehicle_type,
                direction.upper(),
                vehicle_id,
                total_in,
                total_out
            ])
    
    def get_today_summary(self):
        """Get summary of today's events"""
        self._open_log_file()
        
        if not os.path.exists(self.current_file):
            return {'total': 0, 'entries': 0, 'exits': 0, 'by_type': {}}
        
        summary = {
            'total': 0,
            'entries': 0,
            'exits': 0,
            'by_type': {
                'car': {'in': 0, 'out': 0},
                'motorcycle': {'in': 0, 'out': 0},
                'bus': {'in': 0, 'out': 0},
                'truck': {'in': 0, 'out': 0}
            }
        }
        
        with open(self.current_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary['total'] += 1
                vehicle_type = row['vehicle_type']
                direction = row['direction'].lower()
                
                if direction == 'entry':
                    summary['entries'] += 1
                    if vehicle_type in summary['by_type']:
                        summary['by_type'][vehicle_type]['in'] += 1
                else:
                    summary['exits'] += 1
                    if vehicle_type in summary['by_type']:
                        summary['by_type'][vehicle_type]['out'] += 1
        
        return summary
    
    def get_log_files(self):
        """Get list of all log files"""
        files = []
        for f in os.listdir(self.log_dir):
            if f.startswith('vehicle_log_') and f.endswith('.csv'):
                files.append(f)
        return sorted(files, reverse=True)


# Test if run directly
if __name__ == "__main__":
    print("Testing Vehicle Logger...")
    logger = VehicleLogger(log_dir='../logs')
    
    # Log some test events
    logger.log_event('car', 'entry', 1, 1, 0)
    logger.log_event('motorcycle', 'entry', 2, 2, 0)
    logger.log_event('car', 'exit', 1, 2, 1)
    
    print(f"Today's summary: {logger.get_today_summary()}")
    print(f"Log files: {logger.get_log_files()}")
