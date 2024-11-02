import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import json
from collections import defaultdict

class PersonTracker:
    def __init__(self):
        self.current_sessions = {}  # Track ongoing sessions
        self.person_stats = defaultdict(lambda: {
            'total_time': 0,
            'visits': 0,
            'last_seen': None,
            'session_start': None
        })
        self.log_file = 'aisle_tracking_log.json'
        self.load_existing_stats()

    def load_existing_stats(self):
        """Load existing statistics from log file if it exists"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    loaded_stats = json.load(f)
                    # Convert the loaded stats to our defaultdict structure
                    for person, stats in loaded_stats.items():
                        self.person_stats[person].update(stats)
        except Exception as e:
            print(f"Error loading existing stats: {e}")

    def update_person_tracking(self, name, current_time):
        """Update tracking information for a person"""
        if name == "Desconhecido":
            return

        # If this is a new detection
        if name not in self.current_sessions:
            self.current_sessions[name] = current_time
            self.person_stats[name]['visits'] += 1
            self.person_stats[name]['session_start'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # Update last seen time
        self.person_stats[name]['last_seen'] = current_time.strftime('%Y-%m-%d %H:%M:%S')

    def check_for_departures(self, current_time, timeout_seconds=3):
        """Check for people who have left the frame"""
        for name in list(self.current_sessions.keys()):
            time_difference = (current_time - self.current_sessions[name]).total_seconds()
            
            if time_difference > timeout_seconds:
                # Calculate session duration
                session_duration = time_difference
                self.person_stats[name]['total_time'] += session_duration
                
                # Log the departure
                self.log_departure(name, session_duration)
                
                # Remove from current sessions
                del self.current_sessions[name]

    def log_departure(self, name, session_duration):
        """Log departure information to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.person_stats, f, indent=4)
            print(f"{name} left after {session_duration:.1f} seconds. Total time: {self.person_stats[name]['total_time']:.1f} seconds")
        except Exception as e:
            print(f"Error logging departure: {e}")

def load_known_faces(directory):
    """Load known faces with error handling"""
    known_faces = []
    known_names = []
    print(f"Loading known faces from {directory}")
    
    for filename in os.listdir(directory):
        try:
            image_path = os.path.join(directory, filename)
            print(f"Processing {image_path}")
            image = face_recognition.load_image_file(image_path)
            
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                print(f"No faces detected in {filename}")
                continue
                
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            known_faces.append(face_encoding)
            known_names.append(os.path.splitext(filename)[0])
            print(f"Successfully processed {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
            
    return known_faces, known_names

def process_frame(frame, known_faces, known_names, person_tracker):
    """Process a single frame with error handling"""
    try:
        current_time = datetime.now()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            person_tracker.check_for_departures(current_time)
            return frame
            
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Process each face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Desconhecido"
            
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                
                # Update tracking for this person
                person_tracker.update_person_tracking(name, current_time)
                
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Display name and time spent
            if name != "Desconhecido":
                time_info = f"{name} - Total: {person_tracker.person_stats[name]['total_time']:.1f}s"
                cv2.putText(frame, time_info, (left, top - 10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
        
        # Check for people who have left the frame
        person_tracker.check_for_departures(current_time)
        return frame
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return frame

def main():
    try:
        # Initialize person tracker
        person_tracker = PersonTracker()
        
        # Load known faces
        print("Starting face recognition system...")
        known_faces, known_names = load_known_faces('pics/')
        if not known_faces:
            print("No faces loaded from directory")
            return
            
        # Initialize video capture
        print("Initializing video capture...")
        video_capture = cv2.VideoCapture(2)
        if not video_capture.isOpened():
            print("Error: Could not open video capture")
            return
            
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Process frame
            processed_frame = process_frame(frame, known_faces, known_names, person_tracker)
            
            # Display result
            cv2.imshow('Video', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video_capture.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        
if __name__ == "__main__":
    main()
