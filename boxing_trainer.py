import cv2
import time
import random
import numpy as np
import sys
import statistics

def check_camera():
    num_cameras = 0
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available")
            num_cameras += 1
            cap.release()
    return num_cameras

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Error loading face cascade classifier")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

class CameraCalibrator:
    def __init__(self):
        self.calibration_samples = []
        self.is_calibrating = False
        self.flash_start_time = None
        self.calibrated_delay = 0
        self.required_samples = 3  # Number of samples to collect
        self.brightness_threshold = 50
        self.current_state = 'waiting'  # 'waiting', 'white', 'black'
        self.state_change_time = None
        self.current_sample = 0

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_samples = []
        self.flash_start_time = None
        self.current_state = 'waiting'
        self.current_sample = 0
        print("Starting camera calibration. Please ensure you're in a dark room.")
        cv2.namedWindow('Boxing Response Trainer', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Boxing Response Trainer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def process_frame(self, frame):
        if not self.is_calibrating:
            return False

        current_time = time.time()
        avg_brightness = np.mean(frame)

        if self.current_state == 'waiting':
            self.current_state = 'black'
            self.state_change_time = current_time
            return False
            
        elif self.current_state == 'black':
            if current_time - self.state_change_time > 1.0:  # Wait 1 second in black
                self.current_state = 'white'
                self.flash_start_time = current_time
                self.state_change_time = current_time
            return False
            
        elif self.current_state == 'white':
            if avg_brightness > self.brightness_threshold:
                delay = (current_time - self.flash_start_time) * 1000
                self.calibration_samples.append(delay)
                self.current_sample += 1
                print(f"Calibration sample {self.current_sample}: {delay:.1f}ms")
                
                if self.current_sample >= self.required_samples:
                    return True
                    
                self.current_state = 'black'
                self.state_change_time = current_time
            return False

        return False

    def get_calibration_frame(self):
        if self.current_state == 'white':
            frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 255  # White frame
        else:
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Black frame
            
        # Add instructions
        if self.current_state == 'waiting':
            text = "Calibration starting... Please wait"
        else:
            text = f"Calibrating... Sample {self.current_sample + 1}/{self.required_samples}"
            
        cv2.putText(frame, text, (frame.shape[1]//2 - 300, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0) if self.current_state == 'black' else (0, 0, 0), 2)
        
        return frame

    def is_complete(self):
        return len(self.calibration_samples) >= self.required_samples

    def calculate_delay(self):
        if not self.calibration_samples:
            return None
        self.calibrated_delay = statistics.mean(self.calibration_samples)
        self.is_calibrating = False
        return self.calibrated_delay

class BoxingTracker:
    def __init__(self):
        self.stimulus_active = False
        self.stimulus_start_time = 0
        self.response_times = []
        self.last_stimulus_time = time.time()
        self.original_face_pos = None
        self.ready_for_next = True
        self.calibration_frames = []
        self.avg_frame_time = 0.033
        self.countdown_start = None
        self.movement_threshold = 0.5
        self.response_recorded = False
        self.face_width = None
        self.face_height = None
        self.current_response_time = None
        self.camera_delay = 0
    
    def set_camera_delay(self, delay):
        self.camera_delay = delay
        print(f"Camera delay set to: {delay:.1f}ms")
    
    def calibrate_camera_lag(self, frame_time):
        self.calibration_frames.append(frame_time)
        if len(self.calibration_frames) > 30:
            self.calibration_frames.pop(0)
        self.avg_frame_time = sum(self.calibration_frames) / len(self.calibration_frames)
        
    def start_countdown(self):
        self.countdown_start = time.time()
        self.original_face_pos = None
        self.stimulus_active = False
        self.response_recorded = False
        self.current_response_time = None
        
    def get_countdown_state(self):
        if self.countdown_start is None:
            return None
        elapsed = time.time() - self.countdown_start
        if elapsed < 3:
            return 3 - int(elapsed)
        elif elapsed < 3 + random.uniform(1.5, 7):
            return 0
        else:
            self.stimulus_active = True
            self.stimulus_start_time = time.time()
            self.countdown_start = None
            return None

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def main():
    num_cameras = check_camera()
    if num_cameras == 0:
        print("No cameras found!")
        sys.exit(1)
    
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Camera initialized successfully!")
    print("Press 'c' to calibrate camera lag")
    print("Press 'q' to quit the program")

    tracker = BoxingTracker()
    calibrator = CameraCalibrator()
    last_frame_time = time.time()
    
    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame_time = time.time()
        frame_time = current_frame_time - last_frame_time
        tracker.calibrate_camera_lag(frame_time)
        last_frame_time = current_frame_time

        # Calibration mode
        if calibrator.is_calibrating:
            calibration_frame = calibrator.get_calibration_frame()
            cv2.imshow('Boxing Response Trainer', calibration_frame)
            
            if calibrator.process_frame(frame):
                delay = calibrator.calculate_delay()
                if delay:
                    tracker.set_camera_delay(delay)
                    print(f"Calibration complete. Camera delay: {delay:.1f}ms")
                    print("Press SPACE to start testing your reaction time!")
                cv2.waitKey(1000)  # Wait 1 second before returning to normal mode
            
            cv2.waitKey(1)
            continue

        # Normal operation mode
        faces = detect_faces(frame)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            current_face_pos = (x + w//2, y + h//2)
            
            if tracker.original_face_pos is None and tracker.countdown_start is None:
                tracker.original_face_pos = current_face_pos
                tracker.face_width = w
                tracker.face_height = h
            
            if tracker.stimulus_active and not tracker.response_recorded:
                if tracker.original_face_pos is not None:
                    distance = np.sqrt((current_face_pos[0] - tracker.original_face_pos[0])**2 + 
                                     (current_face_pos[1] - tracker.original_face_pos[1])**2)
                    
                    if distance > tracker.face_width * tracker.movement_threshold:
                        response_time = (time.time() - tracker.stimulus_start_time - tracker.camera_delay/1000) * 1000
                        tracker.response_times.append(response_time)
                        tracker.stimulus_active = False
                        tracker.ready_for_next = True
                        tracker.response_recorded = True
                        tracker.current_response_time = response_time
                        print(f"Response time: {response_time:.1f} ms")
        
        clean_frame = frame.copy()
        
        countdown_value = tracker.get_countdown_state()
        if countdown_value is not None:
            if countdown_value > 0:
                cv2.putText(clean_frame, str(countdown_value), 
                           (frame.shape[1]//2-50, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
        
        if tracker.stimulus_active:
            cv2.rectangle(clean_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
        
        if tracker.response_recorded and tracker.original_face_pos is not None:
            orig_x = int(tracker.original_face_pos[0] - tracker.face_width//2)
            orig_y = int(tracker.original_face_pos[1] - tracker.face_height//2)
            cv2.rectangle(clean_frame, 
                         (orig_x, orig_y), 
                         (orig_x + tracker.face_width, orig_y + tracker.face_height), 
                         (255, 0, 0), 2)
        
        if tracker.ready_for_next:
            if tracker.current_response_time is not None:
                cv2.putText(clean_frame, f"Response time: {tracker.current_response_time:.1f} ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(clean_frame, "Press SPACE to try again", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if tracker.response_times:
            avg_time = sum(tracker.response_times) / len(tracker.response_times)
            cv2.putText(clean_frame, f"Avg Response: {avg_time:.1f} ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Boxing Response Trainer', clean_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibrator.start_calibration()
        elif key == ord(' ') and tracker.ready_for_next:
            tracker.ready_for_next = False
            tracker.start_countdown()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
