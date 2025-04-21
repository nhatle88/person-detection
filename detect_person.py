import os
import threading
import torch
import cv2
import numpy as np
import time
import datetime
import subprocess
import math

from ultralytics import YOLO
from camera_list import cameras

# Global dictionary and lock for frames
display_frames = {}
display_lock = threading.Lock()
stop_event = threading.Event()

def get_device(preferred: str = None) -> torch.device:
    """
    Returns the torch device based on a preferred device override,
    auto-detection of CUDA/MPS, or falls back to CPU.
    """
    if preferred is None:
        preferred = os.getenv("TORCH_DEVICE", "auto")
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class Notifier:
    def __init__(self, cooldown: float = 5) -> None:
        self.cooldown = cooldown  # seconds between notifications
        self.last_spoken = 0

    def speak(self, message: str) -> None:
        current_time = time.time()
        if current_time - self.last_spoken >= self.cooldown:
            subprocess.Popen(['say', message])
            self.last_spoken = current_time

class PersonDetection:
    def __init__(self,
                 confidence_threshold: float = 0.4,
                 roi: tuple = None,
                 movement_threshold: float = 5.0,
                 history_size: int = 3) -> None:
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model = YOLO('models/yolov8n.pt')
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold
        self.person_class = 0  # Person class in YOLOv8
        self.notifier = Notifier()
        self.roi = roi  # tuple (x, y, w, h); if None, process full frame
        self.movement_threshold = movement_threshold
        self.previous_frame = None
        self.frame_history = []
        self.history_size = history_size
        self.consecutive_detections = {}

    def _inside_roi(self, box: tuple) -> bool:
        if self.roi is None:
            return True
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
        rx, ry, rw, rh = self.roi
        return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)

    def detect_persons(self, frame: np.ndarray) -> list:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, verbose=False)
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return []

        # Calculate difference between current and previous frame for motion detection
        diff = cv2.absdiff(self.previous_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, thresh_diff = cv2.threshold(blur_diff, 20, 255, cv2.THRESH_BINARY)

        current_time = time.time()
        outdated_keys = [pos for pos, (last_time, _) in self.consecutive_detections.items()
                         if current_time - last_time > 5.0]
        for key in outdated_keys:
            del self.consecutive_detections[key]

        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > self.confidence_threshold and cls == self.person_class:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    w = x2 - x1
                    h = y2 - y1
                    candidate = (x1, y1, w, h)
                    if self._inside_roi(candidate):
                        mask = np.zeros_like(thresh_diff)
                        mask[y1:y2, x1:x2] = 1
                        motion_pixels = cv2.countNonZero(thresh_diff * mask)
                        area = w * h
                        motion_percentage = (motion_pixels / area * 100) if area > 0 else 0
                        grid_x, grid_y = x1 // 50, y1 // 50
                        pos_key = (grid_x, grid_y)
                        if motion_percentage > self.movement_threshold:
                            if pos_key in self.consecutive_detections:
                                _, count = self.consecutive_detections[pos_key]
                                self.consecutive_detections[pos_key] = (current_time, count + 1)
                            else:
                                self.consecutive_detections[pos_key] = (current_time, 1)
                            if self.consecutive_detections[pos_key][1] >= 2 or motion_percentage > self.movement_threshold * 2:
                                persons.append(candidate)
                        else:
                            if pos_key in self.consecutive_detections:
                                _, count = self.consecutive_detections[pos_key]
                                self.consecutive_detections[pos_key] = (current_time, max(0, count - 1))
        self.frame_history.append(frame.copy())
        if len(self.frame_history) > self.history_size:
            self.frame_history.pop(0)
        self.previous_frame = frame.copy()
        return persons

    def draw_persons(self, frame: np.ndarray, persons: list) -> np.ndarray:
        for (x, y, w, h) in persons:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        return frame

    def export_image(self, frame: np.ndarray, camera_name: str) -> None:
        export_dir = "exports"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        timestamp = int(time.time())
        filename = f"{export_dir}/{camera_name}_person_{timestamp}.png"
        if cv2.imwrite(filename, frame):
            print(f"Exported positive result to {filename}")
        else:
            print(f"Failed to export image to {filename}")

def open_stream(rtsp_url: str, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(2)
    return cap

def monitor_camera(camera_url: str, window_name: str, roi: tuple,
                   stop_event: threading.Event,
                   display_frames: dict,
                   display_lock: threading.Lock) -> None:
    detection = PersonDetection(roi=roi)
    if "(HIK)" in window_name:
        print(f"{window_name}: Using HIK Vision capture method")
        cap = open_stream(camera_url)
    else:
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        target_fps = 30
        cap.set(cv2.CAP_PROP_FPS, target_fps)
    target_fps = 30
    frame_time = 1 / target_fps
    frame_count = 0
    start_time = time.time()
    while not stop_event.is_set():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"{window_name}: Unable to capture frame.")
            break
        persons = detection.detect_persons(frame)
        if persons:
            detection.notifier.speak("There is a person")
        processed_frame = detection.draw_persons(frame, persons)
        cv2.putText(processed_frame, window_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        with display_lock:
            display_frames[window_name] = processed_frame
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"{window_name} FPS: {fps:.2f}")
        processing_time = time.time() - loop_start
        delay = max(0, frame_time - processing_time)
        time.sleep(delay)
    cap.release()

def combine_frames(frames: list) -> np.ndarray:
    if not frames:
        return None
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w, channels = frames[0].shape
    resized_frames = [cv2.resize(frame, (w, h)) for frame in frames]
    grid_rows = []
    for i in range(rows):
        row_frames = []
        for j in range(cols):
            idx = i * cols + j
            if idx < len(resized_frames):
                row_frames.append(resized_frames[idx])
            else:
                row_frames.append(np.zeros((h, w, channels), dtype=np.uint8))
        row = cv2.hconcat(row_frames)
        grid_rows.append(row)
    combined_frame = cv2.vconcat(grid_rows)
    return combined_frame

def main() -> None:
    threads = []
    for cam in cameras:
        t = threading.Thread(
            target=monitor_camera,
            args=(cam['url'], cam['name'], cam['roi'],
                  stop_event, display_frames, display_lock)
        )
        t.start()
        threads.append(t)
    while not stop_event.is_set():
        with display_lock:
            frames = list(display_frames.values())
        if frames:
            combined_frame = combine_frames(frames)
            if combined_frame is not None:
                cv2.imshow("Combined Cameras", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    for t in threads:
        t.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()