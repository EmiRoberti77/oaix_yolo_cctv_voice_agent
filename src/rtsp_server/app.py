from csv import Error
import os
import cv2
import time
import glob
import threading
import logging
import datetime
import numpy as np
from typing import Optional, Tuple, Generator, List, Dict
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, PlainTextResponse
from starlette.concurrency import iterate_in_threadpool
from dotenv import load_dotenv
import pygame
pygame.mixer.init()

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root (go up from src/rtsp_server to project root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ALERTS = os.path.join(ROOT, 'src', 'rtsp_server', 'alerts')
audio_path = os.path.join(ALERTS,'emi_1.mp3')  # This gives you the actual path string
print(audio_path)
if not os.path.exists(audio_path):
    raise Error('Can not access alerts')
print('AUDIO_FOUND')


# Load environment variables from .env file in project root
env_path = os.path.join(ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"✓ Loaded .env file from: {env_path}")
else:
    logger.warning(f"⚠ .env file not found at {env_path}, trying current directory...")
    load_dotenv()  # Fallback to current directory

# Try to import YOLO, make it optional
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. YOLO detection will be disabled. Install with: pip install ultralytics")


# -----------------------------
# Environment-driven defaults
# -----------------------------
SOURCE_MODE = os.getenv("SOURCE_MODE", "multifile").lower()  # 'file', 'multifile', or 'dir'
# Resolve paths relative to ROOT (project root)
default_file_path = os.path.join(ROOT, 'src', 'videos', 'introducer_1.mp4')
default_video_dir = os.path.join(ROOT, 'src', 'videos')

# Get paths from env, resolve relative paths from ROOT
file_path_env = os.getenv("FILE_PATH", "")
video_dir_env = os.getenv("VIDEO_DIR", "")
frames_dir_env = os.getenv("FRAMES_DIR", "")

# Resolve paths: if relative, make absolute from ROOT; if absolute, use as-is
if file_path_env:
    FILE_PATH = os.path.abspath(os.path.join(ROOT, file_path_env)) if not os.path.isabs(file_path_env) else file_path_env
else:
    FILE_PATH = default_file_path

if video_dir_env:
    VIDEO_DIR = os.path.abspath(os.path.join(ROOT, video_dir_env)) if not os.path.isabs(video_dir_env) else video_dir_env
else:
    VIDEO_DIR = default_video_dir

if frames_dir_env:
    FRAMES_DIR = os.path.abspath(os.path.join(ROOT, frames_dir_env)) if not os.path.isabs(frames_dir_env) else frames_dir_env
else:
    FRAMES_DIR = ""
LOOP = os.getenv("LOOP", "1") not in ("0", "false", "False")
FOLLOW = os.getenv("FOLLOW", "0") not in ("0", "false", "False")
PLAY_AT_REAL_SPEED = os.getenv("PLAY_AT_REAL_SPEED", "1") not in ("0", "false", "False")
ENABLE_YOLO = os.getenv("ENABLE_YOLO", "1") not in ("0", "false", "False") if YOLO_AVAILABLE else False
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")  # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
try:
    YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))  # Detection confidence threshold
except ValueError:
    logger.warning(f"Invalid YOLO_CONFIDENCE value '{os.getenv('YOLO_CONFIDENCE')}', using default 0.25")
    YOLO_CONFIDENCE = 0.25
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))   # 1..100
TARGET_FPS = float(os.getenv("TARGET_FPS", "20"))      # throttle stream output
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "0"))       # 0 = as-is
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "0"))     # 0 = as-is

# OpenAI and Eleven Labs configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY", "")
ELEVEN_LABS_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID", "")  # Voice ID for TTS
ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "1") not in ("0", "false", "False")
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "4.0"))  # Wait 4 seconds between alerts

class YOLODetector:
    """YOLO detector for all object classes."""
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.25):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        self.model = YOLO(model_path)
        self.confidence = confidence
        # Get class names from the model
        self.class_names = self.model.names
        logger.info(f"YOLO model loaded: {model_path} (confidence threshold: {confidence})")
        logger.info(f"Detecting {len(self.class_names)} classes")
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for each class ID."""
        # Generate colors using a hash-like function for consistency
        colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
            (0, 128, 255),    # Light Blue
            (128, 255, 0),    # Lime
        ]
        return colors[class_id % len(colors)]
    
    def detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int, str]]:
        """
        Detect all objects in frame.
        Returns list of (x1, y1, x2, y2, confidence, class_id, class_name) bounding boxes.
        """
        # Remove classes filter to detect all classes
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), class_id, class_name))
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float, int, str]]) -> np.ndarray:
        """Draw bounding boxes and labels on frame with class-specific colors."""
        frame_copy = frame.copy()
        for x1, y1, x2, y2, conf, class_id, class_name in detections:
            # Get color for this class
            color = self._get_color_for_class(class_id)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with class name and confidence
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, label_size[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame_copy, (x1, label_y - label_size[1] - 10), 
                         (x1 + label_size[0], label_y), color, -1)
            
            # Draw label text
            cv2.putText(frame_copy, label, (x1, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame_copy


class BaseSource:
    def __init__(self, size: Tuple[int, int] = (0, 0), yolo_detector: Optional[YOLODetector] = None) -> None:
        self.size = size
        self.yolo_detector = yolo_detector
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._stop_evt = threading.Event()
        self._new_frame_evt = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._reader_thread and self._reader_thread.is_alive():
            return
        self._stop_evt.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="source-reader", daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

    def _reader_loop(self) -> None:
        raise NotImplementedError

    def _publish(self, frame: np.ndarray) -> None:
        w, h = self.size
        if w > 0 and h > 0 and (frame.shape[1] != w or frame.shape[0] != h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        # Apply YOLO detection if enabled
        detections = []
        if self.yolo_detector is not None:
            try:
                detections = self.yolo_detector.detect_objects(frame)
                if detections:
                    people_count = 0
                    for detection in detections:
                        if detection[6] == 'person':
                            people_count += 1
                    
                    frame = self.yolo_detector.draw_detections(frame, detections)
                    if people_count > 1:
                        print(f'{datetime.datetime.now().isoformat()} PLAY_AUDIO FILE >>>>> ')
                        if not pygame.mixer.music.get_busy():  # Only play if nothing is currently playing
                            pygame.mixer.music.load(audio_path)
                            pygame.mixer.music.play()

            except Exception as e:
                logger.error(f"YOLO detection error: {e}")
        
        with self._frame_lock:
            self._latest_frame = frame
            self._new_frame_evt.set()

    def get_latest_jpeg(self, quality: int = 80) -> Optional[bytes]:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        signalled = self._new_frame_evt.wait(timeout)
        if signalled:
            self._new_frame_evt.clear()
        return signalled


class FileVideoSource(BaseSource):
    def __init__(self, path: str, size: Tuple[int, int] = (0, 0), loop: bool = True, play_at_real_speed: bool = True, yolo_detector: Optional[YOLODetector] = None) -> None:
        if not path:
            raise ValueError("FILE_PATH is empty.")
        self.path = path
        self.loop = loop
        self.play_at_real_speed = play_at_real_speed
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_fps: float = 30.0  # Default FPS

    def _open(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        
        # Check if file exists
        if not os.path.exists(self.path):
            logger.error(f"Video file does not exist: {self.path}")
            return False
        
        if not os.path.isfile(self.path):
            logger.error(f"Path is not a file: {self.path}")
            return False
        
        logger.info(f"Attempting to open video file: {self.path}")
        self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        is_opened = self.cap.isOpened()
        
        if is_opened:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = fps if fps > 0 else 30.0  # Store FPS for timing
            logger.info(f"Video opened successfully: {width}x{height} @ {fps}fps, {frame_count} frames")
        else:
            logger.error(f"Failed to open video file: {self.path}")
        
        return is_opened

    def _reader_loop(self) -> None:
        # Open initially
        if not self._open():
            # Give a small grace period and retry a couple times
            logger.warning("Initial open failed, retrying...")
            for i in range(5):
                time.sleep(0.2)
                if self._open():
                    logger.info(f"Successfully opened video on retry {i+1}")
                    break
        if not (self.cap and self.cap.isOpened()):
            logger.error("Could not open video file after all retries. No frames will be available.")
            return

        logger.info("Starting video frame reading loop...")
        
        frame_count = 0
        frame_interval = 1.0 / self.video_fps if self.play_at_real_speed and self.video_fps > 0 else 0.0
        last_frame_time = time.time()
        
        while not self._stop_evt.is_set():
            # If playing at real speed, wait for the next frame time
            if self.play_at_real_speed and frame_interval > 0:
                now = time.time()
                elapsed = now - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()
            
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.loop:
                    # rewind
                    logger.debug("End of video reached, rewinding...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_time = time.time()  # Reset timing on rewind
                    continue
                else:
                    logger.info("End of video reached, stopping...")
                    break
            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug(f"Read {frame_count} frames so far...")
            self._publish(frame)
        
        logger.info(f"Video reading loop ended. Total frames read: {frame_count}")
        # cleanup
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass


class MultiFileVideoSource(BaseSource):
    """Plays multiple video files in sequence, looping through all of them."""
    def __init__(self, video_dir: str, size: Tuple[int, int] = (0, 0), loop: bool = True, play_at_real_speed: bool = True, yolo_detector: Optional[YOLODetector] = None) -> None:
        super().__init__(size, yolo_detector=yolo_detector)
        if not video_dir:
            raise ValueError("VIDEO_DIR is empty.")
        self.video_dir = video_dir
        self.loop = loop
        self.play_at_real_speed = play_at_real_speed
        self.video_files: List[str] = []
        self.current_file_index = 0
        self.current_source: Optional[FileVideoSource] = None
        
    def _scan_videos(self) -> None:
        """Scan directory for video files."""
        patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
        paths = []
        for p in patterns:
            paths.extend(glob.glob(os.path.join(self.video_dir, p)))
        paths.sort()  # Sort alphabetically
        self.video_files = paths
        logger.info(f"Found {len(self.video_files)} video files: {[os.path.basename(f) for f in self.video_files]}")
    
    def _get_current_file(self) -> Optional[str]:
        """Get the current video file path."""
        if not self.video_files:
            return None
        if self.current_file_index >= len(self.video_files):
            if self.loop:
                self.current_file_index = 0
            else:
                return None
        return self.video_files[self.current_file_index]
    
    def _next_file(self) -> Optional[str]:
        """Move to the next video file."""
        if not self.video_files:
            return None
        self.current_file_index += 1
        if self.current_file_index >= len(self.video_files):
            if self.loop:
                self.current_file_index = 0
                logger.info("Completed all videos, looping back to start...")
            else:
                return None
        return self._get_current_file()
    
    def _reader_loop(self) -> None:
        """Read frames from multiple video files in sequence."""
        self._scan_videos()
        
        if not self.video_files:
            logger.error(f"No video files found in {self.video_dir}")
            return
        
        while not self._stop_evt.is_set():
            current_file = self._get_current_file()
            if not current_file:
                logger.info("No more video files to play.")
                break
            
            logger.info(f"Starting video: {os.path.basename(current_file)} ({self.current_file_index + 1}/{len(self.video_files)})")
            
            # Create a FileVideoSource for the current file
            self.current_source = FileVideoSource(
                current_file,
                size=self.size,
                loop=False,  # We handle looping at the multi-file level
                play_at_real_speed=self.play_at_real_speed,
                yolo_detector=self.yolo_detector,
            )
            
            # Open and read the current video
            if not self.current_source._open():
                logger.warning(f"Failed to open {current_file}, skipping...")
                self._next_file()
                continue
            
            # Read frames from current video
            frame_count = 0
            frame_interval = 1.0 / self.current_source.video_fps if self.play_at_real_speed and self.current_source.video_fps > 0 else 0.0
            last_frame_time = time.time()
            
            while not self._stop_evt.is_set():
                # If playing at real speed, wait for the next frame time
                if self.play_at_real_speed and frame_interval > 0:
                    now = time.time()
                    elapsed = now - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()
                
                ok, frame = self.current_source.cap.read()
                if not ok or frame is None:
                    # End of current video, move to next
                    logger.info(f"Finished video: {os.path.basename(current_file)} ({frame_count} frames)")
                    break
                
                frame_count += 1
                self._publish(frame)
            
            # Cleanup current video
            try:
                if self.current_source.cap:
                    self.current_source.cap.release()
            except Exception:
                pass
            
            # Move to next file
            next_file = self._next_file()
            if not next_file:
                logger.info("All videos completed.")
                break
        
        logger.info("Multi-file video reading loop ended.")


class ImageDirSource(BaseSource):
    def __init__(self, dir_path: str, size: Tuple[int, int] = (0, 0), loop: bool = True, follow: bool = False, yolo_detector: Optional[YOLODetector] = None) -> None:
        super().__init__(size, yolo_detector=yolo_detector, )
        if not dir_path:
            raise ValueError("FRAMES_DIR is empty.")
        self.dir_path = dir_path
        self.loop = loop
        self.follow = follow
        self.index = 0
        self.files: List[str] = []

    def _scan(self) -> None:
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        paths = []
        for p in patterns:
            paths.extend(glob.glob(os.path.join(self.dir_path, p)))
        paths.sort()
        self.files = paths

    def _reader_loop(self) -> None:
        self._scan()
        while not self._stop_evt.is_set():
            if self.index >= len(self.files):
                if self.follow:
                    # wait for new files
                    time.sleep(0.1)
                    self._scan()
                    continue
                elif self.loop and len(self.files) > 0:
                    self.index = 0
                else:
                    time.sleep(0.2)
                    continue

            path = self.files[self.index]
            img = cv2.imread(path)
            self.index += 1
            if img is None:
                continue
            self._publish(img)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="File/Dir → MJPEG", version="1.0")

# Choose and start source on startup
source: Optional[BaseSource] = None


@app.on_event("startup")
def _startup():
    global source
    
    # Initialize YOLO detector if enabled
    yolo_detector = None
    if ENABLE_YOLO:
        if not YOLO_AVAILABLE:
            logger.warning("YOLO requested but ultralytics not installed. Install with: pip install ultralytics")
        else:
            try:
                yolo_detector = YOLODetector(model_path=YOLO_MODEL, confidence=YOLO_CONFIDENCE)
                logger.info("YOLO detection enabled")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO detector: {e}")
                yolo_detector = None
    
    logger.info(f"Starting server with SOURCE_MODE={SOURCE_MODE}")
    if SOURCE_MODE == "file":
        logger.info(f"Using video file: {FILE_PATH}")
        if not os.path.exists(FILE_PATH):
            logger.error(f"Video file does not exist: {FILE_PATH}")
            raise FileNotFoundError(f"Video file not found: {FILE_PATH}")
        source = FileVideoSource(
            FILE_PATH, 
            size=(FRAME_WIDTH, FRAME_HEIGHT), 
            loop=LOOP,
            play_at_real_speed=PLAY_AT_REAL_SPEED,
            yolo_detector=yolo_detector,
        )
    elif SOURCE_MODE == "multifile":
        logger.info(f"Using video directory: {VIDEO_DIR}")
        if not os.path.exists(VIDEO_DIR):
            logger.error(f"Video directory does not exist: {VIDEO_DIR}")
            raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")
        source = MultiFileVideoSource(
            VIDEO_DIR,
            size=(FRAME_WIDTH, FRAME_HEIGHT),
            loop=LOOP,
            play_at_real_speed=PLAY_AT_REAL_SPEED,
            yolo_detector=yolo_detector,
        )
    elif SOURCE_MODE == "dir":
        logger.info(f"Using frames directory: {FRAMES_DIR}")
        source = ImageDirSource(FRAMES_DIR, size=(FRAME_WIDTH, FRAME_HEIGHT), loop=LOOP, follow=FOLLOW, yolo_detector=yolo_detector)
    else:
        raise ValueError("Unsupported SOURCE_MODE. Use 'file', 'multifile', or 'dir'.")
    source.start()
    logger.info("Source started successfully")


@app.on_event("shutdown")
def _shutdown():
    if source:
        source.stop()


@app.get("/health", response_class=PlainTextResponse)
def health():
    has_frame = source and source._latest_frame is not None
    source_status = "active" if source else "none"
    file_exists = os.path.exists(FILE_PATH) if FILE_PATH else False
    
    info = [
        f"ok=1",
        f"mode={SOURCE_MODE}",
        f"file={os.path.basename(FILE_PATH) if FILE_PATH else '-'}",
        f"file_path={FILE_PATH}",
        f"file_exists={'yes' if file_exists else 'no'}",
        f"video_dir={VIDEO_DIR}",
        f"dir={FRAMES_DIR if FRAMES_DIR else '-'}",
        f"loop={'1' if LOOP else '0'}",
        f"play_at_real_speed={'1' if PLAY_AT_REAL_SPEED else '0'}",
        f"yolo_enabled={'1' if ENABLE_YOLO and YOLO_AVAILABLE else '0'}",
        f"yolo_model={YOLO_MODEL if ENABLE_YOLO and YOLO_AVAILABLE else '-'}",
        f"yolo_confidence={YOLO_CONFIDENCE if ENABLE_YOLO and YOLO_AVAILABLE else '-'}",
        f"follow={'1' if FOLLOW else '0'}",
        f"jpeg_quality={JPEG_QUALITY}",
        f"target_fps={TARGET_FPS}",
        f"size=({FRAME_WIDTH},{FRAME_HEIGHT})",
        f"source_status={source_status}",
        f"has_frame={'yes' if has_frame else 'no'}",
    ]
    return "\n".join(info)


def mjpeg_generator() -> Generator[bytes, None, None]:
    boundary = b"--frame"
    min_interval = 1.0 / max(TARGET_FPS, 1.0)
    last_sent = 0.0

    while True:
        # Throttle to target FPS
        now = time.time()
        delay = last_sent + min_interval - now
        if delay > 0:
            time.sleep(delay)
        last_sent = time.time()

        # Prefer a fresh frame; if none, use latest
        if source and not source.wait_for_frame(timeout=0.5):
            pass

        jpeg = source.get_latest_jpeg(quality=JPEG_QUALITY) if source else None
        if jpeg is None:
            # tiny keepalive
            blank = np.zeros((2, 2, 3), dtype=np.uint8)
            ok, buf = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if not ok:
                continue
            jpeg = buf.tobytes()

        yield b"".join([
            boundary, b"\n",
            b"Content-Type: image/jpeg\n",
            f"Content-Length: {len(jpeg)}\n\n".encode('ascii'),
            jpeg, b"\n",
        ])


@app.get("/mjpeg")
def mjpeg() -> StreamingResponse:
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return StreamingResponse(
        iterate_in_threadpool(mjpeg_generator()),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
