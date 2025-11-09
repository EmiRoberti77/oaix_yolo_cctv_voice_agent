# OAIX CCTV Agent

A Next.js application that displays MJPEG streams from a FastAPI RTSP server.

## Setup

### 0. Configure Environment Variables

Create a `.env` file in the project root (see `.env.example` for template):

```bash
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY
# - ELEVEN_LABS_API_KEY
# - ELEVEN_LABS_VOICE_ID
```

### 1. Install Python Dependencies

```bash
# Install all dependencies including YOLO
pip install -r requirements.txt

# Or install individually:
pip install fastapi uvicorn opencv-python-headless numpy ultralytics
```

### 2. Install Next.js Dependencies

```bash
npm install
```

### 3. Run the RTSP Server

```bash
cd src/rtsp_server
python app.py
```

The server will run on `http://localhost:8000` by default.

### 4. Run the Next.js Application

In a separate terminal:

```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

## Configuration

### RTSP Server Environment Variables

- `SOURCE_MODE`: `"file"`, `"multifile"`, or `"dir"` (default: `"multifile"`)
- `FILE_PATH`: Path to video file (if `SOURCE_MODE=file`, default: `src/videos/introducer_1.mp4`)
- `VIDEO_DIR`: Path to directory of video files (if `SOURCE_MODE=multifile`, default: `src/videos`)
- `FRAMES_DIR`: Path to directory of images (if `SOURCE_MODE=dir`)
- `LOOP`: `"1"` or `"0"` (default: `"1"`)
- `FOLLOW`: `"1"` or `"0"` (default: `"0"`) - only for dir mode
- `JPEG_QUALITY`: 1-100 (default: `"80"`)
- `TARGET_FPS`: Target frames per second (default: `"20"`)
- `FRAME_WIDTH`: Resize width, 0 = as-is (default: `"0"`)
- `FRAME_HEIGHT`: Resize height, 0 = as-is (default: `"0"`)
- `ENABLE_YOLO`: Enable YOLO object detection for all classes (default: `"1"`)
- `YOLO_MODEL`: YOLO model to use (default: `"yolov8n.pt"` - options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `YOLO_CONFIDENCE`: Detection confidence threshold 0.0-1.0 (default: `"0.25"`)
- `OPENAI_API_KEY`: OpenAI API key for vision analysis (required for alerts)
- `ELEVEN_LABS_API_KEY`: Eleven Labs API key for text-to-speech (required for alerts)
- `ELEVEN_LABS_VOICE_ID`: Eleven Labs voice ID for TTS (required for alerts)
- `ENABLE_ALERTS`: Enable alert system when people detected (default: `"1"`)
- `ALERT_COOLDOWN_SECONDS`: Seconds to wait between alerts (default: `"4.0"`)
- `UVICORN_HOST`: Server host (default: `"0.0.0.0"`)
- `UVICORN_PORT`: Server port (default: `"8000"`)

### Next.js Environment Variables

- `MJPEG_SERVER_URL`: URL of the MJPEG server (default: `http://localhost:8000/mjpeg`)

## Usage

1. Start the RTSP server
2. Start the Next.js dev server
3. Open `http://localhost:3000` in your browser
4. The MJPEG stream will be displayed automatically with YOLO object detection bounding boxes showing all detected classes

## YOLO Detection

The server includes YOLO object detection by default. All detected objects (people, cars, animals, etc.) are shown with colored bounding boxes, class names, and confidence scores. Each class type gets a different color for easy identification.

## Alert System

When a person is detected, the system will:

1. **Send frame to OpenAI Vision API** - Analyzes the scene and generates a warning message describing the intruders and environment
2. **Generate audio warning** - Sends the warning text to Eleven Labs for text-to-speech conversion
3. **Cooldown period** - Only sends alerts once when first detected, then waits 4 seconds (configurable) before sending another if person is still detected

### Alert Configuration

The alert system requires:

- `OPENAI_API_KEY` - For vision analysis
- `ELEVEN_LABS_API_KEY` - For text-to-speech
- `ELEVEN_LABS_VOICE_ID` - Voice ID from Eleven Labs

To disable alerts:

```bash
export ENABLE_ALERTS="0"
```

To change cooldown period:

```bash
export ALERT_COOLDOWN_SECONDS="6.0"  # Wait 6 seconds between alerts
```

### Disable YOLO Detection

```bash
export ENABLE_YOLO="0"
```

### Use Different YOLO Model

```bash
# Use a larger, more accurate model (slower)
export YOLO_MODEL="yolov8m.pt"

# Use a smaller, faster model (less accurate)
export YOLO_MODEL="yolov8n.pt"  # default
```

### Adjust Detection Confidence

```bash
# Higher confidence (fewer false positives, might miss some people)
export YOLO_CONFIDENCE="0.5"

# Lower confidence (more detections, might include false positives)
export YOLO_CONFIDENCE="0.25"  # default
```

## Testing the Server

To verify that `app.py` is working correctly:

### Quick Health Check

```bash
curl http://localhost:8000/health
```

### Test MJPEG Stream

Open in browser: `http://localhost:8000/mjpeg`

### Automated Test

```bash
pip install requests
python test_server.py
```

### View API Documentation

Open in browser: `http://localhost:8000/docs`

For more detailed testing instructions, see [TESTING.md](./TESTING.md).
