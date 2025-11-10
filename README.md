# OAIX CCTV Agent

A real-time CCTV monitoring system with YOLO object detection, OpenAI Vision analysis, and Eleven Labs text-to-speech alerts. Streams MJPEG video with bounding boxes for detected objects and provides intelligent alerts when people are detected.

## Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **Node.js 18+** (optional, only if using Next.js frontend)
- **API Keys**:
  - OpenAI API key (for vision analysis)
  - Eleven Labs API key (for text-to-speech)
  - Eleven Labs Voice ID

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd OAIX_cctv_agent
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
# Make sure you're in the project root directory
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

- `fastapi` - Web framework for the RTSP server
- `uvicorn` - ASGI server
- `opencv-python-headless` - Video processing
- `numpy` - Numerical operations
- `ultralytics` - YOLO object detection
- `openai` - OpenAI Vision API
- `python-dotenv` - Environment variable management
- `requests` - HTTP requests
- `Pillow` - Image processing
- `pygame` - Audio playback

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env
```

Add the following to your `.env` file:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
ELEVEN_LABS_API_KEY=your_eleven_labs_api_key_here
ELEVEN_LABS_VOICE_ID=your_eleven_labs_voice_id_here

# Optional Configuration
SOURCE_MODE=multifile
VIDEO_DIR=src/videos
ENABLE_YOLO=1
YOLO_MODEL=yolo11n.pt
YOLO_CONFIDENCE=0.25
ENABLE_ALERTS=1
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000
```

**Note**: The `.env` file should be in the project root directory (same level as `requirements.txt`).

### 5. Prepare Video Files (Optional)

Place your video files in `src/videos/` directory. The default mode (`multifile`) will play all videos in sequence.

### 6. Install Next.js Dependencies (Optional - for frontend)

If you plan to use the Next.js frontend:

```bash
npm install
```

## Running the Application

### Start the RTSP Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# or
# venv\Scripts\activate  # On Windows

# Navigate to the server directory
cd src/rtsp_server

# Run the server
python app.py
```

The server will start on `http://localhost:8000` by default.

**Verify the server is running:**

- Health check: `http://localhost:8000/health`
- MJPEG stream: `http://localhost:8000/oaix_live`
- API docs: `http://localhost:8000/docs`

### Start the Next.js Frontend (Optional)

In a separate terminal:

```bash
# Make sure you're in the project root
npm run dev
```

The frontend will be available at `http://localhost:3000`.

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
- `YOLO_MODEL`: YOLO model to use (default: `"yolo11n.pt"` - options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt, or yolov8 variants)
- `YOLO_CONFIDENCE`: Detection confidence threshold 0.0-1.0 (default: `"0.25"`)
- `OPENAI_API_KEY`: OpenAI API key for vision analysis (required for alerts)
- `ELEVEN_LABS_API_KEY`: Eleven Labs API key for text-to-speech (required for alerts)
- `ELEVEN_LABS_VOICE_ID`: Eleven Labs voice ID for TTS (required for alerts)
- `ENABLE_ALERTS`: Enable alert system when people detected (default: `"1"`)
- `ALERT_COOLDOWN_SECONDS`: Seconds to wait between alerts (default: `"4.0"`)
- `UVICORN_HOST`: Server host (default: `"0.0.0.0"`)
- `UVICORN_PORT`: Server port (default: `"8000"`)

### Next.js Environment Variables

- `MJPEG_SERVER_URL`: URL of the MJPEG server (default: `http://localhost:8000/oaix_live`)

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
export YOLO_MODEL="yolo11m.pt"

# Use a smaller, faster model (less accurate)
export YOLO_MODEL="yolo11n.pt"  # default
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

Open in browser: `http://localhost:8000/oaix_live`

### Automated Test

```bash
pip install requests
python test_server.py
```

### View API Documentation

Open in browser: `http://localhost:8000/docs`

For more detailed testing instructions, see [TESTING.md](./TESTING.md).
