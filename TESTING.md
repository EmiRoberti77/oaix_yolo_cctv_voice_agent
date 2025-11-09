# Testing the RTSP Server

Here are several ways to verify that `app.py` is working correctly:

## 1. Quick Health Check

Test the `/health` endpoint to see server status:

```bash
# Using curl
curl http://localhost:8000/health

# Or open in browser
open http://localhost:8000/health
```

Expected output:

```
ok=1
mode=file
file=introducer_1.mp4
dir=-
loop=1
follow=0
jpeg_quality=80
target_fps=20
size=(0,0)
```

## 2. Test MJPEG Stream in Browser

Open the MJPEG stream directly in your browser:

```bash
# macOS
open http://localhost:8000/mjpeg

# Or manually navigate to:
# http://localhost:8000/mjpeg
```

You should see a live video stream if everything is working.

## 3. Use the Test Script

Run the automated test script:

```bash
# Install requests if needed
pip install requests

# Run the test
python test_server.py
```

This will:

- Check the `/health` endpoint
- Verify the `/mjpeg` endpoint is accessible
- Test that frames are being generated

## 4. Check FastAPI Documentation

FastAPI automatically generates interactive API documentation:

```bash
open http://localhost:8000/docs
```

This shows all available endpoints and lets you test them directly.

## 5. Manual curl Test

Test the MJPEG stream with curl:

```bash
curl -N http://localhost:8000/mjpeg > /tmp/stream_test.mjpeg
```

Let it run for a few seconds, then check if the file has content:

```bash
ls -lh /tmp/stream_test.mjpeg
```

## 6. Check Server Logs

When you start the server, look for:

✅ **Good signs:**

- `INFO: Application startup complete.`
- `INFO: Uvicorn running on http://0.0.0.0:8000`
- Successful `GET /health` and `GET /mjpeg` requests

⚠️ **Warnings (usually OK):**

- `VIDEOIO(FFMPEG): backend is generally available but can't be used to capture by name` - This is normal if the video file path is correct

❌ **Errors to watch for:**

- `FILE_PATH is empty` - Video file not found
- `Unsupported SOURCE_MODE` - Invalid configuration
- Connection refused errors

## 7. Verify Video File Path

Make sure the video file exists:

```bash
# Check if the default video file exists
ls -lh src/videos/introducer_1.mp4

# Or check what FILE_PATH is set to
python -c "import os; print(os.getenv('FILE_PATH', 'src/videos/introducer_1.mp4'))"
```

## Common Issues

### Server won't start

- Check if port 8000 is already in use: `lsof -i :8000`
- Verify Python dependencies are installed: `pip list | grep -E "fastapi|uvicorn|opencv"`

### No frames in stream

- Verify the video file path is correct
- Check that the video file is readable: `file src/videos/introducer_1.mp4`
- Look for errors in server logs about opening the video file

### Connection refused

- Make sure the server is actually running
- Check the host/port configuration
- Verify firewall settings
