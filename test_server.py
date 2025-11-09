#!/usr/bin/env python3
"""
Simple test script to verify the RTSP server is working correctly.
"""

import requests
import sys
import time

SERVER_URL = "http://localhost:8000"

def test_health():
    """Test the /health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print("\nServer Status:")
            print(response.text)
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_mjpeg():
    """Test the /mjpeg endpoint"""
    print("\nTesting /mjpeg endpoint...")
    try:
        response = requests.get(
            f"{SERVER_URL}/mjpeg",
            stream=True,
            timeout=10,
            headers={"Accept": "multipart/x-mixed-replace"}
        )
        
        if response.status_code == 200:
            print("✅ MJPEG endpoint is accessible!")
            print(f"   Content-Type: {response.headers.get('Content-Type', 'N/A')}")
            
            # Try to read a few frames
            print("\n   Attempting to read frames...")
            frame_count = 0
            start_time = time.time()
            
            for chunk in response.iter_content(chunk_size=1024):
                if b'--frame' in chunk or b'Content-Type: image/jpeg' in chunk:
                    frame_count += 1
                    if frame_count >= 3:
                        elapsed = time.time() - start_time
                        print(f"   ✅ Successfully received {frame_count} frames in {elapsed:.2f}s")
                        response.close()
                        return True
                if time.time() - start_time > 5:
                    print("   ⚠️  Timeout waiting for frames")
                    response.close()
                    return False
            
            response.close()
            return True
        else:
            print(f"❌ MJPEG endpoint failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 50)
    print("RTSP Server Test")
    print("=" * 50)
    
    health_ok = test_health()
    mjpeg_ok = test_mjpeg()
    
    print("\n" + "=" * 50)
    if health_ok and mjpeg_ok:
        print("✅ All tests passed! Server is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the server logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()

