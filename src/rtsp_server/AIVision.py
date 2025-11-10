from openai import OpenAI
from typing import Optional, Union
import logging
import os
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2
import requests
import time
import pygame

from dotenv import load_dotenv

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root (go up from src/rtsp_server to project root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables from .env file in project root
env_path = os.path.join(ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"✓ Loaded .env file from: {env_path}")
else:
    logger.warning(f"⚠ .env file not found at {env_path}, trying current directory...")
    load_dotenv()  # Fallback to current directory

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY", "")
ELEVEN_LABS_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID", "")

class AIVIsion():
    def __init__(self) -> None:
        self.openai_api_key = OPENAI_API_KEY
        self.eleven_labs_api_key = ELEVEN_LABS_API_KEY
        self.eleven_labs_voice_id = ELEVEN_LABS_VOICE_ID
        print(f'{self.openai_api_key=}')

         
    def call_openai_vision(self, frame_base64, ext:str)-> Optional[str]:
            """Call OpenAI Vision API to analyze frame and generate warning message or description."""
            if not self.openai_api_key:
                logger.error("OpenAI API key not set, cannot call API")
                return None
            
            client = OpenAI(api_key=self.openai_api_key)
            # prompt = ("CONTEXT: the image sent to you is from CCTV footage and you are the Security gard - INSTRUCTIN: create a deterrant message that would threaten them to leave with some personal information about the intruders.")
            ai_prompt = """
                    SYSTEM: You are a professional security assistant. Produce only neutral, factual, non-identifying descriptions of people in an image. Do NOT name, identify, or invent personal data. Avoid speculating about identity, age, race, health, or intent.

                    USER: 
                        Context: I will provide a single CCTV image. Write a 2–3 sentence, colloquial description of the people visible in the frame. Include number of people, clothing and colours, what they appear to be doing, where they are in the frame, and note if faces are covered. Use cautious language (e.g. "appears to be", "possibly"), keep it concise, and do not include any identifying details.
                    
                    OUTPUT: Plain text only — exactly 2–3 short sentences.
            """
            prompt = (ai_prompt)
        
            logger.info(f"Sending request to OpenAI with frame (size: {len(frame_base64)} chars base64)")
            response = client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4-vision-preview" for older models
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{ext};base64,{frame_base64}"
                                }
                            }
                        ]
                    }
                ]
                # max_tokens=600
            )
            message = response.choices[0].message.content.strip()
            return message

    def resize_and_encode_image(self, frame: Union[str, np.ndarray], max_width=800) -> str:
        """Resize and encode image from file path or numpy array."""
        if isinstance(frame, str):
            # File path provided
            img = Image.open(frame)
        elif isinstance(frame, np.ndarray):
            # Numpy array (cv2 frame) provided - convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")
        
        # Resize if needed
        if img.width > max_width:
            ratio = max_width / float(img.width)
            height = int((float(img.height) * ratio))
            img = img.resize((max_width, height))

        # Encode to base64
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str
    
    def text_to_speech_eleven_labs(self, text: str, output_dir: str = None) -> Optional[str]:
        """Convert text to speech using Eleven Labs API and return audio file path."""
        if not self.eleven_labs_api_key or not self.eleven_labs_voice_id:
            logger.error("Eleven Labs API key or Voice ID not set, cannot generate audio")
            return None
        
        try:
            logger.info("Calling Eleven Labs API for TTS...")
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.eleven_labs_voice_id}/stream"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.eleven_labs_api_key
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            logger.info(f"Sending request to Eleven Labs with text: {text[:50]}...")
            response = requests.post(url, json=data, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Stream the audio data
            audio_data = b""
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_data += chunk
            
            # Save audio to file
            if output_dir is None:
                # Default to alerts directory
                output_dir = os.path.join(ROOT, 'src', 'rtsp_server', 'alerts')
            
            os.makedirs(output_dir, exist_ok=True)
            timestamp = int(time.time())
            audio_filename = os.path.join(output_dir, f"tts_{timestamp}.mp3")
            
            with open(audio_filename, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"✓ Eleven Labs audio saved to: {audio_filename} ({len(audio_data)} bytes)")
            return audio_filename
            
        except Exception as e:
            logger.error(f"✗ Eleven Labs API error: {e}", exc_info=True)
            return None
    
    def play_audio_file(self, audio_path: str) -> bool:
        """Play audio file automatically using pygame."""
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Initialize pygame mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Load and play audio (non-blocking)
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            logger.info(f"✓ Playing audio: {os.path.basename(audio_path)}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error playing audio: {e}", exc_info=True)
            return False
    
    def text_to_speech_and_play(self, text: str, output_dir: str = None) -> bool:
        """Convert text to speech and play it automatically. Returns True if successful."""
        audio_file = self.text_to_speech_eleven_labs(text, output_dir)
        if audio_file:
            return self.play_audio_file(audio_file)
        return False
