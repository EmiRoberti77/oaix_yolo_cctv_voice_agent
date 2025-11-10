from openai import OpenAI
from typing import Optional
import logging
import os
from PIL import Image
from io import BytesIO
import base64
from oaix_prompt import oaix_deterrant_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AIVIsion():
    def __init__(self) -> None:
        self.openai_api_key = OPENAI_API_KEY
        print(f'{self.openai_api_key=}')

         
    def _call_openai_vision(self, frame_base64, ext:str)-> Optional[str]:
            """Call OpenAI Vision API to analyze frame and generate warning message or description."""
            if not self.openai_api_key:
                logger.error("OpenAI API key not set, cannot call API")
                return None
            
            client = OpenAI(api_key=self.openai_api_key)
            # prompt = ("CONTEXT: the image sent to you is from CCTV footage and you are the Security gard - INSTRUCTIN: create a deterrant message that would threaten them to leave with some personal information about the intruders.")
            prompt = ('describe the people in the frame')
        
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

    def resize_and_encode_image(self, path, max_width=800):
        img = Image.open(path)
        if img.width > max_width:
            ratio = max_width / float(img.width)
            height = int((float(img.height) * ratio))
            img = img.resize((max_width, height))

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str


def main(): 
    img = 'image1.png'
    ai = AIVIsion()
    base_64_image = ai.resize_and_encode_image(img)    
    response = ai._call_openai_vision(base_64_image, 'png')
    print(response)

if __name__ == "__main__":
    main()