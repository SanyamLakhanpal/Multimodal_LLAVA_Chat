from typing import Tuple, Union
from PIL import Image
from gtts import gTTS
import re
import datetime
import whisper
import torch
from transformers import BitsAndBytesConfig, pipeline
import yaml
from fastapi import FastAPI

from exceptions import WhisperProcessingError
from lru_cache import cached

app = FastAPI()

with open('constants.yaml') as f:
    constants = yaml.safe_load(f)

quantization_config = BitsAndBytesConfig(**constants['quantization_config'])

model_id = constants['model_id']

pipe = pipeline("image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config})

@cached()
async def img2txt(input_text: Union[str, Tuple], input_image: str) -> str:
    """Generate image description using GPT."""
    # Load the image
    image = Image.open(input_image)

    prompt_instructions = (
        """
        Describe the image using as much detail as possible, 
        is it a painting, a photograph, what colors are predominant, 
        what is the image about?
        """
        if isinstance(input_text, tuple)
        else """
        Act as an expert in imagery descriptive analysis, 
        using as much detail as possible from the image, 
        respond to the following prompt:
        """ + input_text
    )

    prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    # Properly extract the response text
    if outputs and len(outputs) > 0 and "generated_text" in outputs[0]:
        match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
        if match:
            return match.group(1)
    raise WhisperProcessingError("Failed to generate image description")
