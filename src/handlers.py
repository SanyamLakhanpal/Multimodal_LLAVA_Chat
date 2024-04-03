from typing import Tuple, Union
from PIL import Image
from gtts import gTTS
import re
import datetime
import whisper
import torch
from transformers import BitsAndBytesConfig, pipeline
import yaml

from exceptions import WhisperProcessingError

with open('constants.yaml') as f:
    constants = yaml.safe_load(f)

quantization_config = BitsAndBytesConfig(**constants['quantization_config'])

model_id = constants['model_id']

pipe = pipeline("image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config})

@cached()  # Use default LRU_CACHE_SIZE and LRU_CACHE_TTL
def img2txt(input_text: Union[str, Tuple], input_image: str) -> str:
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

def transcribe(audio: str) -> str:
    """Convert audio to text using Whisper."""
    if not audio:
        return ''

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text

def text_to_speech(text: str, file_path: str) -> str:
    """Convert text to speech using gTTS."""
    language = 'en'
    audioobj = gTTS(text=text, lang=language, slow=False)
    audioobj.save(file_path)
    return file_path

def write_history(text: str):
    """Write processing history to a log file."""
    tstamp = datetime.datetime.now()
    tstamp = str(tstamp).replace(' ','_')
    logfile = f'{tstamp}_log.txt'
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()
