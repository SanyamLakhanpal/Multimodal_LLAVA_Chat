from fastapi import FastAPI, UploadFile, File
from handlers_async import img2txt

app = FastAPI()

@app.post("/image-to-text")
async def image_to_text(input_text: str, file: UploadFile = File(...)):
    content = await file.read()
    description = await img2txt(input_text, content)
    return {"description": description}
