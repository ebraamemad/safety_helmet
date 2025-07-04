
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io

from pridect import predict
app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    
    image = Image.open(file.file)
    predicted_image = predict(image)
    buffer=io.BytesIO()
    predicted_image.save(buffer, format="png")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
   