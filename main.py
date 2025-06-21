
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import onnxruntime as ort
import numpy as np
import io

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession(r"E:\projects of camp\safety-helmet\yolo_optuna_mlflow\optuna_run5\weights\best.onnx")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = np.array(image.resize((128, 128))).transpose(2, 0, 1) / 255.0  # BGR to CHW, normalize
    img = np.expand_dims(img, axis=0).astype(np.float32)

    outputs = session.run(None, {"images": img})
    return {"detections": outputs[0].tolist()}
