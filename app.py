from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import os
from uuid import uuid4

app = FastAPI()

MODEL_DIR = 'models'
model = None
model_version = 'restv2.h5'

def load_model(version):
    global model
    model_path = os.path.join(MODEL_DIR, version)
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        raise HTTPException(status_code=404, detail="Model not found")

load_model(model_version)

# Utility function to get list of models
def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]

# To store training status
training_status = {
    'status': 'idle',
    'job_id': None
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    image = tf.io.decode_image(await file.read(), channels=3)
    image = tf.image.resize(image, [224, 224])  # Adjust based on model input size
    image = tf.expand_dims(image, 0)  # Add batch dimension

    predictions = model.predict(image)
    predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]

    return {"predicted_class": int(predicted_class)}

@app.post("/train")
async def train():
    job_id = str(uuid4())
    training_status['status'] = 'running'
    training_status['job_id'] = job_id

    # Mocking training process
    # Add your actual training logic here
    
    training_status['status'] = 'completed'
    return {"job_id": job_id}

@app.get("/train/status")
async def train_status():
    return training_status

@app.get("/models")
async def get_models():
    return {"models": list_models()}
