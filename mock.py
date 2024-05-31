from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from uuid import uuid4

app = FastAPI()

MODEL_DIR = 'models'
model = None
model_version = 'restv2.h5'

# Define tattoo types
TATTOO_TYPES = [
    "คน", "จิ้งจก", "นก", "มังกร+พญานาค", "แมงป่อง",
    "เสือ", "อักขระ", "ควาย", "พ่อ", "แม่",
    "ยันต์เก้ายอด", "หัวใจ", "ดอกไม้", "อื่นๆ (กราฟฟิค)"
]

# Mocking the load_model function
def load_model(version):
    global model
    model_path = os.path.join(MODEL_DIR, version)
    if os.path.exists(model_path):
        model = "mock_model"  # Mocked model loading
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

# Request and Response Schemas
class PredictRequest(BaseModel):
    image_path: str

class PredictResponse(BaseModel):
    tattoo_type: str

class TrainResponse(BaseModel):
    job_id: str

class TrainStatusResponse(BaseModel):
    status: str

class ModelsResponse(BaseModel):
    models: List[str]

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    image_path = request.image_path
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Mocking image processing and prediction
    # image = tf.io.decode_image(tf.io.read_file(image_path), channels=3)
    # image = tf.image.resize(image, [224, 224])  # Adjust based on model input size
    # image = tf.expand_dims(image, 0)  # Add batch dimension

    # Mock prediction
    predicted_class = 0  # Mocked prediction result
    tattoo_type = TATTOO_TYPES[predicted_class]

    return {"predicted_class": int(predicted_class), "tattoo_type": tattoo_type}

@app.post("/train", response_model=TrainResponse)
async def train():
    training_status['status'] = 'running'

    # Mocking training process
    training_status['status'] = 'completed'
    return {"job_id": job_id}

@app.get("/train/status", response_model=TrainStatusResponse)
async def train_status():
    return training_status

@app.get("/models", response_model=ModelsResponse)
async def get_models():
    return {"models": list_models()}
