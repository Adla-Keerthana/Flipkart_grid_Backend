from fastapi import FastAPI, HTTPException
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Response model for the freshness score
class FreshnessResponse(BaseModel):
    freshness_score: float

# Pre-trained model loading (assuming MobileNetV2 as in the earlier example)
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)  # Assuming 3 freshness levels
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the trained weights (replace with the path to your trained model)
# model.load_weights("path_to_your_trained_model.h5")

async def predict_freshness(image_url: str) -> float:
    """
    Predict the freshness score of the product based on the image URL.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve image")

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess the image
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the freshness using the model
        predictions = model.predict(image_array)

        # Returning the freshness score (example: probability for 'Fresh')
        freshness_score = np.max(predictions)

        return freshness_score

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/freshness-prediction", response_model=FreshnessResponse)
async def freshness_prediction(image_url: str):
    """
    API endpoint for freshness prediction.
    """
    try:
        freshness_score = await predict_freshness(image_url)
        return FreshnessResponse(freshness_score=freshness_score)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
