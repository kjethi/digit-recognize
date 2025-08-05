import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps, ImageChops
import io
import tensorflow as tf
from keras.models import load_model
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mnist-latest.h5')
model = load_model(model_path)

app = FastAPI(
    title="Digit Recognition API",
    description="A FastAPI service for recognizing handwritten digits using a trained MNIST model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],            # allow all methods (GET, POST, etc.)
    allow_headers=["*"],            # allow all headers
)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the uploaded image to match the model's expected input format.
    """
    # Trim borders
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        image = image.crop(bbox)
    
    # Pad image
    image = ImageOps.expand(image, border=20, fill='#fff')
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Invert colors (MNIST format)
    image = ImageOps.invert(image)
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.BILINEAR)
    image.save("output.png")
    return np.array(image)

def predict_digit(img_array: np.ndarray) -> tuple:
    """
    Predict the digit from the preprocessed image array.
    """
    img_array = img_array.reshape(1, 28, 28)
    prediction = model.predict(img_array/255)
    predicted_digit = np.argmax(prediction)
    confidence = float(max(prediction[0]))
    return predicted_digit, confidence, prediction

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Digit Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload an image to predict the digit",
            "/health": "GET - Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_digit_endpoint(file: UploadFile = File(...)):
    """
    Predict the digit from an uploaded image.
    
    Args:
        file: The image file to process
        
    Returns:
        JSON response with predicted digit and confidence
    """
    try:
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )

        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image.save("output1.png")
        # Preprocess the image
        processed_image = preprocess_image(image)

        
        # Predict the digit
        predicted_digit, confidence, prediction = predict_digit(processed_image)
        
        return JSONResponse({
            "predicted_digit": int(predicted_digit),
            "confidence": round(confidence, 4),
            "prediction" : prediction.tolist(),
            "filename": file.filename,
            "message": f"Predicted digit: {predicted_digit} with {confidence:.2%} confidence"
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )