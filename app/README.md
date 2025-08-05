# Digit Recognition FastAPI Service

This FastAPI service provides endpoints for recognizing handwritten digits using a trained MNIST model.

## Features

- **Single Image Prediction**: Upload a single image to get digit prediction
- **Batch Prediction**: Upload multiple images for batch processing
- **Health Check**: Monitor API health and model status
- **Automatic Image Preprocessing**: Handles image preprocessing to match model requirements

## API Endpoints

### GET `/`
Root endpoint with API information and available endpoints.

### GET `/health`
Health check endpoint to verify API status and model loading.

### POST `/predict`
Upload a single image file to predict the digit.

**Request**: Multipart form data with image file
**Response**: JSON with predicted digit, confidence, and filename

## Running the Service

### Prerequisites
- Python 3.8+
- All dependencies from `requirements.txt`
- Trained model file `mnist-latest.h5` in the parent directory

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Start the Server
```bash
# From the app directory
python server.py

# Or using uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## Testing the API

### Using the test script
```bash
python test_api.py
```

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/image.png"
```

### Using Python requests
```python
import requests

# Upload image for prediction
with open('image.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    result = response.json()
    print(f"Predicted digit: {result['predicted_digit']}")
    print(f"Confidence: {result['confidence']}")
```

## Image Requirements

The API accepts common image formats (PNG, JPG, JPEG, etc.). Images are automatically preprocessed:

1. **Border trimming**: Removes unnecessary white space
2. **Padding**: Adds white borders for better processing
3. **Grayscale conversion**: Converts to grayscale
4. **Color inversion**: Inverts colors to match MNIST format
5. **Resizing**: Resizes to 28x28 pixels for model input

## Response Format

### Single Prediction Response
```json
{
  "predicted_digit": 7,
  "confidence": 0.9876,
  "filename": "image.png",
  "message": "Predicted digit: 7 with 98.76% confidence"
}
```

### Batch Prediction Response
```json
{
  "total_files": 3,
  "processed_files": 3,
  "results": [
    {
      "filename": "digit_1.png",
      "predicted_digit": 1,
      "confidence": 0.9956,
      "message": "Predicted digit: 1 with 99.56% confidence"
    },
    {
      "filename": "digit_2.png",
      "predicted_digit": 2,
      "confidence": 0.9876,
      "message": "Predicted digit: 2 with 98.76% confidence"
    }
  ]
}
```

## Error Handling

The API includes comprehensive error handling:
- **400 Bad Request**: Invalid file type or malformed request
- **500 Internal Server Error**: Processing errors or model issues

## Model Information

- **Model Type**: Neural Network trained on MNIST dataset
- **Input Shape**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Model File**: `mnist-latest.h5` (loaded from parent directory) 