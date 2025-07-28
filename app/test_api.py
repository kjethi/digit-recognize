import requests
import os

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:")
    print(response.json())
    print()

def test_root():
    """Test the root endpoint."""
    response = requests.get(f"{BASE_URL}/")
    print("Root Endpoint Response:")
    print(response.json())
    print()

def test_single_prediction(image_path):
    """Test single image prediction."""
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"Single Prediction Response for {image_path}:")
    print(response.json())
    print()

def test_batch_prediction(image_paths):
    """Test batch image prediction."""
    files = []
    for path in image_paths:
        if os.path.exists(path):
            files.append(('files', open(path, 'rb')))
        else:
            print(f"Image file not found: {path}")
    
    if files:
        response = requests.post(f"{BASE_URL}/predict-batch", files=files)
        print("Batch Prediction Response:")
        print(response.json())
        print()
        
        # Close all file handles
        for _, file in files:
            file.close()

if __name__ == "__main__":
    print("Testing Digit Recognition API")
    print("=" * 40)
    
    # Test basic endpoints
    test_health()
    test_root()
    
    # Test with sample images from the imgs folder
    sample_images = [
        "../imgs/digit_1.png",
        "../imgs/digit_2.png",
        "../imgs/digit_4.png",
        "../imgs/digit_7_003.png",
        "../imgs/digit_8.png"
    ]
    
    # Test single prediction
    if sample_images:
        test_single_prediction(sample_images[0])
    
    # Test batch prediction
    test_batch_prediction(sample_images) 