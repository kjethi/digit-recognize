# find all pdf in folder
from os import path,makedirs
from glob import glob  
# PyMuPDF
from PIL import Image, ImageOps, ImageChops

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

model = load_model('mnist-latest.h5')

# Settings for project
dpi = 300
box_min_area = 8375
box_max_area = 8485
ractangle_tolerance = int(dpi * 0.02)
cut_edge = 10


def predict_digit(img):
    img = np.array(img)
    img = img.reshape(1,28,28)
    res = model.predict(img/255)
    return np.argmax(res), max(res)


# find files in folder
def find_ext(dr, ext):
    print("*.{}".format(ext))
    return glob(path.join(dr,"*.{}".format(ext)))
    
def trim_borders(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    
    return image

def pad_image(image):
    return ImageOps.expand(image, border=20, fill='#fff')

def to_grayscale(image):
    return image.convert('L')

def invert_colors(image):
    return ImageOps.invert(image)

def resize_image(image):
    return image.resize((28, 28), Image.BILINEAR)

def detect_and_extract_text(image_path):
    img = Image.open(image_path)
    # im = Image.fromarray(img) 
    img_gray = trim_borders(img)
    img_gray = pad_image(img_gray)
    img_gray = to_grayscale(img_gray)
    img_gray = invert_colors(img_gray)
    img_gray = resize_image(img_gray)

    predictedDigit = predict_digit(img_gray)
    print(f'Predicted Digit of "{image_path}" is : {predictedDigit[0]}')
   

# find ".jpg" files in "imgs" folder
imageFiles = find_ext('imgs','png')
print(f'imageFiles: {imageFiles}')

for img_file in imageFiles:
    detect_and_extract_text(img_file)