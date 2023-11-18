# digit-recognize
This is project for read single handwritten digit from image using by Tensorflow and keras 

Folder structure 
requirements.txt -> contains all the dependent library
digits_recognition.py -> File which is useful to creating model
main.py -> This script contains code for reading png files from imgs/ folder and use model which is create by digits_recognition.py file
imgs/ -> this folder contains single digit image in png format which do you want to read 

## Steps to follow 
Step 1 : Install all the dependencies related to project
    `pip install -r requirements.txt`

Step 2 : Create the model
    `python digits_recognition.py`
    above command create a model name "mnist-latest.h5" and store in same directory

Step 3 : Read digits using created model "mnist-latest.h5"
    `python main.py`
    above command read images from imgs/ folder and give out of predicted number


->  Note : Images must be in png format, If you want to read other format as well then you have to follow below step

main.py line no.71
`imageFiles = find_ext('imgs','png')`

change the second argument, for an example you have jpg file then `find_ext('imgs','jpg')`



    

## Features

- Import a HTML file and watch it magically convert to Markdown
- Drag and drop images (requires your Dropbox account be linked)
- Import and save files from GitHub, Dropbox, Google Drive and One Drive
- Drag and drop markdown and HTML files into Dillinger
- Export documents as Markdown, HTML and PDF