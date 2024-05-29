# Jaundice Detection System using Computer Vision and Machine Learning

## Features
- User registration and login
- Profile management
- Real-time jaundice detection from sclera images
- Admin functionalities for user and image management
- Intuitive user interface

## System Architecture
The system is built using a Convolutional Neural Network (CNN) to analyze sclera images for jaundice detection. The architecture includes:
- Input Layer: Accepts images of size 100x100 pixels with 3 color channels.
- Convolutional Layers: Extract features from images.
- Pooling Layers: Reduce spatial dimensions.
- Flatten Layer: Converts 3D features to a 1D vector.
- Dense Layers: Perform final classification.
