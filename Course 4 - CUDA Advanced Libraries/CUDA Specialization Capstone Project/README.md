# CUDA Specialization Capstone Project

## Overview

This project uses Tensorflow with cuDNN for GPU Acceleration to train a Convolutional Neural Network to recognize planetary objects in space.

**Key Features:**
- GPU-accelerated neural network training using TensorFlow with cuDNN
- Binary classification CNN model trained to detect planetary objects in images
- Trained on a dataset of 10,000 total images (with augmentation) to accurately recognize planetary objects

**Use Case:**
A CNN (convolutional neural network) is trained on a labeled dataset of planetary images using TensorFlow to detect whether a planetary object is present in a given image. This is especially helpful in automating space telescope and satellite image processing workflows to remove needing to manually filter out potential planetary objects by hand, which can be time consuming as they take 1000s of pictures.

## Installation

From the Coursera lab environment terminal, clone the repository using the following command:

```bash
git clone https://github.com/BrendanCook44/GPU-Programming-Specialization.git
```

Navigate to the project directory:

```bash
cd GPU-Programming-Specialization/
cd Course\ 4\ -\ CUDA\ Advanced\ Libraries/
cd CUDA\ Specialization\ Capstone\ Project/
```

## Running the Application

First, set up the environment by running the setup script. This will install all necessary packages and dependencies:

```bash
bash setup-environment.sh
```

The setup process may display a couple warnings which can be safely ignored. The setup process will take a couple minutes.

Then, run the application:

```bash
bash run-application.sh
```

The application will load the trained model and classify every image in the `data/Application/` directory, determining whether each image contains a planetary object or not.

A set of application images are provided by default. You are free to add your own test images to `data/Application/` for the model to evaluate — simply place any `.jpg`, `.jpeg`, or `.png` images in that directory before running the application.

## Code Organization

```data/```
Contains the training and application image datasets. Training images are organized into `Positive/` (planetary objects) and `Negative/` (non-planetary) categories. The `Application/` subdirectory holds images for inference — you can add your own images here for the model to evaluate.

```libs/```
CUDA helper libraries required for the project:
- CUDA helper functions (`helper_cuda.h`, `helper_string.h`, etc.)
- NPP utility classes
- OpenGL headers in `GL/` for graphics support

```src/```
Source code for the application:
- `application.py` - Main application script for running inference on images
- `model/train_model.py` - Training script for the CNN model
- `model/planetary_detector.keras` - Pre-trained model weights

```README.md```
Project documentation describing the project, its features, and usage instructions.

```setup-environment.sh```
Script to install all necessary packages and set up the Python virtual environment.

```run-application.sh```
Script to run the planetary object detection application.