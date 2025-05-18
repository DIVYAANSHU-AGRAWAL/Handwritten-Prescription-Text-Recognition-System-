# Overview
This project implements an integrated deep learning pipeline to recognize handwritten words from images using a combination of image features extracted via a pretrained ResNet50 and textual features processed with a Bidirectional LSTM. The model is trained on the IAM Handwriting dataset (or similar) with images of handwritten words and their corresponding transcriptions.
The goal is to predict the transcription tokens given an image of a handwritten word and partial textual input.

# Features
- Image preprocessing and loading pipeline for handwritten word images.
- Text tokenization and sequence padding for transcription data.
- Dual-input deep learning model combining:
- ResNet50 for image feature extraction (frozen weights).
- Bidirectional LSTM for text sequence modeling.
- Early stopping to prevent overfitting.
- Training history visualization (loss and accuracy).
- Save and load trained model and tokenizer.
- Predict function for new images and textual input.
- Memory optimization after training.

# Setup
## Prerequisites
-Python 3.8+
-TensorFlow 2.x
-numpy
-pandas
-scikit-learn
-matplotlib
-pillow

# Usage
## Training the model
Update the file paths in train_model.py to point to your dataset CSV and image directories.

## Run the training script:
python scripts/train_model.py

### This will:

-Load and preprocess images and text.
-Build the dual-input model.
-Train the model with early stopping.
-Save the model, tokenizer, and training history.
-Plot training curves.

## Predicting on new images
Use the provided prediction functions in predict.py to load the trained model and tokenizer and make predictions on new handwritten word images.

# Dataset
The project uses a dataset CSV file containing word_id and transcription columns along with corresponding images stored in a directory structure. The script verifies and loads the valid images for training.

# Model Architecture
-Image branch: ResNet50 (pretrained on ImageNet, frozen) followed by Flatten, Dense(128), and Dropout.
-Text branch: Embedding layer → Bidirectional LSTM(64) → Dense(64) → Dropout.
-Combined: Concatenate image and text features → Dense(128) → Dropout → Dense output layer with softmax activation for token classification.

# Results
The model trains up to 50 epochs with early stopping based on validation loss. Training and validation accuracy and loss plots are generated to visualize the performance.

# Future Improvements
-Implement sequence-to-sequence models for full transcription prediction.
-Fine-tune the ResNet50 base model for improved image features.
-Add data augmentation for handwritten images.
-Experiment with different architectures like CNN + Transformer.
-Extend to multi-word or sentence recognition.

