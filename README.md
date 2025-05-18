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
- Python 3.8+
- TensorFlow 2.x
- numpy
- pandas
- scikit-learn
- matplotlib
- pillow

# How to Run the Project?
## 1. Setup Environment
Make sure you have Python installed (preferably 3.7+).

Install required libraries:
pip install numpy pandas tensorflow pillow scikit-learn matplotlib opencv-python

## 2. Prepare Your Dataset
- Place your CSV file (cleaned_words_mapped.csv) with columns word_id and transcription on your local machine.
- Store images in the specified folder structure inside iam_words/words/ as expected by your join_path() function.

## 3. Train the Recognition Model
### Run the first script you provided to:

- Load and preprocess images and transcription data.
- Tokenize text and pad sequences.
- Build the multimodal model (image + text).
- Train the model on your dataset.
- Save the trained model (complete_model.h5), tokenizer (tokenizer.pkl), and training history (history.pkl).
- Visualize training curves.
### Note: Training will require a GPU for speed and might take a few hours depending on dataset size.

## 4. Segment Words from Handwritten Prescription Image
### Use the second script for word segmentation from prescription images:

- Upload an image using Google Colab file uploader or load from disk.
- Convert to grayscale and apply adaptive thresholding.
- Dilate to join words (not letters).
- Detect contours as word bounding boxes.
- Extract and save individual word images in a folder (Segmented_Words).
- Display word bounding boxes for verification.

## 5. Load Model and Tokenizer for Prediction
- Load your saved model and tokenizer files from disk.
- Define preprocessing functions to prepare input images and text for the model.
- Use your predict_on_image() or predict_on_folder() functions to get predictions for segmented words.

## 6. Run Predictions on Segmented Words
- Pass the folder containing segmented word images (Segmented_Words) to the prediction function.
- Provide a sample or heuristic text input to the text input branch of the model (e.g., some guess about the word context or fixed placeholder).
- View the top-k predictions for each word image.

## 7. Cleanup
Clear TensorFlow session and collect garbage to free memory after prediction/training.

# Usage
## Training the model
Update the file paths in train_model.py to point to your dataset CSV and image directories.

## Run the training script:
python scripts/train_model.py

### This will:

- Load and preprocess images and text.
- Build the dual-input model.
- Train the model with early stopping.
- Save the model, tokenizer, and training history.
- Plot training curves.

## Predicting on new images
Use the provided prediction functions in predict.py to load the trained model and tokenizer and make predictions on new handwritten word images.

# Dataset
### IAM Handwriting Database
- Widely-used dataset of handwritten English text for training and testing handwriting recognition systems.
- Contains data from **657 writers**.
- Includes:
  - **1,539 pages** of scanned handwritten text.
  - **5,685 labeled sentences**.
  - **13,353 labeled text lines**.
  - **115,320 labeled words**.
- Images are scanned at **300 dpi**, saved as grayscale PNG files.
- Comes with XML metadata files containing segmentation and annotation information.
- Supports research in handwriting recognition, writer identification, and verification.

**Dataset source:** [IAM Handwriting Database on Kaggle](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)

The project uses a dataset CSV file named cleaned_words_mapped.csv containing word_id and transcription columns along with corresponding images stored in a directory structure. The script verifies and loads the valid images for training.

# Model Architecture
- Image branch: ResNet50 (pretrained on ImageNet, frozen) followed by Flatten, Dense(128), and Dropout.
- Text branch: Embedding layer → Bidirectional LSTM(64) → Dense(64) → Dropout.
- Combined: Concatenate image and text features → Dense(128) → Dropout → Dense output layer with softmax activation for token classification.

# Results
The model trains up to 50 epochs with early stopping based on validation loss. Training and validation accuracy and loss plots are generated to visualize the performance.

# Future Improvements
- Implement sequence-to-sequence models for full transcription prediction.
- Fine-tune the ResNet50 base model for improved image features.
- Add data augmentation for handwritten images.
- Experiment with different architectures like CNN + Transformer.
- Extend to multi-word or sentence recognition.

