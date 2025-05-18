import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Flatten, concatenate, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import gc
import pickle
import matplotlib.pyplot as plt

# Paths to CSV and image directory
csv_path = "C:/Users/divya/Desktop/iam_words/cleaned_words_mapped.csv"
image_dir = "C:/Users/divya/Desktop/iam_words/words"

# Load dataset
df = pd.read_csv(csv_path)

# Check for missing values
if df.isnull().sum().any():
    print("Dataset contains missing values. Handle them before proceeding.")
else:
    print("No missing values in the dataset.")

def join_path(word_id):
    if word_id[7].isalpha():
        return os.path.join(image_dir, word_id[0:3], word_id[0:8], word_id)
    else:
        return os.path.join(image_dir, word_id[0:3], word_id[0:7], word_id)

# Construct image paths
df['image_path'] = df['word_id'].apply(join_path)

# Add file extension and validate file existence
df['image_full_path'] = df['image_path'] + ".png"
df = df[df['image_full_path'].apply(os.path.exists)]
print(f"Number of valid entries: {len(df)}")

df = df.drop('image_full_path', axis=1)

# Image processing parameters
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3

# Function to load and preprocess images
def load_and_preprocess_image(path):
    try:
        img = Image.open(path + ".png").convert('RGB')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

# Apply the function to all image paths
df['image_array'] = df['image_path'].apply(load_and_preprocess_image)

# Stack image arrays into a single array
X_images = np.stack(df['image_array'].values)
print(f"Image data shape: {X_images.shape}")

# Tokenizing transcription data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['transcription'].tolist())

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary Size: {vocab_size}")

# Convert transcriptions to sequences
text_sequences = tokenizer.texts_to_sequences(df['transcription'])

# Maximum sequence length
max_seq_length = df['transcription'].apply(lambda x: len(x.split())).max()
print(f"Maximum Sequence Length: {max_seq_length}")

# Pad sequences
X_text = pad_sequences(text_sequences, maxlen=max_seq_length, padding='post')

# Extract labels
y_labels = X_text[:, 0]

# Train-test split
X_img_train, X_img_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_images, X_text, y_labels, test_size=0.2, random_state=42
)

# ResNet50 for feature extraction
image_input = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name='image_input')
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
base_model.trainable = False
x = base_model(image_input, training=False)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# **Bi-LSTM** for text processing
text_input = Input(shape=(max_seq_length,), name='text_input')
embedding_dim = 64
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length)(text_input)
y = Bidirectional(LSTM(64))(embedding_layer)
y = Dense(64, activation='relu')(y)
y = Dropout(0.5)(y)

# Combine features
combined = concatenate([x, y])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(vocab_size, activation='softmax')(z)

# Create the model
model = Model(inputs=[image_input, text_input], outputs=z)

# Model summary
model.summary()

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ✅ Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    [X_img_train, X_text_train],
    y_train,
    validation_data=([X_img_test, X_text_test], y_test),
    epochs=50,
    batch_size=8,
    callbacks=[early_stopping]
)

# ✅ Save the model 
model.save("complete_model.h5")

# ✅ Save training history
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# Save the tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer, model and history saved successfully.")

# ✅ Plot Training Curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# ✅ Prediction functions
def preprocess_single_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_on_image(image_path, text_input):
    processed_image = preprocess_single_image(image_path)
    if processed_image is None:
        print("Image processing failed.")
        return

    text_sequence = tokenizer.texts_to_sequences([text_input])
    padded_text_sequence = pad_sequences(text_sequence, maxlen=max_seq_length, padding='post')

    prediction_prob = model.predict([processed_image, padded_text_sequence])
    predicted_label_index = np.argmax(prediction_prob, axis=1)[0]

    word_index_to_token = {v: k for k, v in tokenizer.word_index.items()}
    predicted_word = word_index_to_token.get(predicted_label_index, "<Unknown>")

    print(f"Predicted Word: {predicted_word}")

# ✅ Example predictions
example_images = [
    "C:/Users/divya/Desktop/Sir.png",
    "C:/Users/divya/Desktop/this.png",
    "C:/Users/divya/Desktop/get.png",
    "C:/Users/divya/Desktop/night.png"
]
text_input = "Example transcription input"
for img_path in example_images:
    predict_on_image(img_path, text_input)

# ✅ Clear memory
tf.keras.backend.clear_session()
gc.collect()
