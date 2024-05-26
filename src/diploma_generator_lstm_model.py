import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
from matplotlib import pyplot as plt

# Step 1: Load data
dataset_path_file = os.path.join("dataset", "diploma.xlsx")
df = pd.read_excel(dataset_path_file)  # Replace with your actual file path

# Step 2: Text preprocessing
# Combine all text data that the model should learn from
all_text = df['theme'].str.cat(sep=' ') #tema

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([all_text])
encoded = tokenizer.texts_to_sequences([all_text])[0]

# Determine vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size:', vocab_size)

# Create sequences
sequence_length = 11
sequences = []
for i in range(sequence_length, len(encoded)):
    sequence = encoded[i-sequence_length:i+1]
    sequences.append(sequence)

sequences = np.array(sequences)

# Split into X and y
X, y = sequences[:,:-1], sequences[:,-1]
y = np.eye(vocab_size)[y]  # One hot encoding

# Step 3: Model setup
model = Sequential()
model.add(Embedding(vocab_size, 11, input_length=sequence_length))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Summarize the model
model.summary()

# Fit the model
model.fit(X, y, epochs=50, verbose=2, callbacks=[EarlyStopping(monitor='loss', patience=5)])

# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

model.save('thesis_generator_model.h5')

# Generate a new theme
new_text=generate_text("web", 11, model, sequence_length)
print(new_text)

from collections import Counter

def measure_diversity(texts):
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counts)
    diversity_score = unique_words / total_words
    return diversity_score

diversity_score = measure_diversity(new_text)
print(f"Diversity Score: {diversity_score:.3f}")

# Function to plot the training and validation loss
def plot_history(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], 'r', label='Training loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    plt.title('Model Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # If accuracy is being tracked, plot that as well
    if 'accuracy' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
        plt.title('Model Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

history = model.fit(X, y, epochs=50, validation_split=0.2)
plot_history(history)
