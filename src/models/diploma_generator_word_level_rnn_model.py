import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout

# Sample data loading
dataset_path_file = os.path.join("..", "dataset", "diploma.xlsx")
# Step 1: Load data
df = pd.read_excel(dataset_path_file)  # Replace with your actual file path

# Step 2: Text preprocessing
# Combine all text data that the model should learn from
texts = df['tema'].tolist()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Including zero index
print('Vocabulary Size:', vocab_size)

# Create input-output pairs
sequence_length = 10  # Length of input sequences
inputs, outputs = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        input_seq, output_seq = seq[:i], seq[i]
        input_seq = pad_sequences([input_seq], maxlen=sequence_length, padding='pre')[0]
        output_seq = to_categorical([output_seq], num_classes=vocab_size)[0]
        inputs.append(input_seq)
        outputs.append(output_seq)

inputs = np.array(inputs)
outputs = np.array(outputs)

# Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=sequence_length))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(inputs, outputs, epochs=100, verbose=1)

# Function to generate text
def generate_text(seed_text, next_words, max_sequence_len, model, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Generate text
new_text = generate_text("Wireless network", 10, sequence_length, model, tokenizer)
print(new_text)

#history = model.fit(X, Y, epochs=50, validation_split=0.2)
history = model.fit(inputs, outputs, epochs=100, validation_split=0.2)

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

# Call to plot function
plot_history(history)

from collections import Counter

def measure_diversity(texts):
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counts)
    diversity_score = unique_words / total_words
    return diversity_score

# Example usage with a list of generated themes
#generated_themes = [generate_text("Deep learning", 5, model, sequence_length) for _ in range(10)]
diversity_score = measure_diversity(new_text)
print(f"Diversity Score: {diversity_score:.3f}")
