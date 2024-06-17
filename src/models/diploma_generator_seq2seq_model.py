import numpy as np
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path_file = os.path.join("..", "dataset", "diploma.xlsx")
df = pd.read_excel(dataset_path_file)  # Replace with your actual file path

# Select the column with the themes
texts = df['tema'].values

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Pad sequences
max_seq_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

# Split the data into training and validation sets
X_train, X_val = train_test_split(data, test_size=0.2, random_state=42)

"""Build the Model"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

# Define the model
embedding_dim = 100
latent_dim = 256
vocab_size = len(word_index) + 1  # Plus 1 for padding

# Encoder
encoder_inputs = Input(shape=(None,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(x)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn encoder_input_data & decoder_input_data into decoder_target_data
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""Train the Model"""

import matplotlib.pyplot as plt
# Prepare the data for the decoder
decoder_input_data = np.zeros_like(X_train)
decoder_input_data[:, 1:] = X_train[:, :-1]

# Define targets
decoder_target_data = np.expand_dims(X_train, -1)

# Train the model
history = model.fit([X_train, decoder_input_data], decoder_target_data,
                    batch_size=64,
                    epochs=50,
                    validation_split=0.2)

#history = model.fit(inputs, outputs, epochs=50, validation_split=0.2)


if 'val_accuracy' in history.history:
    mean_val_accuracy = np.mean(history.history['val_accuracy'])
    print("Mean Validation Accuracy over all epochs:", mean_val_accuracy)

if 'val_loss' in history.history:
    mean_loss_accuracy = np.mean(history.history['val_loss'])
    print("Mean Validation Loss over all epochs:", mean_loss_accuracy)

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

"""Adjust Inference Models and Generate New Themes"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

latent_dim = 256  # Latent dimensionality of the encoding space
max_seq_length = 11  # Max sequence length to consider

# Define encoder model
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
encoder_embedded = encoder_embedding(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedded)
encoder_states = [state_h, state_c]

encoder_model = Model(encoder_inputs, encoder_states)

# Define decoder model
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
decoder_embedded = decoder_embedding(decoder_inputs)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedded, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1 with only the start character (index 0).
    target_seq = np.zeros((1, 1))

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        # Prepare the decoder input to be passed to the decoder
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = tokenizer.index_word.get(sampled_token_index, '')

        decoded_sentence.append(sampled_token)

        # Exit condition: either hit max length or find stop character (empty string).
        if sampled_token == '' or len(decoded_sentence) > max_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return ' '.join(decoded_sentence)

# Test the generation
test_seq = np.array([X_val[0]])  # Using an actual sequence from the validation data
new_theme = decode_sequence(test_seq)
print(new_theme)

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
diversity_score = measure_diversity(new_theme)
print(f"Diversity Score: {diversity_score:.3f}")