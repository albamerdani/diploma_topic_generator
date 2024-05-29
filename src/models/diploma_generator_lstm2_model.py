import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from transformers import TrainerCallback
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from matplotlib import pyplot as plt

class HistoryCallback(TrainerCallback):
    "A custom callback that records loss and other metrics."
    def __init__(self):
        self.history = {'loss': [], 'perplexity': []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Logs might include loss and other metrics like learning rate
        if 'loss' in logs:
            self.history['loss'].append(logs['loss'])
        if 'perplexity' in logs:  # Ensure your perplexity metric is logged appropriately
            self.history['perplexity'].append(logs['perplexity'])

# Initialize the callback
history = HistoryCallback()

def perplexity(labels, logits):
    return tf.exp(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)))


# Define perplexity metric for Keras
def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.exp(cross_entropy)
    return perplexity

# Load data
dataset_path_file = os.path.join("..", "dataset", "diploma.xlsx")
df = pd.read_excel(dataset_path_file)  # Replace with your actual file path

# Text preprocessing
all_text = df['theme'].str.cat(sep=' ')  # Assuming 'theme' is the correct column name

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([all_text])
encoded = tokenizer.texts_to_sequences([all_text])[0]

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size:', vocab_size)

# Create sequences
sequence_length = 11
sequences = []
for i in range(sequence_length, len(encoded)):
    sequence = encoded[i-sequence_length:i+1]
    sequences.append(sequence)

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = np.eye(vocab_size)[y]  # One hot encoding

# Model setup
model = Sequential([
    Embedding(vocab_size, 11, input_length=sequence_length),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dense(50, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy', perplexity])

# Model summary
model.summary()
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)  # Make sure vocab_size is correctly set to 1007

# Fit the model

#history =
model.fit(X, y, batch_size=32, epochs=50, validation_split=0.2, callbacks=[EarlyStopping(monitor='loss', patience=5)])

#model.fit(X, y, epochs=50, verbose=2, callbacks=[EarlyStopping(monitor='loss', patience=5)])
#history = model.fit(X, y, epochs=50, validation_split=0.2)
#model.fit(X, y, epochs=50, validation_split=0.2)

# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        output_word = tokenizer.index_word.get(predicted_index, 'UNK')
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage
field1 = 'web'
field2 = 'security'
seed_text = f"{field1} and {field2}"
new_theme = generate_text(seed_text, 11, model, sequence_length)
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
generated_themes = [generate_text("Deep learning", 11, model, sequence_length) for _ in range(10)]
diversity_score = measure_diversity(generated_themes)
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

# Call to plot function
plot_history(history)