import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

dataset_path_file = os.path.join("dataset", "diploma.xlsx")
# Step 1: Load data
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
sequence_length = 10
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
model.add(Embedding(vocab_size, 10, input_length=sequence_length))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# Summarize the model
model.summary()

# Fit the model
model.fit(X, y, epochs=100, verbose=2, callbacks=[EarlyStopping(monitor='loss', patience=5)])

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
#print(generate_text("web", 9, model, sequence_length))