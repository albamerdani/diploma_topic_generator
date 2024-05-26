import pandas as pd
import numpy as np
import string
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Sample data loading
dataset_path_file = os.path.join("dataset", "diploma.xlsx")
# Step 1: Load data
df = pd.read_excel(dataset_path_file)  # Replace with your actual file path

# Define a clean text function that handles non-string data
def clean_text(data):
    if isinstance(data, str):
        data = "".join(v for v in data if v not in string.punctuation).lower()
        data = data.encode("utf8").decode("ascii", 'ignore')
    else:
        data = ""
    return data

sentence_start = "sentence_start"
sentence_end = "sentence_end"

# Apply text processing to the 'Theme' and 'Field' columns
df['cleaned_theme'] = df['theme'].apply(clean_text)
df['cleaned_field'] = df['field'].apply(clean_text)

# Append special tokens
#df['cleaned_theme'] = sentence_start + " " + df['cleaned_theme'] + " " + sentence_end

# Initialize and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_theme'].tolist() + df['cleaned_field'].tolist())
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
df['theme_seq'] = tokenizer.texts_to_sequences(df['cleaned_theme'])
df['field_seq'] = tokenizer.texts_to_sequences(df['cleaned_field'])

# Determine maximum sequence length
max_length_theme = max(df['theme_seq'].apply(len))
max_length_field = max(df['field_seq'].apply(len))
max_length = max(max_length_theme, max_length_field)

# Pad sequences
df['theme_seq'] = list(pad_sequences(df['theme_seq'], maxlen=max_length, padding='post'))
df['field_seq'] = list(pad_sequences(df['field_seq'], maxlen=max_length, padding='post'))

# Prepare data for the model
encoder_input_data = np.array(df['field_seq'].tolist())
decoder_input_data = np.array(df['theme_seq'].tolist())
decoder_target_data = np.roll(decoder_input_data, -1, axis=1)
decoder_target_data = to_categorical(decoder_target_data, num_classes=vocab_size)

# Model architecture
embedding_size = 50

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, embedding_size)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(50, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size, embedding_size)(decoder_inputs)
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define and compile the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32, epochs=50, validation_split=0.2)

# Define the encoder model for later use in inference
encoder_model = Model(encoder_inputs, encoder_states)


encoder_model.summary()

# Decoder setup for inference (needs defining similar to the training setup but step by step)
decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)


# Model to predict
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1 with only the 'sentence_start'.
    target_seq = np.zeros((1, 1))
    #target_seq[0, 0] = tokenizer.word_index['sentence_start']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word.get(sampled_token_index, 'UNK')
        decoded_sentence += ' ' + sampled_char

        # Exit condition: either hit max length or find stop token.
        #sampled_char == 'sentence_end' or
        if (len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


# Example decoding
test_seq = ['web', 'security', 'network']#encoder_input_data[0:1]  # Simulating a field input
print(decode_sequence(test_seq))
