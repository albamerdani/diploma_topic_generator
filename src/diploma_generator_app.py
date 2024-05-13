from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from simpletransformers.language_generation import LanguageGenerationModel

app = Flask(__name__)

# Load the model (ensure this is in the same directory or specify the path)
model = load_model('thesis_generator_model.h5')
tokenizer_path = 'tokenizer.pkl'

# Load your tokenizer
import pickle

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)


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

# API for LSTM model
@app.route('/generate_thesis', methods=['POST'])
def generate_thesis():
    data = request.get_json()
    seed_text = data['seed_text']
    next_words = int(data.get('next_words', 8))  # Default to 5 words if not specified
    max_sequence_len = 10  # You should adjust this based on how you trained your model
    generated_text = generate_text(seed_text, next_words, model, max_sequence_len)
    return jsonify({'generated_text': generated_text})


# API for GPT2 model that is trained with DB
@app.route('/generate_gpt', methods=['POST'])
def generate_gpt():
    data = request.get_json()
    prompt = data.get('prompt', '')
    generated_text = model.generate(prompt=prompt, max_length=50, num_return_sequences=1)
    return jsonify({'generated_text': generated_text[0]})


if __name__ == "__main__":
    app.run(debug=False)
