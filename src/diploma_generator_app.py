from flask import Flask, request, jsonify
from diploma_generator_model import model, vectorizer

# Initialize Flask app
app = Flask(__name__)

# Define route for generating diploma thesis
@app.route('/generate_thesis', methods=['POST'])
def generate_thesis():
    # Get user input from request
    user_input = request.json
    user_areas_of_interest = user_input.get('areas_of_interest')

    # Preprocess and convert user input into numerical features
    user_input_features = vectorizer.transform([' '.join(user_areas_of_interest)])

    # Generate a new thesis title based on user input
    generated_title = model.predict(user_input_features)

    # Return generated title as JSON response
    return jsonify({'generated_title': generated_title.tolist()})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)