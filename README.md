# Diploma Title Generator

This is a small project using Machine Learning models to create new diploma titles based on the user input of wanted field/area.

Some diploma titles for a 5 year range are used as dataset to train the models.

Used ML models:

- LSTM
- World-level RNN
- GPT2
- Seq2Seq
- k-NearestNegihbor


- All models and tests are placed under diploma_generator_latest_notebook.ipynb file. You can test those using Google Colab or Jupiter notebook.

- To install all necessary libraries use:

```bash
pip install -r requirements.txt
```

- Flask - Python Framework is used for a simple app with API endpoints for generating and recommanding diploma titles.

    To install it use the below command:

```
pip install flask
```

- Main file is diploma_generator_app.py. Start application using flask command or

```bash
python -m flask.
```

- After starting the flask app you can use curl command to test API calls and get different results based on your input data.
```bash
curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d "{\"seed_text\":\"deep learning\", \"next_words\": 11}"
```

Postman tool is also another alternative to use for tests on API calls.

