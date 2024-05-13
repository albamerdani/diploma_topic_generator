# diploma_topic_generator

pip install flask

curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d "{\"seed_text\":\"deep learning\", \"next_words\": 10}"
