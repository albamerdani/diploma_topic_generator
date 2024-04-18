import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPRegressor

# Placeholder for data collection code
thesis_titles = [
    "A Study of Network Security Protocols",
    "Web Application Development Best Practices",
    "Introduction to Cryptography Techniques",
    "Machine Learning Algorithms for Predictive Analysis",
    "Internet of Things (IoT) Applications in Healthcare",
    "Cloud Computing Architectures and Solutions",
    "Operating System Design Principles",
    "Game Development Strategies for Mobile Platforms",
    "Mobile App Development Trends and Technologies"
]

y_years = [2015, 2018, 2017, 2016, 2019, 2020, 2019, 2018, 2017]
# Verify the target labels (years)

# Preprocess the titles by removing special characters and tokenizing
def preprocess_title(title):
    title = re.sub(r'[^\w\s]', '', title.lower())
    return title.split()

# Apply preprocessing to all thesis titles
preprocessed_titles = [preprocess_title(title) for title in thesis_titles]

# Convert preprocessed titles into numerical features using Bag-of-Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([' '.join(title) for title in preprocessed_titles])

# Train a neural network model for predicting the year
year_model = MLPRegressor(hidden_layer_sizes=(3,1), activation='tanh', solver='adam', max_iter=2000)
year_model.fit(X, y_years)  # Assuming you have target labels y, such as publication year

# Process user input areas of interest
user_areas_of_interest = ["network", "machine learning", "cloud"]

# Preprocess and convert user input into numerical features
user_input_features = vectorizer.transform([' '.join(user_areas_of_interest)])

# Train a separate model for predicting the area
area_model = MLPRegressor(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=2000)
area_labels = np.arange(9)  # Assuming there are 9 areas
area_model.fit(X, area_labels)

# Predict the year and area
predicted_year = year_model.predict(user_input_features)
predicted_area = int(round(area_model.predict(user_input_features)[0]))  # Round and convert to integer

print("Predicted year:", predicted_year)
print("Predicted area index:", predicted_area)
