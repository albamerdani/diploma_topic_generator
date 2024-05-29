import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import os

# Sample data loading
dataset_path_file = os.path.join("..", "dataset", "diploma.xlsx")
# Step 1: Load data
df = pd.read_excel(dataset_path_file)  # Replace with your actual file path

# Vectorization of the themes
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['theme'])

# Setup k-NN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

# Define a function to make recommendations based on user input
def recommend_themes(field1, field2):
    query = f"{field1} and {field2}"
    query_vec = vectorizer.transform([query])
    distances, indices = knn.kneighbors(query_vec)
    similarities = 1 - distances

    # Plotting the similarities
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(similarities[0])), similarities[0], color='blue', alpha=0.7)
    plt.title('Cosine Similarities for the Nearest Neighbors')
    plt.xlabel('Neighbor Index')
    plt.ylabel('Cosine Similarity')
    plt.xticks(range(len(similarities[0])), labels=[titles[i] for i in indices[0]])
    plt.xticks(rotation=45, ha="right")
    plt.show()

    entries = []

    # Fetch the themes that are closest to the query
    for index in indices[0]:
        year = df['year'].iloc[index]
        theme = df['theme'].iloc[index]
        entries.append((year, theme))

    # Sort the list by year in descending order
    entries_sorted = sorted(entries, reverse=True, key=lambda x: x[0])

    # Print the sorted entries
    for entry in entries_sorted:
        print(entry[0], entry[1])

# Example usage
recommend_themes('network', 'security')

#English Input (theme)
#Wireless network security
#Implementing network security - case study.
#Improving encryption techniques to increase network security
#Wireless network security: Creating a rogue access point
#Implementation and configuration of a VOIP network.

#Albania Input (tema)
#Text classification through the application of neural networks
#Automatic verification of identity card credentials using Tesseract OCR and OpenCV in Python
#Detection of cyberbullying through deep learning in social networking platforms
#Study of disk I/O performance degradation due to interference between Virtual Machines in Cloud Computing
#Comparison of machine learning algorithms for distinguishing anomalies