import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Sample data loading
df = pd.read_excel('diploma.xlsx')

# Vectorization of the themes
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['theme']) #testime dhe me tema per temat shqip

# Setup k-NN
# experiment with the number of neighbors (n_neighbors) and the distance metric (e.g., Euclidean, Manhattan, cosine) to optimize the recommendations.
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

# Define a function to make recommendations based on user input
def recommend_themes(field1, field2):
    query = f"{field1} and {field2}"
    query_vec = vectorizer.transform([query])
    distances, indices = knn.kneighbors(query_vec)

    # Fetch the themes that are closest to the query
    for index in indices[0]:
        print(df['theme'].iloc[index])

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