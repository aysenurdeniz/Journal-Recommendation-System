import json

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, euclidean, cityblock, chebyshev, minkowski, mahalanobis
import numpy as np

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)


class DistanceMethodsTFIDF:
    def __init__(self, data, user_input):
        self.data = data
        self.user_input = user_input

        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.data + [self.user_input])

    def compute_similarity(self, distance_metric):
        self.similarities = []
        for i in range(len(self.data)):
            similarity = distance_metric(self.tfidf_matrix[i].toarray(), self.tfidf_matrix[-1].toarray())
            self.similarities.append(similarity)

    def get_best_recommendations(self, top_k=3):
        recommendations = []
        for i in range(top_k):
            max_index = np.argmin(self.similarities)
            recommendations.append((self.data[max_index], self.similarities[max_index]))
            self.similarities[max_index] = np.inf  # Mark the used index with infinity
        return recommendations


methods = [cosine, euclidean, cityblock, chebyshev, minkowski]

user_input = "information systems retrieval knowledge discovery and data mining"
texts = [entry["Aims and Scope"] for entry in data]

for distance_metric in methods:
    method = DistanceMethodsTFIDF(texts, user_input)
    method.compute_similarity(distance_metric)
    recommendations = method.get_best_recommendations(top_k=3)

    print("\nEn İyi Tavsiyeler ({})".format(distance_metric.__name__))
    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{i}. Metin: {recommendation[0]}\nBenzerlik Skoru: {recommendation[1]:.3f}")

