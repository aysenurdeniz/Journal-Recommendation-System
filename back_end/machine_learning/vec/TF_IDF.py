import json

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine, euclidean, cityblock, chebyshev, minkowski, mahalanobis
import numpy as np

# JSON dosyasını okuyun
from sklearn.metrics import mean_squared_error

with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)


class DistanceMethodsTFIDF:
    def __init__(self, data, user_input):
        self.data = data
        self.user_input = user_input
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            [entry['Journal Name'] for entry in self.data] + [self.user_input])

    def compute_similarity(self, distance_metric):
        self.similarities = []
        for i in range(len(self.data)):
            print("sim-for-in")
            similarity = distance_metric(self.tfidf_matrix[i].toarray(), self.tfidf_matrix[-1].toarray())
            self.similarities.append(similarity)

    def get_best_recommendations(self, top_k=3):
        recommendations = []
        for i in range(top_k):
            max_index = np.argmax(self.similarities)
            recommendations.append((self.data[max_index], self.similarities[max_index]))
            self.similarities[max_index] = -np.inf
        return recommendations


methods = [cosine, euclidean, cityblock, chebyshev, minkowski]

user_input = "information systems retrieval knowledge discovery and data mining"
texts = [entry["Aims and Scope"] for entry in data]

for distance_metric in methods:
    method = DistanceMethodsTFIDF(data, user_input)
    method.compute_similarity(distance_metric)
    recommendations = method.get_best_recommendations(top_k=3)

    print("\nEn İyi Tavsiyeler ({})".format(distance_metric.__name__))
    for i, recommendation in enumerate(recommendations, start=1):
        journal_name = recommendation[0]['Journal Name']
        similarity_score = recommendation[1]
        print(f"{i}. Journal: {journal_name}\nBenzerlik Skoru: {similarity_score:.3f}")

# RMSE sonuçlarını saklamak için bir liste oluşturun
rmse_results = []

for distance_metric in methods:
    print("-------1--------")
    method = DistanceMethodsTFIDF(data, user_input)
    print("-------2--------")
    method.compute_similarity(distance_metric)
    print("-------3--------")
    recommendations = method.get_best_recommendations(top_k=3)

    print("-------4--------")
    # Tahmin edilen benzerlik skorlarını alın
    predicted_scores = [entry[1] for entry in recommendations]

    print("-------5--------")
    # RMSE hesaplayın
    rmse = np.sqrt(mean_squared_error([1.4142135623730951] * len(predicted_scores), predicted_scores))

    print("-------6--------")
    # RMSE sonucunu listeye ekleyin
    rmse_results.append((distance_metric.__name__, rmse))

# RMSE sonuçlarını yazdırın
print("RMSE Sonuçları:")
for metric, rmse in rmse_results:
    print(f"{metric}: {rmse:.3f}")


# ---------------- TF-IDF w N-gram ------------------------------

import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import time

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Örnek kullanıcı girdisi
user_input = "information systems retrieval knowledge discovery and data mining"

# Verileri ayıklayın
texts = [entry["Aims and Scope"] for entry in data]

# İşlem süresini başlat
start_time = time.time()

# Bag-of-Words ve TF-IDF işlemlerini içeren bir pipeline oluşturun
pipeline = Pipeline([
    ('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3))),  # N-gram'lar
    ('tfidf', TfidfTransformer()),                   # TF-IDF
])

# Bag-of-Words ve TF-IDF işlemlerini uygulayın
tfidf_matrix = pipeline.fit_transform(texts + [user_input])
user_input_vector = tfidf_matrix[-1]  # Kullanıcı girdisinin vektör temsili

# Kozinüs benzerliği hesaplayın
cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix[:-1])

# İşlem süresini hesapla
end_time = time.time()
process_time = end_time - start_time

# En benzer 5 sonucu alın
top_indices = cosine_similarities.argsort()[0][-5:][::-1]
for i, index in enumerate(top_indices, start=1):
    journal_name = data[index]['Journal Name']
    similarity_score = cosine_similarities[0][index]
    print(f"{i}. Journal: {journal_name}\nBenzerlik Skoru: {similarity_score:.3f}")

# İşlem süresini yazdır
print(f"İşlem Süresi: {process_time:.4f} saniye")
