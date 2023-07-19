# -------------- with Kmeans --------------------------

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Aims and Scope metinlerini al
corpus = [journal["Aims and Scope"] for journal in data]

# Metinleri TF-IDF vektörlerine dönüştür
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# K-Means modelini eğitme
k = 5  # Küme sayısı
model = KMeans(n_clusters=k)
model.fit(X)

# Test örneği için tavsiyeleri almak
test_sample = [str(input("Aranacak text:"))]  # Test örneği Aims and Scope metni
test_sample_vector = vectorizer.transform(test_sample)
cluster_id = model.predict(test_sample_vector)[0]

# Tavsiye sonuçlarını yazdırma
for i, label in enumerate(model.labels_):
    if label == cluster_id:
        print(data[i]["Journal Name"])

# Doğruluk metriklerini hesaplama
true_labels = [journal["Journal Name"] for journal in data]  # Gerçek sınıf etiketleri
predicted_labels = [data[i]["Journal Name"] for i in model.labels_]  # Küme etiketleri

precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)