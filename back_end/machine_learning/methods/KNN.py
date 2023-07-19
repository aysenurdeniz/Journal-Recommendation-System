# ------------------- with KNN -------------------

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Aims and Scope metinlerini ve etiketleri alın
corpus = [journal["Aims and Scope"] for journal in data]
labels = [journal["Journal Name"] for journal in data]

# TF-IDF vektörlerine dönüştürme
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Chi-Square özelliği seçimini uygulama
k = 1000  # Özellik sayısı
selector = SelectKBest(chi2, k=k)
X_selected = selector.fit_transform(X, labels)

# KNN modelini eğitme
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_selected, labels)

# Kullanıcıdan metin girişi al
user_input = input("Metin girin: ")

# Kullanıcının girdisini TF-IDF vektörüne dönüştürme
user_input_vector = vectorizer.transform([user_input])

# Seçilen özelliklere uygulama
user_input_selected = selector.transform(user_input_vector)

# En yakın komşuları bulma
distances, indices = knn.kneighbors(user_input_selected)

# Tavsiye sonuçlarını yazdırma
print("Tavsiye Edilen Dergiler:")
for index in indices.flatten():
    print(data[index]["Journal Name"])

# Doğruluk metriğini hesaplama
true_labels = labels  # Gerçek sınıf etiketleri
predicted_labels = knn.predict(X_selected)  # Öngörülen sınıf etiketleri

accuracy = accuracy_score(true_labels, predicted_labels)
print("Doğruluk:", accuracy)

# Hassasiyet metriğini hesaplama
precision = precision_score(true_labels, predicted_labels, average='weighted')
print("Precision:", precision)

# Duyarlılık metriğini hesaplama
recall = recall_score(true_labels, predicted_labels, average='weighted')
print("Recall:", recall)

# F1 skoru metriğini hesaplama
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print("F1 Score:", f1)