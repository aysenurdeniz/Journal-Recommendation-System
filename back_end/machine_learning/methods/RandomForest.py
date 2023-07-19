import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Aims and Scope metinlerini ve etiketleri alın
corpus = [journal["Aims and Scope"] for journal in data]
labels = [journal["Journal Name"] for journal in data]

# Veri setini DataFrame'e dönüştürme
df = pd.DataFrame(data)

# Metin verilerini ve hedef değişkeni ayırma
X = df["Aims and Scope"]
y = df["Journal Name"]

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Random Forest modelini eğitme
rf = RandomForestClassifier()
rf.fit(X, y)

# Kullanıcıdan keyword girişi alma
user_input = "algorithm, system, printing"
user_input = user_input.split(",")

# Kullanıcının girdiği keyword'leri TF-IDF vektörlerine dönüştürme
user_input_vector = vectorizer.transform(user_input)

# Tahmin yapma
predictions = rf.predict(user_input_vector)

# Tahmin sonuçlarını ekrana yazdırma
for keyword, prediction in zip(user_input, predictions):
    print(f"{keyword}: {prediction}")

# Gerçek etiketleri elde etme
y_true = ["ACM TRANSACTIONS ON COMPUTER-HUMAN INTERACTION"] * len(user_input)  # ?

# Doğruluk metriğini hesaplama
accuracy = accuracy_score(y_true, predictions)
print("Doğruluk: ", accuracy)
