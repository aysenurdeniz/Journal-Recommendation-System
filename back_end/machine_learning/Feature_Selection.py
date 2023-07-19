import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, RFE
from sklearn.linear_model import LogisticRegression
import json

# JSON dosyasından verileri yükleme
with open("C:\\Users\\anurd\\Downloads\\wos.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Metinleri çıkarma
texts = [entry["Aims and Scope"] for entry in data]
labels = [entry["Journal Name"] for entry in data]

# TF-IDF temsili
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # En yüksek 5000 TF-IDF değerine sahip özellikleri seçelim
tfidf_matrix = tfidf_vectorizer.fit_transform(texts).toarray()

# Chi-Kare Özellik Seçimi
print("Chi-Kare Özellik Seçimi:")
scores, p_values = chi2(tfidf_matrix, labels)
selected_features = np.argsort(scores)[-10:]  # En yüksek puanlı 10 özellik seçildi
features_selected = tfidf_matrix[:, selected_features]

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)

# K-NN sınıflandırıcısı
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Test verilerini kullanarak tahmin yapma
y_pred = knn.predict(X_test)

# Doğruluk değerini hesaplama ve yazdırma
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")

# Information Gain
print("Information Gain Özellik Seçimi:")
scores = mutual_info_classif(tfidf_matrix, labels)
selected_features = np.argsort(scores)[-10:]  # En yüksek puanlı 10 özellik seçildi
features_selected = tfidf_matrix[:, selected_features]
X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")

# SelectKBest
print("SelectKBest Özellik Seçimi:")
selector = SelectKBest(chi2, k=10)
features_selected = selector.fit_transform(tfidf_matrix, labels)

X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")

# SelectPercentile
print("SelectPercentile Özellik Seçimi:")
selector = SelectPercentile(mutual_info_classif, percentile=10)
features_selected = selector.fit_transform(tfidf_matrix, labels)

X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")

# SelectFpr
print("SelectFpr Özellik Seçimi:")
selector = SelectFpr(chi2, alpha=0.1)
features_selected = selector.fit_transform(tfidf_matrix, labels)

X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")

# SelectFdr
print("SelectFdr Özellik Seçimi:")
selector = SelectFdr(chi2, alpha=0.1)
features_selected = selector.fit_transform(tfidf_matrix, labels)

X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")

# SelectFwe
print("SelectFwe Özellik Seçimi:")
selector = SelectFwe(chi2, alpha=0.1)
features_selected = selector.fit_transform(tfidf_matrix, labels)

X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")

# RFE
print("RFE Özellik Seçimi:")
selector = RFE(LogisticRegression(), n_features_to_select=10)
features_selected = selector.fit_transform(tfidf_matrix, labels)

X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Değeri: {accuracy}")


# ------------- For -----------------------------

# Özellik seçimi yöntemleri
methods = [
    ('Chi-Kare', chi2),
    ('Information Gain', mutual_info_classif),
    ('SelectKBest', SelectKBest(chi2, k=10)),
    ('SelectPercentile', SelectPercentile(mutual_info_classif, percentile=10)),
    ('SelectFpr', SelectFpr(chi2, alpha=0.1)),
    ('SelectFdr', SelectFdr(chi2, alpha=0.1)),
    ('SelectFwe', SelectFwe(chi2, alpha=0.1)),
    ('RFE', RFE(LogisticRegression(), n_features_to_select=10))
]

for name, method in methods:
    if name == 'RFE':
        features_selected = method.fit_transform(tfidf_matrix, labels)
    elif name == 'SelectKBest' or name == 'SelectPercentile' or name == 'SelectFpr' or name == 'SelectFdr' or name == 'SelectFwe':
        features_selected = method.fit_transform(tfidf_matrix, labels)
    else:
        scores, p_values = method(tfidf_matrix, labels)
        selected_features = np.argsort(scores)[-10:]  # En yüksek puanlı 10 özellik seçildi
        features_selected = tfidf_matrix[:, selected_features]

    # Eğitim ve test verilerini oluştur
    X_train, X_test, y_train, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)

    # KNN sınıflandırıcısını eğit
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Test verilerini tahmin et
    y_pred = knn.predict(X_test)

    # Doğruluk değerini hesapla ve yazdır
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Doğruluk Değeri: {accuracy}")
