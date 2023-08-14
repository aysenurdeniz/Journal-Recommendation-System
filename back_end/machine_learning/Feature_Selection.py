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

# ------------------------------------------------

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Metinleri ve etiketleri çıkarma
texts = [entry["Aims and Scope"] for entry in data]
labels = [entry["Journal Name"] for entry in data]

# Veri setini eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF temsili
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_matrix_test = tfidf_vectorizer.transform(X_test)

# Word2Vec temsili
word2vec_model = Word2Vec([text.split() for text in texts], vector_size=100, window=5, min_count=1)
word2vec_matrix_train = np.array([np.mean([word2vec_model.wv[word] for word in text.split()], axis=0) for text in X_train])
word2vec_matrix_test = np.array([np.mean([word2vec_model.wv[word] for word in text.split()], axis=0) for text in X_test])

# Cosine similarity ile benzerlikleri hesaplama
cosine_similarities_tfidf = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
cosine_similarities_word2vec = cosine_similarity(word2vec_matrix_test, word2vec_matrix_train)

top_n = 5

# Tavsiye edilen makalelerin indekslerini saklamak için bir dizi oluşturma
tfidf_recommendations_indices = [np.argsort(similarity)[-top_n:][::-1] for similarity in cosine_similarities_tfidf]
word2vec_recommendations_indices = [np.argsort(similarity)[-top_n:][::-1] for similarity in
                                    cosine_similarities_word2vec]

# Tavsiye edilen makaleleri etiketlerle eşleştiren bir dizi oluşturma
tfidf_recommendations = [[y_train[idx] for idx in indices] for indices in tfidf_recommendations_indices]
word2vec_recommendations = [[y_train[idx] for idx in indices] for indices in word2vec_recommendations_indices]

# Hedef değerler ile tavsiye çıktılarının uyumunu kontrol etmek için metrikleri hesaplama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_recommendations(y_true, recommendations):
    y_pred = [item for sublist in recommendations for item in sublist]
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (Micro): {precision_micro:.2f}")
    print(f"Recall (Micro): {recall_micro:.2f}")
    print(f"F1 Score (Micro): {f1_micro:.2f}")


# Tavsiye sistemi performansını değerlendirme
print("\nTF-IDF Tavsiye Performansı:")
evaluate_recommendations(y_test, tfidf_recommendations)

print("\nWord2Vec Tavsiye Performansı:")
evaluate_recommendations(y_test, word2vec_recommendations)

# -----------------------------

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Metinleri çıkarma
texts = [entry["Aims and Scope"] for entry in data]

# TF-IDF vektörleri oluşturma
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Kullanıcıdan metin girdisi alınması
user_input = input("Enter your keywords: ")

# Kullanıcının girdisi üzerinden bir TF-IDF vektörü oluşturma
user_input_tfidf = tfidf_vectorizer.transform([user_input])

# Tavsiye sistemi oluşturma
top_n = 5
cosine_similarities_tfidf = cosine_similarity(user_input_tfidf, tfidf_matrix)
tfidf_recommendations_indices = [np.argsort(similarity)[-top_n:][::-1] for similarity in cosine_similarities_tfidf]
tfidf_recommendations = [[texts[idx] for idx in indices] for indices in tfidf_recommendations_indices]

# Tavsiye sistemi performansını değerlendirme
y_test = [entry["Aims and Scope"] for entry in data]
print("\nTF-IDF Tavsiye Performansı:")
evaluate_recommendations(y_test, tfidf_recommendations)


# Tavsiye sistemi performansını değerlendiren fonksiyon
def evaluate_recommendations(y_true, recommendations):
    y_pred = [item for sublist in recommendations for item in sublist]
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (Micro): {precision_micro:.2f}")
    print(f"Recall (Micro): {recall_micro:.2f}")
    print(f"F1 Score (Micro): {f1_micro:.2f}")

#  -------------------------------------------------

import requests
import time

def get_similar_documents_solr(solr_url, core_name, keywords):
    keyword_query = ' OR '.join(keywords)
    url = f"{solr_url}/{core_name}/mlt?q=content:{keyword_query}&mlt.mindf=1&mlt.mintf=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def measure_performance(function, *args):
    start_time = time.time()
    result = function(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Test senaryosu
solr_url = "http://localhost:8983/solr"  # Solr URL
core_name = "your_core_name"            # Solr çekirdek adı

# Kullanıcı tarafından alınan keyword'ler
user_keywords = ["python", "machine learning", "data science"]

# Solr performans ölçümü ve benzerlik sorgusu
solr_result, solr_execution_time = measure_performance(get_similar_documents_solr, solr_url, core_name, user_keywords)
print("Solr Result:", solr_result)
print("Solr Execution Time:", solr_execution_time, "seconds")


import requests
import time

def get_similar_documents_es(es_url, index_name, user_keywords):
    keyword_query = ' OR '.join(user_keywords)
    url = f"{es_url}/{index_name}/_search"
    headers = {'Content-Type': 'application/json'}
    data = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"content": keyword_query}}  # Belge içeriğindeki "content" alanına göre keyword eşleştirmesi
                ]
            }
        }
    }
    response = requests.get(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def measure_performance(function, *args):
    start_time = time.time()
    result = function(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Test senaryosu
es_url = "http://localhost:9200"  # Elasticsearch URL
index_name = "your_index_name"    # Elasticsearch index adı

# Kullanıcı tarafından alınan keyword'ler
user_keywords = ["python", "machine learning", "data science"]

# Elasticsearch performans ölçümü ve benzerlik sorgusu
es_result, es_execution_time = measure_performance(get_similar_documents_es, es_url, index_name, user_keywords)
print("Elasticsearch Result:", es_result)
print("Elasticsearch Execution Time:", es_execution_time, "seconds")


# -----------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Örnek metin veri kümesi
texts = ["Bu bir örnek cümle.",
         "Python programlama dilini öğreniyorum.",
         "Makine öğrenmesi ilgi çekici bir konudur.",
         "BoW ve Tf-idf yöntemlerini karşılaştırıyorum."]

labels = [0, 1, 1, 0]

# Veri kümesini eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tf-idf dönüşümü
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# BoW dönüşümü
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Naive Bayes sınıflandırıcı
nb_classifier = MultinomialNB()

# Tf-idf ile modeli eğitme ve değerlendirme
nb_classifier.fit(X_train_tfidf, y_train)
y_pred_tfidf = nb_classifier.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

# BoW ile modeli eğitme ve değerlendirme
nb_classifier.fit(X_train_bow, y_train)
y_pred_bow = nb_classifier.predict(X_test_bow)
accuracy_bow = accuracy_score(y_test, y_pred_bow)

# Sonuçları karşılaştırma
print("Tf-idf Doğruluk:", accuracy_tfidf)
print("BoW Doğruluk:", accuracy_bow)


# ----------------------------

import requests

def more_like_this_solr(solr_url, core_name, user_keywords):
    keyword_query = ' AND '.join(user_keywords)
    url = f"{solr_url}/{core_name}/mlt?q={keyword_query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Test senaryosu
solr_url = "http://localhost:8983/solr"  # Solr URL
core_name = "wos2"            # Solr çekirdek adı

# Kullanıcı tarafından alınan keyword'ler
user_keywords = ["python", "machine learning", "data science"]

# Solr ile MLT sorgusu
result = more_like_this_solr(solr_url, core_name, user_keywords)
print(result)




