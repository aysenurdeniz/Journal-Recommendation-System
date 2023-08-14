import json
from warnings import filterwarnings

import pandas as pd

from back_end.machine_learning.Comparison_Similarity import TFIDFMethods, BowMethods, Word2VecMethods, Doc2VecMethods, \
    NgramMethods
from back_end.machine_learning.FeatureEngineering import text_preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import time

filterwarnings('ignore')
# bütün sütunları göster
pd.set_option('display.max_columns', None)
# max genişlik 200 olsun
pd.set_option('display.width', 200)
# virgülden sonra iki basamak göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 1. Exploratory Data Analysis
# -----------------------------

df = pd.read_csv("back_end/dataset/wos-engineering.csv", sep=",")

with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

fields, fields_size = df.columns, df.columns.size

# --------- Preprocessing ---------

df["cleaned_aims_and_scope"] = text_preprocessing(df, "Aims and Scope")

# Örnek kullanıcı girdisi
user_input = "information systems retrieval knowledge discovery data mining"

texts = df["cleaned_aims_and_scope"]

# Verileri ayıklayın
# texts = [entry["Aims and Scope"] for entry in data]

# ---------- All Methods --------------

methods = [TFIDFMethods, BowMethods, Word2VecMethods, Doc2VecMethods, NgramMethods]

for method_class in methods:
    method = method_class(texts, user_input)
    method.compute_similarity()
    recommendations = method.get_best_recommendations(top_k=5)

    print("\nEn İyi Tavsiyeler ({})".format(method_class.__name__))
    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{recommendation[0]},{recommendation[1]:.3f}")
    print(f"Süre: {method.processing_time:.4f} s\n")


# --------- TF-IDF w. N-grams ---------

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


# --------- BoW w. N-grams ---------

# İşlem süresini başlat
start_time = time.time()

# CountVectorizer ile Bag-of-Words matrisini oluşturun (N-gram desteği ekleyin)
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))

bow_matrix = vectorizer.fit_transform(texts + [user_input])

# Kullanıcı girdisinin vektör temsili
user_input_vector = bow_matrix[-1]

# Kozinüs benzerliği hesaplayın
cosine_similarities = cosine_similarity(user_input_vector, bow_matrix[:-1])

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
