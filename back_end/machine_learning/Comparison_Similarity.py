# ---------------------------------------------
import json

import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, Doc2Vec, FastText, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Kullanıcıdan metin girdisi alınması
user_input = "information systems retrieval knowledge discovery and data mining"

# Metinleri çıkarma
texts = [entry["Aims and Scope"] for entry in data]

# TF-IDF temsili
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts + [user_input])

# Bag-of-Words (BoW) temsili
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(texts + [user_input])

# Word2Vec temsili
word2vec_model = Word2Vec([text.split() for text in texts], vector_size=100, window=5, min_count=1)
user_input_word2vec = np.mean([word2vec_model.wv[word] for word in user_input.split()], axis=0)

# FastText temsili
# fasttext_model = FastText(sentences=[text.split() for text in texts], vector_size=100, window=5, min_count=1)
# user_input_fasttext = np.mean([fasttext_model.wv[word] for word in user_input.split()], axis=0)

# GloVe temsili
# glove_file = "path/to/glove.6B.100d.txt"      # indirilmeli
# tmp_file = "path/to/glove.6B.100d.word2vec.txt"
# glove2word2vec(glove_file, tmp_file)
# glove_model = KeyedVectors.load_word2vec_format(tmp_file)
# user_input_glove = np.mean([glove_model[word] for word in user_input.split()], axis=0)

# Doc2Vec temsili
tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
doc2vec_model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
user_input_doc2vec = doc2vec_model.infer_vector(user_input.split())

# Ngram temsili
ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
ngram_matrix = ngram_vectorizer.fit_transform(texts + [user_input])

# Metinler arasındaki benzerlikleri hesaplama
cosine_similarities_tfidf = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
cosine_similarities_bow = cosine_similarity(bow_matrix[-1], bow_matrix[:-1])
cosine_similarities_word2vec = cosine_similarity([user_input_word2vec], word2vec_model.wv.vectors)
# cosine_similarities_fasttext = cosine_similarity([user_input_fasttext], fasttext_model.wv.vectors)
# cosine_similarities_glove = cosine_similarity([user_input_glove], glove_model.vectors)
cosine_similarities_doc2vec = cosine_similarity([user_input_doc2vec], [doc2vec_model.docvecs[str(i)] for i in range(len(texts))])
cosine_similarities_ngram = cosine_similarity(ngram_matrix[-1], ngram_matrix[:-1])

# En iyi tavsiyeyi bulma
index_tfidf_best = np.argmax(cosine_similarities_tfidf)
index_bow_best = np.argmax(cosine_similarities_bow)
index_word2vec_best = np.argmax(cosine_similarities_word2vec)
# index_fasttext_best = np.argmax(cosine_similarities_fasttext)
# index_glove_best = np.argmax(cosine_similarities_glove)
index_doc2vec_best = np.argmax(cosine_similarities_doc2vec)
index_ngram_best = np.argmax(cosine_similarities_ngram)

# Sonuçları yazdırma
print("\nEn İyi Tavsiye (TF-IDF Temsili):")
print(f"Metin: {data[index_tfidf_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_tfidf[0][index_tfidf_best]}\n")

print("\nEn İyi Tavsiye (Bag-of-Words - BoW Temsili):")
print(f"Metin: {data[index_bow_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_bow[0][index_bow_best]}\n")

print("\nEn İyi Tavsiye (Word2Vec Temsili):")
print(f"Metin: {data[index_word2vec_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_word2vec[0][index_word2vec_best]}\n")

# print("\nEn İyi Tavsiye (FastText Temsili):")
# print(f"Metin: {data[index_fasttext_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_fasttext[0][index_fasttext_best]}\n")

# print("\nEn İyi Tavsiye (GloVe Temsili):")
# print(f"Metin: {data[index_glove_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_glove[0][index_glove_best]}\n")

print("\nEn İyi Tavsiye (Doc2Vec Temsili):")
print(f"Metin: {data[index_doc2vec_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_doc2vec[0][index_doc2vec_best]}\n")

print("\nEn İyi Tavsiye (Ngram Temsili):")
print(f"Metin: {data[index_ngram_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_ngram[0][index_ngram_best]}\n")

# ------------------------------------------------

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Kullanıcıdan metin girdisi alınması
user_input = "information systems retrieval knowledge discovery and data mining"

# Metinleri çıkarma
texts = [entry["Aims and Scope"] for entry in data]

# TF-IDF temsili
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts + [user_input])

# Metinler arasındaki benzerlikleri hesaplama
cosine_similarities_tfidf = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
euclidean_distances_tfidf = euclidean_distances(tfidf_matrix[-1], tfidf_matrix[:-1])
manhattan_distances_tfidf = manhattan_distances(tfidf_matrix[-1], tfidf_matrix[:-1])

# L1, L2, Chebyshev ve Minkowski benzerliklerini hesaplama
l1_distances_tfidf = np.linalg.norm(tfidf_matrix[-1].toarray() - tfidf_matrix[:-1].toarray(), ord=1, axis=1)
l2_distances_tfidf = np.linalg.norm(tfidf_matrix[-1].toarray() - tfidf_matrix[:-1].toarray(), ord=2, axis=1)
chebyshev_distances_tfidf = np.max(np.abs(tfidf_matrix[-1].toarray() - tfidf_matrix[:-1].toarray()), axis=1)
minkowski_distances_tfidf = np.linalg.norm(tfidf_matrix[-1].toarray() - tfidf_matrix[:-1].toarray(), ord=3, axis=1)

# En iyi tavsiyeyi bulma
index_tfidf_best = np.argmax(cosine_similarities_tfidf)
index_euclidean_best = np.argmin(euclidean_distances_tfidf)
index_manhattan_best = np.argmin(manhattan_distances_tfidf)
index_l1_best = np.argmin(l1_distances_tfidf)
index_l2_best = np.argmin(l2_distances_tfidf)
index_chebyshev_best = np.argmin(chebyshev_distances_tfidf)
index_minkowski_best = np.argmin(minkowski_distances_tfidf)

# Sonuçları yazdırma
print("\nEn İyi Tavsiye (TF-IDF Temsili - Cosine Benzerliği):")
print(f"Metin: {data[index_tfidf_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_tfidf[0][index_tfidf_best]}\n")

print("\nEn İyi Tavsiye (TF-IDF Temsili - Euclidean Benzerliği):")
print(f"Metin: {data[index_euclidean_best]['Journal Name']}\nBenzerlik Skoru: {1 / (1 + euclidean_distances_tfidf[0][index_euclidean_best])}\n")

print("\nEn İyi Tavsiye (TF-IDF Temsili - Manhattan Benzerliği):")
print(f"Metin: {data[index_manhattan_best]['Journal Name']}\nBenzerlik Skoru: {1 / (1 + manhattan_distances_tfidf[0][index_manhattan_best])}\n")

print("\nEn İyi Tavsiye (TF-IDF Temsili - L1 Benzerliği):")
print(f"Metin: {data[index_l1_best]['Journal Name']}\nBenzerlik Skoru: {1 / (1 + l1_distances_tfidf[index_l1_best])}\n")

print("\nEn İyi Tavsiye (TF-IDF Temsili - L2 Benzerliği):")
print(f"Metin: {data[index_l2_best]['Journal Name']}\nBenzerlik Skoru: {1 / (1 + l2_distances_tfidf[index_l2_best])}\n")

print("\nEn İyi Tavsiye (TF-IDF Temsili - Chebyshev Benzerliği):")
print(f"Metin: {data[index_chebyshev_best]['Journal Name']}\nBenzerlik Skoru: {1 / (1 + chebyshev_distances_tfidf[index_chebyshev_best])}\n")

print("\nEn İyi Tavsiye (TF-IDF Temsili - Minkowski Benzerliği):")
print(f"Metin: {data[index_minkowski_best]['Journal Name']}\nBenzerlik Skoru: {1 / (1 + minkowski_distances_tfidf[index_minkowski_best])}\n")

# En iyi benzerlik ölçüsünü belirleme
best_similarity_tfidf = max(cosine_similarities_tfidf[0][index_tfidf_best],
                            1 / (1 + euclidean_distances_tfidf[0][index_euclidean_best]),
                            1 / (1 + manhattan_distances_tfidf[0][index_manhattan_best]),
                            1 / (1 + l1_distances_tfidf[index_l1_best]),
                            1 / (1 + l2_distances_tfidf[index_l2_best]),
                            1 / (1 + chebyshev_distances_tfidf[index_chebyshev_best]),
                            1 / (1 + minkowski_distances_tfidf[index_minkowski_best]))

best_method = None

if best_similarity_tfidf == cosine_similarities_tfidf[0][index_tfidf_best]:
    best_method = "Cosine"
elif best_similarity_tfidf == 1 / (1 + euclidean_distances_tfidf[0][index_euclidean_best]):
    best_method = "Euclidean"
elif best_similarity_tfidf == 1 / (1 + manhattan_distances_tfidf[0][index_manhattan_best]):
    best_method = "Manhattan"
elif best_similarity_tfidf == 1 / (1 + l1_distances_tfidf[index_l1_best]):
    best_method = "L1"
elif best_similarity_tfidf == 1 / (1 + l2_distances_tfidf[index_l2_best]):
    best_method = "L2"
elif best_similarity_tfidf == 1 / (1 + chebyshev_distances_tfidf[index_chebyshev_best]):
    best_method = "Chebyshev"
elif best_similarity_tfidf == 1 / (1 + minkowski_distances_tfidf[index_minkowski_best]):
    best_method = "Minkowski"

print(f"\nEn iyi benzerlik yöntemi (TF-IDF Temsili): {best_method}")