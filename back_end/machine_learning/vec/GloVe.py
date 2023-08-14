import json
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, Doc2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from glove import Corpus, Glove
from sklearn.metrics.pairwise import cosine_similarity

# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

# Kullanıcıdan metin girdisi alınması
user_input = "information systems retrieval knowledge discovery and data mining"

# Metinleri çıkarma
texts = [entry["Aims and Scope"] for entry in data]

# GloVe modelini oluşturma
corpus = Corpus()
corpus.fit(texts, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Kullanıcı girdisini GloVe vektörüne çevirme
user_input_glove = np.mean([glove.word_vectors[glove.dictionary[word]] for word in user_input.split()], axis=0)

# Metinler arasındaki benzerlikleri hesaplama
cosine_similarities_glove = cosine_similarity([user_input_glove], glove.word_vectors)
index_glove_best = np.argmax(cosine_similarities_glove)

# En iyi tavsiyeyi yazdırma
print("\nEn İyi Tavsiye (GloVe Temsili):")
print(f"Metin: {data[index_glove_best]['Journal Name']}\nBenzerlik Skoru: {cosine_similarities_glove[0][index_glove_best]}\n")
