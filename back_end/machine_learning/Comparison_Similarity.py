import json
import time

import numpy as np
import pandas as pd

from back_end.machine_learning.FeatureEngineering import text_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity


class GenericMethods:
    def __init__(self, data, user_input, texts):
        self.data = data
        self.user_input = user_input
        self.texts = texts
        self.cosine_similarities = None
        self.processing_time = None

    def compute_similarity(self):
        pass

    def get_best_recommendations(self, top_k=5):
        indices = np.argsort(self.cosine_similarities[0])[-top_k:][::-1]
        recommendations = [(self.data[index]['Journal Name'], self.cosine_similarities[0][index]) for index in indices]
        return recommendations


class TFIDFMethods(GenericMethods):
    def compute_similarity(self):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.texts + [self.user_input])
        self.cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])


class Word2VecMethods(GenericMethods):
    def compute_similarity(self):
        word2vec_model = Word2Vec([text.split() for text in self.texts], vector_size=100, window=5, min_count=1)
        user_input_word2vec = np.mean([word2vec_model.wv[word] for word in self.user_input.split()], axis=0)
        self.cosine_similarities = cosine_similarity([user_input_word2vec], word2vec_model.wv.vectors)


class Doc2VecMethods(GenericMethods):
    def compute_similarity(self):
        tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(self.texts)]
        doc2vec_model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
        doc2vec_model.build_vocab(tagged_data)
        doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
        user_input_doc2vec = doc2vec_model.infer_vector(self.user_input.split())
        self.cosine_similarities = cosine_similarity([user_input_doc2vec],
                                                     [doc2vec_model.dv[str(i)] for i in range(len(self.texts))])


class BowMethods(GenericMethods):
    def compute_similarity(self):
        bow_vectorizer = CountVectorizer()
        bow_matrix = bow_vectorizer.fit_transform(self.texts + [self.user_input])
        self.cosine_similarities = cosine_similarity(bow_matrix[-1], bow_matrix[:-1])


class NgramMethods(GenericMethods):
    def compute_similarity(self):
        ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
        ngram_matrix = ngram_vectorizer.fit_transform(self.texts + [self.user_input])
        self.cosine_similarities = cosine_similarity(ngram_matrix[-1], ngram_matrix[:-1])


# JSON dosyasını okuyun
with open('C:\\Users\\anurd\\Downloads\\wos.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

texts = text_preprocessing(df, "Aims and Scope")

user_input = "information system retrieval knowledge discovery data mining"
# user_input = "Novel additive manufacturing processes and techniques Improvements of established methods and materials"

methods = [TFIDFMethods, BowMethods, Word2VecMethods, Doc2VecMethods, NgramMethods]

for method_class in methods:
    started = time.time()
    method = method_class(data, user_input, texts)
    method.compute_similarity()
    recommendations = method.get_best_recommendations(top_k=5)
    distance = time.time() - started

    print("\nEn İyi Tavsiyeler ({})".format(method_class.__name__))
    for i, recommendation in enumerate(recommendations, start=1):
        print(f"{recommendation[0]},{recommendation[1]:.3f}")
    print(f"Süre: {distance:.4f} s\n")
