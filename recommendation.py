# !pip install nltk
# !pip install textblob
# !pip install wordcloud

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

# warningleri göz ardı et
filterwarnings('ignore')
# bütün sütunları göster
pd.set_option('display.max_columns', None)
# max genişlik 200 olsun
pd.set_option('display.width', 200)
# virgülden sonra iki basamak göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import nltk
nltk.download('wordnet')
# nltk kütüphanesinden stopwords listesini indir
# nltk.download('stopwords')
sw = stopwords.words('english')

# ****************************
# 1. Text Preprocessing
# ****************************

df = pd.read_csv("back_end/dataset/wos-engineering.csv", sep=",")


def text_preprocessing(field):
    # Normalizing Case Folding
    df[field] = df[field].str.lower()
    # -----------------------------------
    # Punctuations
    # [^\u\s] -> a regular expression
    df[field] = df[field].str.replace('[^\w\s]', '')
    # -----------------------------------
    # Numbers Remove
    # \d -> a regular expression for numbers
    df[field] = df[field].str.replace('\d', '')
    # -----------------------------------
    # Stopwords
    # stopword listesinde bulunanları datasette bulup bu kelimeler silinmeli
    # Öncelikle her satırda gezilmeli (apply) ve her satırda bütün kelimeleri de gezmeli (lambda)
    df[field] = df[field].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # -----------------------------------
    # Rarewords
    # kelimelerin frekanslarına bakarak belli bir sayı sınırından önceki kelimeler kaldırılabiilir
    temp_df = pd.Series(' '.join(df[field]).split()).value_counts()
    # örn. birden az olan frekansa sahip olanlar alınsın
    drops = temp_df[temp_df <= 1]
    df[field] = df[field].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    # -----------------------------------
    # Lemmatization
    df[field] = df[field].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return df[field]


text_preprocessing("Aims and Scope")


# TF-IDF Yöntemi
# ---------------

# Count vektörün açığa çıkarabileceği yanlılıkları giderebilmek adına
# standartize edilmiş bir kelime vektörü oluşturma yöntemidir.

# for words
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(df["Aims and Scope"])
X_tf_idf_word.shape

similarity_matrix = linear_kernel(X_tf_idf_word, X_tf_idf_word)
similarity_matrix

# df index mapping
mapping = pd.Series(df.index, index=df["Journal Name"])
mapping


def recommend_word_based_on_plot(word_input):
    movie_index = mapping[word_input]
    # get similarity values with other movies
    # similarity_score is the list of index and similarity matrix
    similarity_score = list(enumerate(similarity_matrix[movie_index]))
    # sort in descending order the similarity score of movie inputted with all the other movies
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    # Get the scores of the 15 most similar movies. Ignore the first movie.
    similarity_score = similarity_score[1:15]
    # return movie names using the mapping series
    df_indices = [i[0] for i in similarity_score]
    return df["Aims and Scope"].iloc[df_indices]


recommend_word_based_on_plot("ACI STRUCTURAL JOURNAL")


# df.iloc[df_indices]["Aims and Scope"]