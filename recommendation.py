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

# ****************************
# 2. Text Visualization
# ****************************

# Terim Frekanslarının Hesaplanması
# ----------------------------------

tf = df['Aims and Scope'].apply(lambda x: pd.value_counts(x.split())).sum(axis=0).reset_index()
# Sütun isimlerini güncelleme
# index, 0 -> words, tf
tf.columns = ["words", "tf"]
# Azalan olacak şekilde sıralama
tf.sort_values("tf", ascending=False)

# Barplot / Sütun grafik
# ------------------------

# Sütun grafikte bütün değerler göstermek yerine sınır belirlemek daha mantıklı olacaktır

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

# Word Cloud / Kelime bulutu
# ---------------------------

# Kelimelerin frekanslarına göre resim oluşturma işlemidir
# Bu işlem için veri setindeki bütün satırların tek bir string olarak ifade edilmesi gerekir
text = " ".join(i for i in df['Aims and Scope'])

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud_black.png")

# Örnek özelleştirme
wordcloud = WordCloud(
    max_font_size=50,
    max_words=100,
    background_color="white"
).generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud_white.png")

# Şablonlara göre Word Cloud
# ---------------------------

# bir resmin üzerine wordcloud yapılması amaçlanmaktadır
tr_mask = np.array(Image.open("L5-Dogal-Dil-Islemeye-Giris/tr.png"))

wc = WordCloud(
    background_color="white",
    max_words=1000,
    mask=tr_mask,
    contour_width=3,
    contour_color="firebrick"
)

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wc_sablon.png")

# Count Vectors
# ----------------

# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)

# Neye göre count vektör işlemi yapılabilir:

# 1. words
# kelimelerin nümerik temsilleridir

# 2. characters
# karakterlerin numerik temsilleridir

# 3. ngram
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir ve feature üretmek için kullanılır"""

TextBlob(a).ngrams(3)

# Count Vectors

from sklearn.feature_extraction.text import CountVectorizer

# corpus dört farklı birim olarak düşünülebilir
corpus = ['big data',
          'search']

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(df)

vectorizer.get_feature_names_out()
# unique olacak şekilde kelimeleri listeler ve bunlar sütun isimleri olur
# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

X_c.toarray()
# vektörel hali - words bazında
# [[0, 1, 1, 1, 0, 0, 1, 0, 1],
#  [0, 2, 0, 1, 0, 1, 1, 0, 1],
#  [1, 0, 0, 1, 1, 0, 1, 1, 1],
#  [0, 1, 1, 1, 0, 0, 1, 0, 1]]


# n-gram frekans

# ngram_range ile yapılar oluşturulur.
# ngram_range=(2, 2) -> 2 şer kelimelik yapılar oluşturma işlemi yapılır
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(df)
vectorizer2.get_feature_names_out()
# ['and this', 'document is', 'first document', 'is the', 'is this',
#        'second document', 'the first', 'the second', 'the third',
#        'third one', 'this document', 'this is', 'this the']

X_n.toarray()

vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X_n)

vectorizer.get_feature_names_out()[10:15]
X_count.toarray()[10:15]


# Logistic Regression
# ---------------------

y = df["Aims and Scope"]  # bağımlı değişken
X = df["Journal Name"]  # bağımsız değişken

log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()

new_search = pd.Series("big data")

# yeni gelen search için de yukarıdaki işlemler (vektör counts) uygulanır
new_review = TfidfVectorizer().fit(X).transform(new_search)

# Orijinal veri setinde herhangi bir yorumu sorma işlemi yapılmak istenirse
random_review = pd.Series(df["Aims and Scope"].sample(1).values)
new_review = TfidfVectorizer().fit(X).transform(random_review)
log_model.predict(new_review)



# Random Forests
# --------------
# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()
# 0.8413021363173957



# Hiperparametre Optimizasyonu
# -----------------------------
rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

# Olası kombinasyonlardan en iyisini bulmak için GridSearch'e bakılır
rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_count, y)

rf_best_grid.best_param

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)
# hatayı tekrar değerlendirme
cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean()





# ****************************
# 3. Sentiment Analysis
# ****************************

# Verilerin taşıdığı duygu durumunu matematiksel olarak ifade etmeyi amaçlamaktadır.
df["Aims and Scope"].head()

# Python'da duygu analizi için free trains modeller mevcut
# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
# sia değişkenini kullanarak polarite skoru getirilir
sia.polarity_scores("The journal big data")

# compound -> duygu skorları (-1 ile 1 arası)
# 0'dan küçük olanlar negatif düşünce, büyük olanlar pozitif düşünce olarak düşünülebilir

# Example 2 - Comment
sia.polarity_scores("I liked this music but it is not good as the other one")

# Her bir review için uygulanırsa
df["Aims and Scope"][0:10].apply(lambda x: sia.polarity_scores(x))
# Sadece compound değişkeni için
df["Aims and Scope"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])
# veri setinde yeni bir değişken oluşturup yukarıdaki compound skorları kalıcı hale getirme
df["polarity_score"] = df["Aims and Scope"].apply(lambda x: sia.polarity_scores(x)["compound"])

# ************************
# 4. Feature Engineering
# ************************

# Amaç: Gelen bir yorum için pozitif ya da negatif olma durumu
# İki durumda ilerlenebilir:
# 1. Unsupervised -> Supervised geçiş ile bir label (target) yapma işlemi (sınıflama problemi)
# 2. Makine öğrenmesi modelleri ile metin sınıflandırma


# Sınıflandırma problemi ile model oluşturma, pos or neg
df["Aims and Scope"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
# Target veri setine ekleniyor
df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
# pos    3944
# neg     971
df["sentiment_label"].value_counts()

# Yeni targetın overall'a bağlı olarak ortalamasına bakılır
df.groupby("sentiment_label")["overall"].mean()
# Label encoderdan geçirilir
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]  # bağımlı değişken
X = df["reviewText"]  # bağımsız değişken
# Bağımsız değişkenler numerik olmadığından makine öğrenmesi modelinden geçirmeden önce vektörel işlemlerden geçirilmesi gerekir