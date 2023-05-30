from collections import Counter
from warnings import filterwarnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# warningleri göz ardı et
from wordcloud import WordCloud

filterwarnings('ignore')
# bütün sütunları göster
pd.set_option('display.max_columns', None)
# max genişlik 200 olsun
pd.set_option('display.width', 200)
# virgülden sonra iki basamak göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)

sw = stopwords.words('english')

df = pd.read_csv("back_end/dataset/wos-engineering.csv", sep=",")

# --------- Preprocessing ---------

fields, fields_size = df.columns, df.columns.size


def text_preprocessing(fields):
    for field in fields:
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
    return df


df1 = text_preprocessing(fields)

freq_count = []
for item in df1['Aims and Scope']:
    count = Counter(str(item).split())
    freq_count.append(count)
df1['word_count'] = freq_count
df1.head()

tfidfVect = TfidfVectorizer()
tfidf = tfidfVect.fit_transform(df1['Aims and Scope'])
print(tfidf)

print(tfidfVect.vocabulary_.get('printing'))

text_try = df1[df1['Aims and Scope'].str.contains("printing material")]

for words in text_try.word_count.items():
    print(set(words[1].elements()))
    text_try_words = set(words[1].elements())

text_try_tfidfVect = TfidfVectorizer()
text_try_tfidfVect = text_try_tfidfVect.fit(df1['Aims and Scope'])
text_try_tfidf = text_try_tfidfVect.transform(text_try['Aims and Scope'])

text_try_tfidf_table = pd.DataFrame(sorted(text_try_tfidfVect.vocabulary_.items(),
                                           key=lambda pair: pair[1], reverse=True))

feature_names = text_try_tfidfVect.get_feature_names_out()
for col in text_try_tfidf.nonzero()[1]:
    print(feature_names[col], '  :  ', text_try_tfidf[0, col])

feature_array = np.array(feature_names)
tfidf_sorting = np.argsort(text_try_tfidf.toarray()).flatten()[::-1]
top_n = feature_array[tfidf_sorting][:20]

nbrs = NearestNeighbors(n_neighbors=20).fit(tfidf)

distances, indices = nbrs.kneighbors(text_try_tfidf)

names_similar = pd.Series(indices.flatten()).map(df1.reset_index()['Aims and Scope'])

result = pd.DataFrame({'distance': distances.flatten(), 'Aims and Scope': names_similar})

# 2. Text Visualization
# ****************************

# Terim Frekanslarının Hesaplanması
# ----------------------------------

tf = df1['Aims and Scope'].apply(lambda x: pd.value_counts(x.split())).sum(axis=0).reset_index()
# Sütun isimlerini güncelleme
# index, 0 -> words, tf
tf.columns = ["words", "tf"]
# Azalan olacak şekilde sıralama
tf.sort_values("tf", ascending=False)

# Barplot / Sütun grafik
# ------------------------

tf.sort_values("tf", ascending=False)[:10].plot.bar(x="words", y="tf",
                                                    title='top 10 words in Aims and Scope',
                                                    xlabel="Words",
                                                    ylabel="Word Count", color='darkblue')
plt.figure(figsize=(10, 10))
plt.show()

# Word Cloud / Kelime bulutu
# ---------------------------

# Kelimelerin frekanslarına göre resim oluşturma işlemidir
# Bu işlem için veri setindeki bütün satırların tek bir string olarak ifade edilmesi gerekir

text = " ".join(i for i in df1['Aims and Scope'])

# Wordcloud özelleştirme
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
