# TF-IDF Yöntemi
# ---------------
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(df, field):
    freq_count = []
    for item in df[field]:
        count = Counter(str(item).split())
        freq_count.append(count)
    df['word_count'] = freq_count
    df.head()

    tfidf_vect = TfidfVectorizer()
    X = tfidf_vect.fit_transform(df[field])
    return X
