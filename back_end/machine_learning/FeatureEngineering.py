import pandas as pd
from nltk.corpus import stopwords
from textblob import Word


def text_preprocessing(df, field):
    sw = stopwords.words('english')
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
