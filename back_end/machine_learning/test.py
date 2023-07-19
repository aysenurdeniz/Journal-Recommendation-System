import json
from warnings import filterwarnings

import pandas as pd
from back_end.machine_learning.FeatureEngineering import text_preprocessing
from back_end.machine_learning.TF_IDF import tf_idf

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

fields, fields_size = df.columns, df.columns.size

# --------- Preprocessing ---------

df["cleaned_aims_and_scope"] = text_preprocessing(df, "Aims and Scope")

# --------- TF-IDF ---------

X = tf_idf(df, "cleaned_aims_and_scope")
