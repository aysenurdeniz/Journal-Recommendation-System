import pandas as pd
from back_end.machine_learning.FeatureEngineering import text_preprocessing
from back_end.machine_learning.TF_IDF import tf_idf
from back_end.machine_learning.Text_Visualization import bar_plot, word_cloud

df = pd.read_csv("back_end/dataset/wos-engineering.csv", sep=",")
fields, fields_size = df.columns, df.columns.size

# --------- Preprocessing ---------

df["cleaned_aims_and_scope"] = text_preprocessing(df, "Aims and Scope")

# --------- TF-IDF ---------

X = tf_idf(df, "cleaned_aims_and_scope")

# ---------- Text Visualization --------------

bar_plot(df, "cleaned_aims_and_scope", 10)

word_cloud(df, "cleaned_aims_and_scope")


