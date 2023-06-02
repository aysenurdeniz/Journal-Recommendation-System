# 2. Text Visualization
# ****************************
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud


# Terim Frekanslarının Hesaplanması
# ----------------------------------

def tf_cal(df, field):
    tf = df[field].apply(lambda x: pd.value_counts(x.split())).sum(axis=0).reset_index()
    # Sütun isimlerini güncelleme
    # index, 0 -> words, tf
    tf.columns = ["words", "tf"]
    # Azalan olacak şekilde sıralama
    return tf


def bar_plot(df, field, n):
    tf = tf_cal(df, field)
    tf.sort_values("tf", ascending=False)[:n].plot.bar(x="words", y="tf",
                                                       title='top 10 words in Aims and Scope',
                                                       xlabel="Words",
                                                       ylabel="Word Count", color='darkblue')
    plt.figure(figsize=(10, 10))
    plt.show()


def word_cloud(df, field, save=False):
    text = " ".join(i for i in df[field])

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
    if save == "True":
        wordcloud.to_file("wordcloud_white.png")
