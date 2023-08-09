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


# --------------------- heatmap for similarity scores -------------------------


import matplotlib.pyplot as plt

# Varsayılan font boyutunu ayarlayın
plt.rcParams.update({'font.size': 17})  # Yeni font boyutunu burada belirleyin

# Verileri düzenleyin (örnek veri)
journals = [
    "ADDITIVE MANUFACTURING",
    "PROCEEDINGS OF THE INSTITUTION OF MECHANICAL ENGINEERS PART B-JOURNAL OF ENGINEERING MANUFACTURE",
    "INTERNATIONAL JOURNAL OF ADVANCED MANUFACTURING TECHNOLOGY",
    "3D PRINTING AND ADDITIVE MANUFACTURING",
    "PROCESSES",
    "JOURNAL OF MANUFACTURING PROCESSES",
    "JOURNAL OF MANUFACTURING SCIENCE AND ENGINEERING-TRANSACTIONS OF THE ASME",
    "CANADIAN GEOTECHNICAL JOURNAL",
    "CHEMISTRY OF MATERIALS",
    "SOLAR RRL",
    "GEOTHERMAL ENERGY",
    "CONTROL ENGINEERING PRACTICE",
    "ACI MATERIALS JOURNAL",
    "SURFACE COATINGS INTERNATIONAL",
    "MATERIALS PERFORMANCE",
    "ACI STRUCTURAL JOURNAL",
    "J-FOR-JOURNAL OF SCIENCE & TECHNOLOGY FOR FOREST PRODUCTS AND PROCESSES"
]
techniques = ["TF-IDF", "BoW", "Word2Vec", "Doc2Vec", "N-grams"]
similarity_scores = [
    [0.707, 0.597, 0, 0, 0.688],
    [0.411, 0, 0, 0, 0.631],
    [0.378, 0, 0, 0, 0.601],
    [0.373, 0.495, 0, 0, 0.605],
    [0.364, 0.549, 0, 0, 0],
    [0, 0.502, 0, 0, 0.64],
    [0, 0.500, 0, 0, 0],
    [0, 0, 0.997, 0, 0],
    [0, 0, 0.997, 0, 0],
    [0, 0, 0.993, 0, 0],
    [0, 0, 0.993, 0, 0],
    [0, 0, 0.993, 0, 0],
    [0, 0, 0, 0.804, 0],
    [0, 0, 0, 0.796, 0],
    [0, 0, 0, 0.792, 0],
    [0, 0, 0, 0.788, 0],
    [0, 0, 0, 0.786, 0]
]

# Metin kırpma işlemi için maksimum karakter sayısı
max_char_length = 30

# Verileri ısı haritası olarak görselleştirin
plt.figure(figsize=(15, 8))  # Boyutları artırarak metinlerin daha iyi görünmesini sağlayın
plt.imshow(similarity_scores, cmap="Blues", aspect="auto", vmin=0, vmax=1)  # Sadece pozitif değerleri göster

# Eksen etiketlerini ayarlayın (kırpılmış metinler)
plt.xticks(range(len(techniques)), [tech[:max_char_length] for tech in techniques], rotation=45)
plt.yticks(range(len(journals)), [journal[:max_char_length] for journal in journals])

# Renk çubuğunu ekleyin
plt.colorbar(label="Similarity Score")

# Başlığı ve eksen etiketlerini ekleyin
plt.title("Similarity Scores Heatmap for Journals and Techniques")
plt.xlabel("Techniques")
plt.ylabel("Journals")

# Görseli gösterin
plt.tight_layout()
# Resmi kaydetmek için savefig kullanın
plt.savefig("heatmap.png")  # Resim dosyasını istediğiniz isimle ve uzantıyla belirtin
plt.show()

