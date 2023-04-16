# TF-IDF Yöntemi
# ---------------
from sklearn.metrics.pairwise import linear_kernel
from pandas import pd
from warnings import filterwarnings
# for words
from sklearn.feature_extraction.text import TfidfVectorizer

# warningleri göz ardı et
filterwarnings('ignore')
# bütün sütunları göster
pd.set_option('display.max_columns', None)
# max genişlik 200 olsun
pd.set_option('display.width', 200)
# virgülden sonra iki basamak göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Count vektörün açığa çıkarabileceği yanlılıkları giderebilmek adına
# standartize edilmiş bir kelime vektörü oluşturma yöntemidir.


df = pd.read_csv("back_end/dataset/wos-engineering.csv", sep=",")

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