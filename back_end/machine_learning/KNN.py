from warnings import filterwarnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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


# -----------------------------

from sklearn.feature_extraction.text import TfidfVectorizer

ted = df["cleaned_aims_and_scope"]

# Create TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Generate matrix of word vectors
tfidf_matrix = vectorizer.fit_transform(ted)

# Print the shape of tfidf_matrix
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)

import time

from sklearn.metrics.pairwise import linear_kernel

# Record start time
start = time.time()

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print cosine similarity matrix
print(cosine_sim)

# Print time taken
print("Time taken: %s seconds" % (time.time() - start))

indices = pd.Series(df.index, index=df["cleaned_aims_and_scope"]).drop_duplicates()


def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    talk_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return df["Journal_Name"].iloc[talk_indices]


indices = pd.Series(ted.index, index=df["Journal Name"]).drop_duplicates()
transcripts = df["cleaned_aims_and_scope"]
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(transcripts)

# Generate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

idx = indices["ACI STRUCTURAL JOURNAL"]
# Get the pairwsie similarity scores
sim_scores = list(enumerate(cosine_sim[idx]))
# Sort the movies based on the similarity scores
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
# Get the scores for 10 most similar movies
sim_scores = sim_scores[1:11]
# Get the movie indices
talk_indices = [i[0] for i in sim_scores]
# Return the top 10 most similar movies
new = df[["Journal Name", "Aims and Scope"]].iloc[talk_indices]

# Generate recommendations
print(get_recommendations("printing additive manufacturing", cosine_sim, indices))


# Activate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words="english", min_df=0.005)
wm = count_vectorizer.fit_transform(df["cleaned_aims_and_scope"])
print(count_vectorizer.get_feature_names_out())

df1 = pd.DataFrame(wm.toarray(), columns=count_vectorizer.get_feature_names_out())

# --------------------------------------------

from sklearn.metrics.pairwise import cosine_similarity
df2 = pd.DataFrame(cosine_similarity(df1, dense_output=True))
df2.head()

search_df = pd.DataFrame([df1["printing"], df1["ability"], df1["able"]], index=["printing", "ability", "able"]).T
search_df = search_df[search_df[["printing", "ability", "able"]] > 0]
search_df
test = search_df.sort_values(["ability"], ascending=[False])
test

# --------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df["cleaned_aims_and_scope"].values, test_size=0.2, random_state=123, stratify=df["cleaned_aims_and_scope"].values)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_train_vectors = vectorizer.fit_transform(df["cleaned_aims_and_scope"]).toarray()
tfidf_test_vectors = vectorizer.transform(["printing adaptive aaai"]).toarray()

classifier = RandomForestClassifier()
classifier.fit(tfidf_train_vectors, tfidf_train_vectors)
y_pred = classifier.predict(tfidf_test_vectors)
print(classification_report(y_test, y_pred))


# ---------------

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

aas_encoded = le.fit_transform(df["Indexing and Abstracting"])
print('Aims and Scope Encoded', aas_encoded)

is_encoded = le.fit_transform(df["Indexing and Abstracting"])
print('Aims and Scope Encoded', is_encoded)

# ------------------

features = list(zip(aas_encoded, is_encoded))
print('Features', features)

# ------------------

from sklearn.neighbors import KNeighborsClassifier

# Applying k = 3, default Minkowski distance metrics
model = KNeighborsClassifier(n_neighbors=3)
# Training the classifier
model.fit(X, X)

# ------------------

# Testing the classifier
y_pred = model.predict(X_test)
print('Predicted', y_pred)
print('Actual data', y_test)

# ------------------

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy', accuracy)

# 2. Data Preprocessing & Feature Engineering
# -------------------------------------

# TF-IDF Yöntemi
# ---------------
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

freq_count = []
for item in df['cleaned_aims_and_scope']:
    count = Counter(str(item).split())
    freq_count.append(count)
df['word_count'] = freq_count
df.head()

tfidf_vect = TfidfVectorizer()
X = tfidf_vect.fit_transform(df['cleaned_aims_and_scope'])
tfidf_vectt = tfidf_vect.transform(df['cleaned_aims_and_scope'])

one_hot_encoded_data = pd.get_dummies(tfidf_vectt, columns=['cleaned_aims_and_scope'])

# ---------

y = df["cleaned_aims_and_scope"]
X = df.drop(["cleaned_aims_and_scope"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

# 3. Modeling & Prediction
# --------------------------

knn_model = KNeighborsClassifier().fit(one_hot_encoded_data, one_hot_encoded_data)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

# 4. Model Evaluation
# -------------------

y_pred = knn_model.predict(X)
y_prob = knn_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

knn_model.get_params()

# 5. Hyperparameter Optimization
# --------------------------------

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_

# 6. Final Model
# ----------------

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)
