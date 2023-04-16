import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


# 1. Exploratory Data Analysis
# -----------------------------

df = pd.read_csv("back_end/dataset/wos-engineering.csv", sep=",")

df.head()
df.shape
df.describe().T
df["Journal_Name"].value_counts()


# 2. Data Preprocessing & Feature Engineering
# ----------------------------------------------

y = df["Journal_Name"]
X = df.drop(["Journal_Name"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)


# 3. Modeling & Prediction
# --------------------------

knn_model = KNeighborsClassifier().fit(X, y)

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

