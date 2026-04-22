
# --- Mini-Project -- Spam or Ham? A Classifier Shootout ---

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- Task 1: Load and Explore ---

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

# Load spambase dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

# Read data into DataFrame
df = pd.read_csv(BytesIO(response.content), header=None)

# Assign column names for better readability
df.columns = COLUMN_NAMES

# Quick check of data structure
print(df.head())
print(df.shape)

# Check class balance (spam vs ham)
print(df["spam_label"].value_counts())

# If classes are imbalanced, accuracy can be misleading,
# because the model may predict only the most common class and still get a high score.

# Boxplot: word frequency of "free" vs spam/ham
sns.boxplot(x='spam_label', y='word_freq_free', data=df)
plt.title("word_freq_free")
plt.savefig('outputs/word_freq_free_distribution.png')
plt.show()

# Boxplot: punctuation "!"
sns.boxplot(x='spam_label', y='char_freq_!', data=df)
plt.title("char_freq_!")
plt.savefig('outputs/char_freq_!_distribution.png')
plt.show()

# Boxplot: capital letter usage
sns.boxplot(x='spam_label', y='capital_run_length_total', data=df)
plt.title("capital_run_length_total")
plt.savefig('outputs/capital_run_length_total_distribution.png')
plt.show()

"""
The features show noticeable differences between spam and ham, but there is still overlap between the distributions.
This means no single feature can perfectly separate the two classes.

Most features are 0 because most words do not appear in most emails.
Different features have different scales (some are small fractions, some are large numbers).
This matters because some models are sensitive to scale, so large values can dominate the result if data is not scaled.

"""

# --- Task 2: Prepare Your Data ---

# Separate features and target 
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling is required because features are on very different scales 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA is fit only on training data to avoid data leakage 
pca = PCA()
pca.fit(X_train_scaled)

# Cumulative explained variance  
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot explained variance
plt.plot(cumulative_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Explained Variance")

plt.savefig('outputs/pca_variance_explained_project.png')
plt.show()

# Find number of components that explain 90% variance
n = np.argmax(cumulative_variance >= 0.90) + 1
print("Components for 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]


# --- Task 3: A Classifier Comparison ---

# KNN on raw (unscaled data)
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
pred = knn_unscaled.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Classification report for unscaled data: \n", classification_report(y_test, pred))

# KNN on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
pred_scaled = knn_scaled.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, pred_scaled))
print("Classification report for scaled data: \n", classification_report(y_test, pred_scaled))

# KNN on PCA-reduced data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
pred_pca = knn_pca.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, pred_pca))
print("Classification report for PCA : \n", classification_report(y_test, pred_pca))

# Decision Tree 

# max_depth = 3
d_tree_3 = DecisionTreeClassifier(max_depth=3, random_state=42)
d_tree_3.fit(X_train, y_train)
train_preds_3 = d_tree_3.predict(X_train)
test_preds_3 = d_tree_3.predict(X_test)

print("Accuracy for train set max_depth=3:", accuracy_score(y_train, train_preds_3))
print("Accuracy for test set max_depth=3:", accuracy_score(y_test, test_preds_3))

# max_depth = 5
d_tree_5 = DecisionTreeClassifier(max_depth=5, random_state=42)
d_tree_5.fit(X_train, y_train)
train_preds_5 = d_tree_5.predict(X_train)
test_preds_5 = d_tree_5.predict(X_test)

print("Accuracy for train set max_depth=5:", accuracy_score(y_train, train_preds_5))
print("Accuracy for test set max_depth=5:", accuracy_score(y_test, test_preds_5))

# max_depth = 10
d_tree_10 = DecisionTreeClassifier(max_depth=10, random_state=42)
d_tree_10.fit(X_train, y_train)
train_preds_10 = d_tree_10.predict(X_train)
test_preds_10 = d_tree_10.predict(X_test)

print("Accuracy for train set max_depth=10:", accuracy_score(y_train, train_preds_10))
print("Accuracy for test set max_depth=10:", accuracy_score(y_test, test_preds_10))

# max_depth = None
d_tree_None = DecisionTreeClassifier(max_depth=None, random_state=42)
d_tree_None.fit(X_train, y_train)
train_preds_None = d_tree_None.predict(X_train)
test_preds_None = d_tree_None.predict(X_test)

print("Accuracy for train set max_depth=None:", accuracy_score(y_train, train_preds_None))
print("Accuracy for test set max_depth=None:", accuracy_score(y_test, test_preds_None))

# max_depth=None achieves the highest training accuracy but shows signs of overfitting, as test accuracy does not improve. 
# max_depth=10 provides the best balance between
# training and test performance, so it is the best choice for production.

# Best balance chosen based on train/test gap and test accuracy
print("Accuracy:", accuracy_score(y_test, test_preds_10))
print(classification_report(y_test, test_preds_10))

# Random Forest 

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# Feature importance comparison

# Importance from Decision Tree and Random Forest
d_tree_10_importances = d_tree_10.feature_importances_
rf_importances = rf.feature_importances_

feature_names = X.columns

# Create tables to match features with important values
d_tree_10_df = pd.DataFrame({
    "feature": feature_names,
    "importance": d_tree_10_importances
}).sort_values(by="importance", ascending=False)

rf_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_importances
}).sort_values(by="importance", ascending=False)

print("Decision Tree Top 10:", d_tree_10_df.head(10))
print("\nRandom Forest Top 10:", rf_df.head(10))

# Both models show that symbols like "$" and "!" and words like "free" are important for detecting spam.
# The Decision Tree focuses on a few main features, while Random Forest uses many features more evenly.
# This makes sense because spam emails often contain money-related words, promotions, and strong punctuation.

# Logistic Regression

# With scaled data
log_reg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_scaled.fit(X_train_scaled, y_train)  

pred_log_reg_scaled = log_reg_scaled.predict(X_test_scaled)
print("Accuracy for Logistic Regression on scaled data", accuracy_score(y_test, pred_log_reg_scaled))
print(classification_report(y_test, pred_log_reg_scaled))

# With PCA data
log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_pca.fit(X_train_pca, y_train)  

pred_log_reg_pca = log_reg_pca.predict(X_test_pca)
print("Accuracy for Logistic Regression on PCA", accuracy_score(y_test, pred_log_reg_pca))
print(classification_report(y_test, pred_log_reg_pca))

"""
Final Conclusion: 

Random Forest performs the best overall with the highest test accuracy (~0.94).
Logistic Regression also performs very well, slightly behind Random Forest.
Decision Tree performs worse and shows signs of overfitting as depth increases.

For PCA vs non-PCA:
For KNN, PCA and scaled data perform similarly.
For Logistic Regression, scaled data perform slightly better than PCA.
This mostly matches the expectation that scaling is very important, while PCA can reduce performance slightly by removing information.

For a spam filter, accuracy is not enough.
It is more important to reduce false positives(legitimate emails marked as spam)
because those emails are more costly for users than spam emails.
Therefore, presicion for the "spam" class is especially important.

"""

# --- Task 4: Cross-Validation ---

# KNN (unscaled data)
cv_scores_knn_unscaled = cross_val_score(knn_unscaled, X_train, y_train, cv=5)
print("KNN unscaled")
print(f"Mean: {cv_scores_knn_unscaled.mean():.3f}")
print(f"Std: {cv_scores_knn_unscaled.std():.3f}")

# KNN (scaled data)
cv_scores_knn_scaled = cross_val_score(knn_scaled, X_train_scaled, y_train, cv=5)
print("\nKNN scaled")
print(f"Mean: {cv_scores_knn_scaled.mean():.3f}")
print(f"Std: {cv_scores_knn_scaled.std():.3f}")

# KNN (PCA data)
cv_scores_knn_pca = cross_val_score(knn_pca, X_train_pca, y_train, cv=5)
print("\nKNN PCA")
print(f"Mean: {cv_scores_knn_pca.mean():.3f}")
print(f"Std: {cv_scores_knn_pca.std():.3f}")

# Decision Tree (max_depth=3)
cv_scores_d_tree_3 = cross_val_score(d_tree_3, X_train, y_train, cv=5)
print("\nDecision Tree (max_depth=3)")
print(f"Mean: {cv_scores_d_tree_3.mean():.3f}")
print(f"Std: {cv_scores_d_tree_3.std():.3f}")

# Decision Tree (max_depth=5)
cv_scores_d_tree_5 = cross_val_score(d_tree_5, X_train, y_train, cv=5)
print("\nDecision Tree (max_depth=5)")
print(f"Mean: {cv_scores_d_tree_5.mean():.3f}")
print(f"Std: {cv_scores_d_tree_5.std():.3f}")

# Decision Tree (max_depth=10)
cv_scores_d_tree_10 = cross_val_score(d_tree_10, X_train, y_train, cv=5)
print("\nDecision Tree (max_depth=10)")
print(f"Mean: {cv_scores_d_tree_10.mean():.3f}")
print(f"Std: {cv_scores_d_tree_10.std():.3f}")

# # Decision Tree (max_depth=None)
cv_scores_d_tree_None = cross_val_score(d_tree_None, X_train, y_train, cv=5)
print("\nDecision Tree (max_depth=None)")
print(f"Mean: {cv_scores_d_tree_None.mean():.3f}")
print(f"Std: {cv_scores_d_tree_None.std():.3f}")

# Random Forest
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5)
print("\nRandom Forest")
print(f"Mean: {cv_scores_rf.mean():.3f}")
print(f"Std: {cv_scores_rf.std():.3f}")

# Logistic Regression (scaled)
cv_scores_log_reg_scaled = cross_val_score(log_reg_scaled, X_train_scaled, y_train, cv=5)
print("\nLogistic regression scaled")
print(f"Mean: {cv_scores_log_reg_scaled.mean():.3f}")
print(f"Std: {cv_scores_log_reg_scaled.std():.3f}")

# Logistic Regression (PCA)
cv_scores_log_reg_pca = cross_val_score(log_reg_pca, X_train_pca, y_train, cv=5)
print("\nLogistic Regression PCA")
print(f"Mean: {cv_scores_log_reg_pca.mean():.3f}")
print(f"Std: {cv_scores_log_reg_pca.std():.3f}")

"""
Random Forest is the best model because it has highest average accuracy.
Logistic Regression (scaled) is the second best and shows strong performance with good stability.
KNN improves significantly after scaling and PCA, showing that feature scaling is important for distance-based model.
Decision Tree performs worse and is less stable, especially at higher depths, showing signs of overfitting.

The most stable model is Logistic Regression with PCA (lowest standard deviation), 
while Random Forest achieves the best overall accuracy.

Cross-validation gives a more reliable result than a single test/train split.

"""

# --- Task 5: Building a Prediction Pipeline ---

# Tree-based model pipeline (Random Forest does not need scaling) 
tree_based_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model on training data
tree_based_pipeline.fit(X_train, y_train)
#Predict on test data
y_pred = tree_based_pipeline.predict(X_test)

# Evaluate performance (accuracy + full metrics)
print("Random Forest Pipeline Accuracy:", tree_based_pipeline.score(X_test, y_test))
print(classification_report(y_test, y_pred))

# Non-tree model pipeline (Logistic Regression requires feature scaling)
log_reg_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

# Train pipeline (scaling + model)
log_reg_pipeline.fit(X_train, y_train)
# Predict on test data
log_reg_pred = log_reg_pipeline.predict(X_test)

# Evaluate performance
print("\nLogistic Regression Pipeline Accuracy:", log_reg_pipeline.score(X_test, y_test))
print(classification_report(y_test, log_reg_pred))

"""
Pipelines are different because models have different needs.
Logistic Regression needs scaling, while Random Forest does not.

Pipelines are useful because they combine all steps into one object.
This makes the workflow simpler and helps avoid mistakes when reusing or deploying the model.

"""