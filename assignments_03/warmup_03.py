import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

# Q1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Q2

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.round(X_train_scaled.mean(axis=0), 3))
# Fit only on X_train to avoid data leakage from test set.

# --- KNN ---

# Q1 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Classification report: \n", classification_report(y_test, preds))

# Q2

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
preds_scaled = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, preds_scaled))
# Scaling did not help for this dataset because the original feature values already worked well for distance calculation.

# Q3

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: {score:.3f}")

print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std: {cv_scores.std():.3f}")

# Cross-validation is more trustworthy than a single train/test split because it evaluates the model on multiple different data splits.

# Q4

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}: mean={scores.mean():.3f}")

# I would choose k=5 because it has the highest cross-validation accuracy and is simpler than k=7 while giving the same performance.

# --- Classifier Evaluation ---

# Q1

cm = confusion_matrix(y_test, preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()

# The model does not confuse any species, all predictions are correct.

# --- Decision Trees ---

# Q1

d_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
d_tree.fit(X_train, y_train)
tree_preds = d_tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, tree_preds))
print("Classification report: \n", classification_report(y_test, tree_preds))

# Desicion Tree performs slightly worse than KNN in terms of accuracy.
# Scaling does not affect Decision Trees because they are not based on distance calculations.


# --- Logistic Regression and Regularization ---

# Q1

log_reg_1 = LogisticRegression(C=0.01, max_iter=1000, solver='liblinear')
log_reg_1.fit(X_train_scaled, y_train)

log_reg_2 = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_2.fit(X_train_scaled, y_train)

log_reg_3 = LogisticRegression(C=100, max_iter=1000, solver='liblinear')
log_reg_3.fit(X_train_scaled, y_train)

print("C=0.01:", np.abs(log_reg_1.coef_).sum())
print("C=1.0:", np.abs(log_reg_2.coef_).sum())
print("C=100:", np.abs(log_reg_3.coef_).sum())

# As C increases, the total magnitude of the coefficients increases.
# This show that regularization becomes weaker and the model is allowed to use large weights.

# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting


# Q1

print(X_digits.shape)
print(images.shape)

fig, axes = plt.subplots(1, 10, figsize=(12,3))
for i in range(10):
    axes[i].imshow(images[i], cmap='gray_r')
    axes[i].set_title(f"{i}")
    axes[i].axis('off')
plt.tight_layout()

plt.savefig('outputs/sample_digits.png')
plt.show()

# Q2

pca = PCA(svd_solver="randomized", random_state=0)
pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.savefig('outputs/pca_2d_projection.png')
plt.show()

# Same digits tend to form clusters in the 2D PCA space, but there is still overlap between different classes.

# Q3

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumulative_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Explained Variance")

plt.savefig('outputs/pca_variance_explained.png')
plt.show()

# Approximately 10-15 principal components are needed to explain about 80% of the variance.

# Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]
num_digits = 5

fig, axes = plt.subplots(len(n_values) + 1, num_digits, figsize=(10,8))

for i in range(num_digits):
    axes[0, i].imshow(images[i], cmap='gray_r')
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')

for row, n in enumerate(n_values, start=1):
    for col in range(num_digits):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap='gray_r')
        axes[row, col].set_title(f"n={n}")
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/pca_reconstructions.png')
plt.show()

# Digits become recognizable at about 10-15 components , which is also where the variance curve reaches ~80%.
 