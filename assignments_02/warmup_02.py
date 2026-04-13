# --- scikit-learn API ---

# scikit-learn Q1

import numpy as np
from sklearn.linear_model import LinearRegression

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years,salary)

new_years = np.array([[4], [8]])
predicted = model.predict(new_years)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction for 4 years:", predicted[0])
print("Prediction for 8 years:", predicted[1])

# scikit-learn Q2

x = np.array([10, 20, 30, 40, 50])
print(x.shape)
x = np.array([10, 20, 30, 40, 50]).reshape(-1, 1) 
print(x.shape)
# We need to reshape it because the input x must be a 2D array. Scikit learn does not accept a 1D array, otherwise it may cause an error.

# scikit-learn Q3

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pathlib import Path

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print(kmeans.cluster_centers_)
print(np.bincount(labels))

plt.scatter(X_clusters[:,0], X_clusters[:,1], c = labels, cmap="viridis", s = 60, alpha=0.7)
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c="black", marker="x", s=200)

plt.title("K-Means Clusters ")
plt.xlabel("X")
plt.ylabel("Y")

output_path = Path("outputs/kmeans_clusters.png")
plt.savefig(output_path)
plt.show()
plt.close()

# --- Linear Regression ---

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Q1

plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")

output_path = Path("outputs/cost_vs_age.png")
plt.savefig(output_path)
plt.show()
plt.close()

# Yes, two distinct groups are clearly visible.
# Non-smokers tend to have lower medical cost, while smokers have significantly higher cost.
# This suggests that the smoker variable has a strong impact on medical expenses.

# Linear Regression Q2

X = age.reshape(-1, 1)
y = cost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Linear Regression Q3

model =LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope :", model.coef_[0])
print("Intercept :", model.intercept_)
print("RMSE : ", np.sqrt(np.mean((y_pred - y_test) ** 2)))
print("R-2 : ", model.score(X_test, y_test))

# The slope shows us how much the predicted medical cost increases for each additional year of age.

# Linear Regression Q4

X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)
y_pred_f = model_full.predict(X_test_f)

print("New R-2 : ", model_full.score(X_test_f, y_test_f))
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# The smoker coefficient represents how mush more (or less) medical cost is for smokers compared to non-smokers, holding age constant.

# Linear Regression Q5

plt.scatter(y_pred_f, y_test_f)
plt.plot([y_pred_f.min(), y_pred_f.max()],
         [y_pred_f.min(), y_pred_f.max()])
plt.title("Predicted vs Actual")
plt.xlabel("Predicted")
plt.ylabel("Actual")

output_path = Path("outputs/predicted_vs_actual.png")
plt.savefig(output_path)
plt.show()
plt.close()

# Points above the diagonal mean the model underestimates the true value.
# Points below the diagonal mean the model overestimates the true value.