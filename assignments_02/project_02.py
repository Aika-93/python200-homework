# - - - Mini-Project -- Predicting Student Math Performance - - -

# The CSV file uses semicolons (;) as separators between fields.
# Therefore, we need to use sep=";", when reading it with pandas

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# --- Task 1: Load and Explore ---

# load dataset
data = pd.read_csv("student_performance_math.csv", sep=";")

# basic exploration
print(data.shape)
print(data.head(5))
print(data.dtypes)

# plot distribution of final grades (G3)
plt.hist(data["G3"], bins=21, color="purple", edgecolor="black")

# chart labels
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3 Grade")
plt.ylabel("Frequency")

# save figure
output_path = Path("outputs/g3_distribution.png")
plt.savefig(output_path)

plt.show()
plt.close()



# --- Task 2: Preprocess the Data ---

# shape before removing zeros
print("Before:", data.shape)

# remove rows where G3 == 0
data_clean = data[data["G3"] != 0].copy()

# shape after removing zeros
print("After:", data_clean.shape)

# reason for filtering:
# Keeping G3 = 0 rows could distort the model because they likely represent student who did not take the exam , not real performance.

# convert yes/no columns to 1/0
data_clean["schoolsup"] = data_clean["schoolsup"].map({"yes": 1, "no": 0})
data_clean["internet"] = data_clean["internet"].map({"yes": 1, "no": 0})
data_clean["higher"] = data_clean["higher"].map({"yes": 1, "no": 0})
data_clean["activities"] = data_clean["activities"].map({"yes": 1, "no": 0})

# convert sex column to 0/1
data_clean["sex"] = data_clean["sex"].map({"M": 1, "F": 0})

# correlation in original data (includes G3 = 0 cases)
corr_coeff = data["absences"].corr(data["G3"])

# correlation after removing students who did not take the exam (G3 = 0)
corr_coeff_2 = data_clean["absences"].corr(data_clean["G3"])

print(corr_coeff)
print(corr_coeff_2)

# G3 = 0 students likely did not take the exam , which adds noise and weakens the relationship between absences and G3.



# --- Task 3: Exploratory Data Analysis ---

# Compute Pearson correlationbetween numeric features and G3, sorted from negative to positive. 
corrs = data_clean.corr(numeric_only=True)["G3"]
print(corrs.sort_values())

# Scatter plot: relationship between past failures and final grade
plt.scatter(data_clean["failures"], data_clean["G3"], color="teal")
plt.title("Failures vs Final Grade (G3)")
plt.xlabel("Failures")
plt.ylabel("Final Grade (G3)")

output_path = Path("outputs/failures_vs_G3.png")
plt.savefig(output_path)
plt.show()
plt.close()

# Students with more past failures tend to have lower final grades (G3)
 
# Heatmap: correlations between all numeric features and G3
corrs_matrix = data_clean.corr(numeric_only=True)[["G3"]]
plt.figure(figsize=(10,6))
sns.heatmap(corrs_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

output_path = Path("outputs/correlation_heatmap.png")
plt.savefig(output_path)
plt.show()
plt.close()

# The heatmap shows that failures and absences have the strongest negative correlation with G3,
# while G1 and G2 are strongly positively correlated with G3



# --- Task 4: Baseline Model ---

# Use only failures to predict final grade (G3)
X = data_clean[["failures"]]
y = data_clean["G3"]

# split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on test set
y_pred = model.predict(X_test)

# slope shows how G3 changes with each additional failure
print("Slope :", model.coef_[0])

# RMSE = average prediction error 
print("RMSE : ", np.sqrt(mean_squared_error(y_test, y_pred)))

# R2 = how well the model explains the data 
print("R2 : ", model.score(X_test, y_test))

# Each additional failure decreases the grade by about 1.4 points.
# The model has an average error of about 3 points,
# and R2 is low(~0.09), meaning failures alone explains very little of the variation in student performance.



# --- Task 5: Build the Full Mode ---

# features used for prediction
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X1 = data_clean[feature_cols].values
y1 = data_clean["G3"].values

# train/test split(80/20)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# create and train linear regression model
model_1 = LinearRegression()
model_1.fit(X_train1, y_train1)

# make predictions on test set
y_pred1 = model_1.predict(X_test1)

# model performance (R2)
print("R2 of test set: ", model_1.score(X_test1, y_test1))
print("R2 of train set: ", model_1.score(X_train1, y_train1))

# show feature importance (coefficients)
for name, coef in zip(feature_cols, model_1.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Schoolsup is negatively correlated, likely because it is given to weaker students, not because it lowers grades.
# Train and test R2 are close, so there is no strong overfitting, but the model is still weak overall.

# I would keep failures, studytime, higher, because they have the strongest and most meaningful relationships with G3.
# I would drop freetime, and activities since they show almost no impact.
# Schoolsup may be dropped because it reflects which students receive support rather than directly causing higher or lower grades.



# --- Task 6: Evaluate and Summarize ---

# scatter plot: predicted vs actual value
plt.scatter(y_pred1, y_test1)

# find range for diagonal reference line 
min_val = min(y_pred1.min(), y_test1.min())
max_val = max(y_pred1.max(), y_test1.max())

# perfect prediction line (y = x)
plt.plot([min_val, max_val], [min_val, max_val])

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# save plot to file
output_path = Path("outputs/predicted_vs_actual.png")
plt.savefig(output_path)
plt.show()
plt.close()

# The model is slightly less accurate at higher grades, likely because there are fewer examples and more factors affecting high scores.
# Points above the diagonal mean the model underestimated the grade, while points below the diagonal mean it overestimated it.

"""
Summarize:

- The filtered dataset contains 357 students, and the test set is about 20% of the data.

- The RMSE means the model is typically off by about 3 grade points on a 0-20 scale, which is a noticeable but not extreme error.
- The R2 is low (~0.15), meaning the model explains only a small part of the differences in student grades.

- The strongest positive coefficient is "higher" (students who want to pursue higher education tend to have higher grades).
- The strongest negative coefficient is "failures" (more past failures strongly reduce final grades).

- A surprising result is that schoolsup has a negative coefficient, which likely reflects that it is given 
    more often to weaker students rather than improving grades directly.
"""



# --- Neglected Feature: The Power of G1 ---

# features used for prediction
feature_cols_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime", "G1"]
X2 = data_clean[feature_cols_g1].values
y2 = data_clean["G3"].values

# train/test split(80/20)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# create and train linear regression model
model_2 = LinearRegression()
model_2.fit(X_train2, y_train2)

# make predictions on test set
y_pred2 = model_2.predict(X_test2)

# model performance (R2)
print("R2 with G1 (test): ", model_2.score(X_test2, y_test2))
print("R2 with G1 (train): ", model_2.score(X_train2, y_train2))


"""
A high R2 does not mean G1 causes G3, it only shows theay are strongly related.
This model is not useful for early intervention because G1 is not available at the beginning.
To help students early, teachers would need to use information available before G1, like attendance , study habits and past performance.
"""