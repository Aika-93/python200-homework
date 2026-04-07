# --- Pandas ---

# Pandas Q1

import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# First three rows
print(f"First three rows:\n{df.head(3)}\n")

# Shape
print(f"Shape: {df.shape}\n")

# Data types
print(f"Data types:\n{df.dtypes}")

# Pandas Q2

# Print only students who passed and have a grade above 80
print(df[(df["passed"]) & (df["grade"] > 80)])

# Pandas Q3

# Add a new column 
df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4

# Add a new column
df["name_upper"] = df["name"].apply(str.upper)
print(df[["name", "name_upper"]])

# Pandas Q5

# Compute the mean grade for each city
print(df.groupby("city")["grade"].mean())

# Pandas Q6

# Replace the value "Austin" in the "city" column with "Houston"
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Q7

# Sort the DataFrame by "grade" in descending order and print the top 3 rows.
print(df.sort_values("grade", ascending = False).head(3))

# --- Numpy ---

# Numpy Q1

import numpy as np

numbers = np.array([10, 20, 30, 40, 50])

# Print shape
print(f"Shape: {numbers.shape}")

#Print data types
print(f"Data types: {numbers.dtype}")

#Print dimensions
print(f"Dimensions: {numbers.ndim}")


# Numpy Q2

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Print shape
print(f"Shape: {arr.shape}")

# Print size
print(f"Size: {arr.size}")

# Numpy Q3

# Slice out the top-left 2x2 block and print it
print(arr[0:2, 0:2])

# Numpy Q4

# 3x4 array of zeros
zeros_arr = np.zeros((3, 4))
print(f"Zeros array: {zeros_arr}")

# 2x5 array of ones
ones_arr = np.ones((2, 5))
print(f"Ones array: {ones_arr}")

# Numpy Q5

arr1 = np.arange(0, 50, 5)
print(arr1)

# Print shape
print(f"Shape: {arr1.shape}")

# Print mean
print(f"Mean: {np.mean(arr1)}")

# Print sum
print(f"Sum: {np.sum(arr1)}")

#Print standard deviation 
print(f"Standard Deviation: {np.std(arr1)}")

# Numpy Q6

#Generate 200 random values
arr2 = np.random.normal(0, 1, 200)

# Print mean
print(f"Mean: {np.mean(arr2)}")

#Print Standard Deviation
print(f"Standard Deviation: {np.std(arr2)}")

# --- Matplotlib ---

# Matplotlib Q1

import matplotlib.pyplot as plt

# Plot the following data as a line plot.
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y, color="red")
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2

# Create a bar plot for the following subject scores.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores, color="red")
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3

# Create a scatter plot on the same figure
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="purple", label="Dataset 1")
plt.scatter(x2, y2, color="yellow", label="Dataset 2")

plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# Matplotlib Q4

# Create a figure with 1 row and 2 subplots side by side
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

fig, axes = plt.subplots(1, 2, figsize=(12,6))

# Left plot
axes[0].plot(x, y, color="red")
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# Right plot
axes[1].bar(subjects, scores, color="blue")
axes[1].set_title("Subject Scores")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")

plt.tight_layout()
plt.show()

# --- Descriptive Statistics Review ---

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")


# Descriptive Stats Q2

data_normal = np.random.normal(65, 10, 500)

# Create a histogram of 500 normally distributed values(mean=65, std=10)
plt.hist(data_normal, bins=20, color="purple", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

#Plot a boxplot comparing two groups and label them
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()


# Descriptive Stats Q4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Compare normal and exponential distributions using boxplots
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()

"""
- The exponential distribution is right-skewed.
- The right whisker is longer, indicating the presence of larger values(outliers) on the right side.
- The median is closer to the left side of the box, showing that most values are smaller.
- For this distribution, the median is a better measure of central tendency because the mean is affected by large values.

"""

# Descriptive Stats Q5

import statistics as stats

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"Mean(data1): {np.mean(data1)}, Mean(data2): {np.mean(data2)}")
print(f"Median(data1): {np.median(data1)}, Median(data2): {np.median(data2)}")
print(f"Mode(data1): {stats.mode(data1)}, Mode(data2): {stats.mode(data2)}")

"""
- The mean is much higher in data2 because the value 150 is an outlier that pulls the average upward.
- The median is not affected by extreme values , so it remains more representive of the central tendency.
"""
# --- Hypothesis Testing Review ---

# Hypothesis Q1

from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Independent samples t-test
t_stat1, p_val1 = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat1}")
print(f"P-value: {p_val1:.6f}")

# Hypothesis Q2

if p_val1 < 0.05:
    print("The difference is statistically significant")
else:
    print("No statistically significant difference detected")

# Hypothesis Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Paired t-test on the before/after scores
t_stat2, p_val2 = stats.ttest_rel(before, after)

print(f"T-statistic: {t_stat2}")
print(f"P-value: {p_val2:.6f}")

# Hypothesis Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]

# One-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70.
t_stat3, p_val3 = stats.ttest_1samp(scores, 70)
print(f"T-statistic: {t_stat3}")
print(f"P-value: {p_val3:.6f}")

# Hypothesis Q5

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores.
t_stat4, p_val4 = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"P-value: {p_val4:.6f}")

# Hypothesis Q6

print("The difference in average is statistically significant and group_b scores are higher than group_a, so it's unlikely due to chance")

# --- Correlation Review ---

# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Compute the Pearson correlation between x and y
corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
print(corr_matrix[0,1])
"""
- The correlation is expected to be 1 because y increases perfectly linearly as x increases.
- This means there is a perfect positive linear relationship between x and y.

"""
# Correlation Q2

from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

# Compute the correlation between x and y
r, p = pearsonr(x, y)
print(f"Correlation: {round(r,2)}")
print(f"P-value: {round(p,4)}")

# Correlation Q3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr = df.corr()

# Use df.corr() to compute the correlation matrix. 
print(corr)

# Correlation Q4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

# Create a scatter plot of x and y below, which have a negative relationship.
plt.scatter(x, y, color="teal")
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Correlation Q5

import seaborn as sns

# Using the correlation matrix from Q3, create a heatmap.
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines ---

# Pipeline Q1

import numpy as np
import pandas as pd

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": np.mean(series),
        "median": np.median(series),
        "std": np.std(series),
        "mode": series.mode()[0]
    }

def data_pipeline(arr):
    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)
    return summary

result = data_pipeline(arr)

for key, value in result.items():
    print(f"{key}: {value}")