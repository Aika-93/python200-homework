import numpy as np
import pandas as pd
from prefect import task, flow
from prefect.logging import get_run_logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path
import seaborn as sns
from scipy.stats import pearsonr

# Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_data():
    """
    - Load multiple yearly CSV files
    - Fix formatting issues
    - Add year column
    - Merge into one dataset for analysis

    """
    logger = get_run_logger()

    folder = Path("resources/happiness_project")
    files = list(folder.glob("*.csv"))

    data_list = []

    for file in files:
        df = pd.read_csv(file, sep=";", decimal=",")

        df["year"] = int(file.stem.split("_")[-1])

        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    print(data.columns)
    print(data["Regional indicator"].unique())
    
    output_path = Path("outputs/merged_happiness.csv")
    data.to_csv(output_path, index=False)

    return data

@task
def clean_data(data):
    """
    - Remove rows with missing important values
    - Keep only clean data for analysis
    """
    logger = get_run_logger()

    cols_needed = [
        "Happiness score",
        "GDP per capita", 
        "Social support",
        "Healthy life expectancy",
        "Freedom to make life choices",
        "Perceptions of corruption"
    ]

    data = data.dropna(subset=cols_needed)

    logger.info(f"After cleaning shape: {data.shape}")

    return data

# Task 2: Descriptive Statistics
@task
def describe(data):
    """
    - Calculate overall statistics (mean, median, std)
    - Group average happiness by year and region
    """
    logger = get_run_logger()

    mean = data["Happiness score"].mean()
    median = data["Happiness score"].median()
    std = data["Happiness score"].std()

    year_mean = data.groupby("year")["Happiness score"].mean()

    region_mean = data.groupby("Regional indicator")["Happiness score"].mean()

    logger.info(f"Mean: {mean}")
    logger.info(f"Median: {median}")
    logger.info(f"Std: {std}")
    logger.info(f"Year_mean: {year_mean}")
    logger.info(f"Region_mean: {region_mean}")

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "year_mean": year_mean,
        "region_mean": region_mean
        }

# Task 3: Visual Exploration
@task
def visual_explore(data):
    """
    - Create histogram of happiness scores
    - Create boxplot by year
    - Create scatter plot (GDP vs Happiness)
    - Create correlation heatmap
    """
    logger = get_run_logger()

    #1. Histogram
    plt.figure()
    plt.hist(data["Happiness score"], bins=20, color="purple", edgecolor="black")
    plt.title("Happiness scores across all years")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")

    output_path = Path("outputs/happiness_histogram.png")
    plt.savefig(output_path)
    logger.info(f"Histogram saved to {output_path}")
    plt.close()

    # 2. Boxplot
    plt.figure()
    sns.boxplot(x="year", y="Happiness score", data=data)
    plt.title("Happiness score distributions across years")
    plt.xlabel("Year")
    plt.ylabel("Score")

    output_path = Path("outputs/happiness_by_year.png")
    plt.savefig(output_path)
    logger.info(f"Boxplot saved to {output_path}")
    plt.close()

    # 3. Scatter plot
    plt.figure()
    plt.scatter(data["GDP per capita"], data["Happiness score"], color="purple")
    plt.title("Relationship between GDP per capita and happiness score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness score")
    
    output_path = Path("outputs/gdp_vs_happiness.png")
    plt.savefig(output_path)
    logger.info(f"Scatter plot saved to {output_path}")
    plt.close()

    # 4. Heatmap
    numeric_data = data.select_dtypes(include="number")
    corr = numeric_data.corr()

    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson correlations between all numeric columns")
    
    output_path = Path("outputs/correlation_heatmap.png")
    plt.savefig(output_path)
    logger.info(f"Heatmap plot saved to {output_path}")
    plt.close()

# Task 4: Hypothesis Testing
@task
def run_ttest(data):
    """
    - Compare happiness between 2019 and 2020
    - Calculate t-test and means
    """
    logger = get_run_logger()
    score_2019 = data[data["year"] == 2019]["Happiness score"].dropna()
    score_2020 = data[data["year"] == 2020]["Happiness score"].dropna()

    t_stat, p_val = ttest_ind(score_2019, score_2020)
    
    mean_2019 = score_2019.mean()
    mean_2020 = score_2020.mean()

    logger.info("T-test result: t=%.2f, p=%.4f", t_stat, p_val)
    logger.info("Mean happiness 2019: %.1f, Mean happiness 2020: %.1f", mean_2019, mean_2020)

    if p_val < 0.05:
        logger.info("Conclusion: There is a statistically significant difference in happiness scores between 2019 and 2020, "
                    "suggesting the pandemic may have impacted global happiness levels")
    else:
        logger.info("Conclusion: No statistically significant difference in happiness scores between 2019 and 2020, "
                    "suggesting no clear measurable impact in this dataset")

    return t_stat, p_val, mean_2019, mean_2020

# Task 4: Second Hypothesis Testing
@task
def run_ttest2(data):
    """
    - Compare happiness between Asia and Europe 
    - Run second t-test
    """
    logger = get_run_logger()
    asia = data[data["Regional indicator"].isin([
        "Southeast Asia",
        "East Asia",
        "South Asia"
    ])]["Happiness score"]

    europe = data[data["Regional indicator"].isin([
        "Western Europe",
        "Central and Eastern Europe"
    ])]["Happiness score"]

    t_stat2, p_val2 = ttest_ind(asia, europe)

    logger.info("Second t-test result: t=%.2f, p=%.4f", t_stat2, p_val2)

    if p_val2 < 0.05:
        logger.info("Conclusion: There is a significant difference between regions")
    else:
        logger.info("Conclusion: No significant difference between regions.")
    
    return t_stat2, p_val2

# Task 5: Correlation and Multiple Comparisons
@task
def correlation(data):
    """
    - Calculate correlation with happiness score
    - Log r and p values
    - Apply Bonferroni correction
    """
    logger = get_run_logger()

    target = "Happiness score"
    numeric_cols = data.select_dtypes(include="number").columns
    
    results = []

    for col in numeric_cols:
        if col == target:
            continue

        subset = data[[col, target]].dropna()

        if len(subset) < 2:
            logger.info(f"{col}: not enough data")
            continue

        r, p = pearsonr(subset[col], subset[target])

        logger.info(f"{col}: r = {round(r, 2)}, p = {round(p, 4)}")

        results.append((col, r, p))

    # Bonferroni correction
    number_of_tests = len(results)
    adjusted_alpha = 0.05 / number_of_tests

    logger.info(f"Adjusted alpha: {adjusted_alpha}")

    for col, r, p in results:
        logger.info(
            f"{col} | raw: {p < 0.05} | corrected: {p < adjusted_alpha}"
        )
    return results

# Task 6: Summary Report
@task
def summary_report(data, ttest_result, ttest2_result, correlation):
    """
    - Log dataset size (countries,year)
    - Show top and bottom regions
    - Log t-test results
    - Find strongest correlation
    """
    logger = get_run_logger()

    # 1. size
    logger.info(f"Countries: {data['Country'].nunique()}")
    logger.info(f"Years: {data['year'].nunique()}")

    # 2. regions
    region_mean = data.groupby("Regional indicator")["Happiness score"].mean()

    logger.info(f"Top 3 regions: {region_mean.sort_values(ascending=False).head(3).to_dict()}")
    logger.info(f"Bottom 3 regions: {region_mean.sort_values().head(3).to_dict()}")

    # 3. t-test results (from existing tasks)

    t1, p1, mean_2019, mean_2020 = ttest_result
    t2, p2 = ttest2_result

    logger.info(
        f"T-test 1: p={round(p1, 4)} -> {'significant' if p1 < 0.05 else 'not significant'}"
    )
    logger.info(
        f"T-test 2: p={round(p2, 4)} -> {'significant' if p2 < 0.05 else 'not significant'}"
    )

    # 4. strongest correlation
    best_var = ""
    best_r = -1

    for col, r, p in correlation:
        if abs(r) > best_r:
            best_r = abs(r)
            best_var = col

    logger.info(f"Strongest correlation: {best_var} (r={round(best_r, 2)})")

@flow
def pipeline_flow():
    data = load_data()
    data = clean_data(data)

    describe(data)
    visual_explore(data)

    t1 = run_ttest(data)
    t2 = run_ttest2(data)
    corr = correlation(data)
    
    summary_report(data, t1, t2, corr)

if __name__=="__main__":
    pipeline_flow()