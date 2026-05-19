# ---------- Mini-Project — World Happiness Agent ----------

# Pre-task: Load the Data

import numpy as np
import pandas as pd
import os 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from prefect import task, flow
from prefect.logging import get_run_logger
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
from scipy.stats import pearsonr
import glob

DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"

# Task 1: Define Your Tools

# Shared dataframe used accross tools
df = None 

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    
    Returns: 
        dict: A dictionary containing the 'shape' and 'columns' of the dataframe.
    """
    global df 

    fallback_pattern = "../assignments_01/resources/happiness_project/*.csv"

# Load merged file if available
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print("Loaded merged_happiness.csv directly from assignments_01")
    else:
        # Fallback: merge yearly CSV files
        print("Merged file not found. Falling back to merging CSVs from assignments_01")
        files = glob.glob(fallback_pattern)

    # Return error if no files found 
        if not files:
            return{"error": f"No data found!"}
        
        # Merge all CSV files into one dataframe
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"Successfully merged {len(files)} files from assignments_01 resources")
    
    # Save current dataset for plotting access
    df.to_csv("current_happiness_data.csv", index=False)

    # Return dataset metadata
    return {
        "shape": df.shape,
        "columns": list(df.columns)
        }
    
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    
    Args:
        column (str): Name of the column to summarize.

    Returns: 
        dict: Dictionary with descriptive statistics.
    """
    global df

    # check if data is loaded 
    if df is None:
        load_happiness_data()
    
    # check if column exists 
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    
    # Return summary statistics
    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    
    Args:
        col1 (str): First numeric column name.
        col2 (str): Second numeric column name.

    Returns:
        dict: Dictionary containing correlation coefficient and p-value.
    """
    global df

    # check dataset loaded 
    if df is None:
        load_happiness_data()
        
    # check column exists
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found."}
    
    # Remove missing values before correlation 
    df_clean = df[[col1, col2]].dropna()
    corr, p_value = pearsonr(df_clean[col1], df_clean[col2])

    # Return rounded correlation results
    return {
        "col1": col1, 
        "col2": col2,
        "pearson_r": round(corr, 4),
        "p_value": round(p_value, 4)
    }

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column (str): Name of the numeric column to sort by.
        year (int): Year to filter dataset.
        n (int): Number of top countries to return.

    Returns:
        dict: Dictionary containing a list of top countries with their values.
    """
    global df

    # check dataset loaded
    if df is None:
        load_happiness_data()
    
    # check column exists
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    
    # check year exists
    if "year" not in df.columns:
        return {"error": "Year column not found in dataset."}
    
    try: 
        # Filter dataset by selected year
        filtered = df[df["year"] == year]

        # Sort and select top countries
        top = (
            filtered.sort_values(by=column, ascending=False)
            .head(n)[["Country", column]]
        )
        
        # Return results as records
        return {
            "results": top.to_dict(orient="records")
        }
    
    # Handle unexpected errors
    except Exception as e:
        return {"error": str(e)}


# Task 2: Build the Agent

from smolagents import CodeAgent, OpenAIServerModel
from dotenv import load_dotenv

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.

CRITICAL FOR CUSTOM PYTHON CODE/PLOTS:
The global variable 'df' belongs to the host script and is NOT defined in your code interpreter sandbox.
Whenever you write custom Python code blocks(like plotting a chart), you MUST explicitly load the data first using:
import pandas as pd
df = pd.read_csv("current_happiness_data.csv")

pay close attention to column names (e.g., 'Regional indicator', 'Happiness score', 'year'). Use them exactly as they appear.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats", "numpy"],
    max_steps=8,
)


if __name__ == "__main__":

    # Task 3: Run Guided Queries

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)


    # Task 4: Your Own Questions

    # My query 1
    my_query_1 = "What are the top 5 countries in 2019 by happiness score?"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    #Query 1: tool use (get_top_n_countries)

    # My query 2
    my_query_2 = "Plot the average happiness score per year and save it as a line chart. Save the plot to outputs/average_happiness_over_time.png."
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # Query 2: code generation (matplotlib plot)


# Task 5: Reflection

"""
1. In Query 3, the agent used the p-value to determine statistical significance.
   The p-value was 0.0, which is far below the common threshold of 0.05,
   so the correlation was correctly identified as statistically significant.

2. I was surprised by the agent's ability to generate and execute its own Python code when no predefined tool was available.
   In particular, for the plotting task, the agent independently wrote Matplotlib code to create a visualization.
   This shows that the model can go beyond fixed tools and dynamically solve tasks by writing custom code when needed, 
   which makes it much more flexible and powerful than I initially expected.

3. One additional tool that would make the agent more useful is a built-in safeguard that prevents it from using synthetic or 
   fabricated data when the real dataset is unavailable.
   In my experience, the agent sometimes generated synthetic datasets in order to continue completing the task instead 
   of clearly failing or asking for clarification. 
   A stricter mechanism that forces the agent to either use real data or return an error would make the outputs more reliable 
   and prevent misleading results.

"""