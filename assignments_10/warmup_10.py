# --- LLMs as Transform ---

# Q1

"""
For each task below, write a one-sentence comment saying whether you would use an LLM or deterministic code, and why.

1. Parse the string "Jan 5th, 2024" into an ISO date format like "2024-01-05".
   Deterministic code: I would use deterministic code because date parsing follows fixed rules and does not require reasoning.

2. Classify a customer support ticket -- "my card was charged twice" -- into one of: billing, technical, or general.
    LLM: I would use an LLM because it needs to understand the meaning of the text and classify it into the correct category.

3. Calculate the average of a list of numbers.
    Deterministic code: I would use deterministic code because calculating an average is a precise mathematical operation.

4. Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)".
    LLM: I would use an LLM because company names may appear in different formats and require understanding of the text.

5. Determine whether a product review is more than 100 words long.
    Deterministic code: I would use deterministic code because counting words is a simple rule-based task.

"""

# Q2

"""
system = "Summarize this product review in a few sentences."

The propmpt is too vague and may produce summaries with different lengths and formats.
This makes the output difficult to parse and store consistently in a pipeline.
I would rewrite the prompt to require a specific format.

New prompt:
system = "Summarize this product review in exactly two sentences.
          Return only the summary text and nothing else."
"""

# Q3

"""
Sequential processing would take about 50,000 seconds, which is approximately 13.9 hours.

One practical strategy would be to use Batch API processing so that many records can be handled more efficiently at scale.

"""

# --- Azure OpenAI ---

# Q1

"""
One reason is security and integration with Azure services.
Organizations can manage everything in one environment using Azure authentication and permissions.

Another reason is that data sent to Azure OpenAI is not used for model training, 
which is important for privacy and compliance requirenments.

"""

# Q2

"""
1. azure_endpoint:
    The URL of the Azure OpenAI resource where requests are sent.
2. api_key: 
    The Azure OpenAi key used for authentication and access control.
3. api_version: 
    The version of the Azure OpenAI being used.
    It ensures compatibility between the application and the Azure service.

"""

# Q3

"""
In Azure OpenAI, the model parameter does not take a model name like "gpt-4o-mini".
Instead, it takes the deployment name created in Azure OpenAI.
You can find the deployment name in the Azure Portal under the Azure OpenAI resource in the Deployments section, 
where the model is deployed.

"""