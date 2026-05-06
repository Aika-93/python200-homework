# --- Completions API ---


# API Q1

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response:",response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)


# API Q2

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )

    print(f"Temperature {temp}:")
    print(response.choices[0].message.content)

# As temperature increases, responses become more creative and varied.
# At temperature 0. responses are consistent and reproducible.
# I would use temperature = 0 for consistent output.


# API Q3

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for i, choice in enumerate(response.choices):
    print(f"Response {i+1}:")
    print(choice.message.content)


# API Q4

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    temperature=0,
    max_tokens=15
)
print(response.choices[0].message.content)

# The response was cut off because max_tokens limits the output length.
# In real applications max_tokens is used to control cost and response length.


# --- System Messages and Personas ---

# System Messages Q1

# Tutor personality 1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = messages
)

print(response.choices[0].message.content)

# Tutor personality 2
messages = [
    {"role": "system", "content": "You are a strict coding mentor who gives short answers"},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response2 = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = messages
)

print(response2.choices[0].message.content)


# System Messages Q2

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(response.choices[0].message.content)

# The model knows Jordan's name because the full conversation history is passed in the same request.
# The model is stateless, so it does not remember anything between calls.


# --- Prompt Engineering ---

# Prompt Engineering Q1

prompt = "Classify the sentiment of each review below as positive, negative, or mixed. Return one label per review in order."

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

content = f"""{prompt}
Reviews:
{reviews[0]}
{reviews[1]}
{reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": content}],
    temperature=0
)

print(response.choices[0].message.content)


# Prompt Engineering Q2

prompt = """
Classify the sentiment of each review below as positive, negative, or mixed. Return one label per review in order. 
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
"""

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

content = f"""{prompt}
Reviews:
{reviews[0]}
{reviews[1]}
{reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": content}],
    temperature=0
)

print(response.choices[0].message.content)

# Yes, adding one example made the output more structured.
# The model follows the format shown in the example.


# Prompt Engineering Q3

prompt = """
Classify the sentiment of each review below as positive, negative, or mixed. Return one label per review in order. 

Example 1: Review: "The product works perfectly."
Sentiment: positive

Example 2: Review: "The app is slow and crashes frequently."
Sentiment: negative

Example 3: Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
"""

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

content = f"""{prompt}
Reviews:
{reviews[0]}
{reviews[1]}
{reviews[2]}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": content}],
    temperature=0
)

print(response.choices[0].message.content)

# Zero-shot: used for simple tasks without examples.
# One-shot: used when you need to show format with one example.
# Few-shot: used for better accuracy and more consistent outputs by giving multiple examples.


# Prompt Engineering Q4

prompt = """
Show your step by step reasoning, then give the final answer on its own line labeled: Final answer: <value>.

Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?

"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0
)

print(response.choices[0].message.content)

# Asking the model reason step by step improves accuracy because it breaks the problem into smaller parts, reducing mistakes in multi-step calculation.


# Prompt Engineering Q5

import json

prompt = """
Classify the sentiment of review and respond ONLY with valid JSON.
Respond ONLY with raw JSON. Do not include markdown or backticks.

Keys: sentiment(positive/negative/mixed), confidence(0-1 scale), reason (one sentence)

Review: "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0
)

raw = response.choices[0].message.content
print("Raw response: ", raw)

try:
   result=json.loads(raw)
   print("Sentiment:", result["sentiment"])
   print("Confidence: ", result["confidence"])
   print("Reason: ", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")
    print("Raw response:", raw)


# Prompt Engineering Q6

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0
)
print(response.choices[0].message.content)

user_text2= "I love cooking Korean food and trying new recipes."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text2}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}], 
    temperature=0
)
print(response.choices[0].message.content)

# Delimiters help separate instructions from user input, preventing the model from confusing the task with the data.


# --- Local Models with Ollama ---

# Ollama Q1

# Ollama output: 
"""
A large language model is a type of AI model designed to understand and generate human-like text, such as writing essays or translating 
between languages. It works by analyzing vast amounts of text and learning patterns, allowing it to perform tasks like language 
translation, text generation, or answering questions with accuracy and fluency.
"""

messages = [
    {"role": "user",
     "content": "Explain what a large language model is in two sentences."}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages, 
    temperature=0
)
print(response.choices[0].message.content)

# Differences : 
# Both responses explain the same idea, but the OpenAI response more formal and uses more technical terms like "deep learning", 
# and "neural networks".
# The Ollama response is slightly simpler.

# Advantage of local model:
# It can run without internet and does not require payment after setup. 

# Disadvantage:
# It may be less advanced and provide less detailed answers compared OpenAI.