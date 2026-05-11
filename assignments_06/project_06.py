# ------- Part 2: Mini-Project: Groundwork Coffee Co. Q&A Assistant ---------------

# ---------- Step 1: Setup ----------

from dotenv import load_dotenv
from openai import OpenAI
import os 
from pathlib import Path

# Load API key from .env
# Print a confirmation message

if load_dotenv():
    print("Successfully loaded API key")
else:
    print("Failed to load API key from .env file")

# Create OpenAi client using API key
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# Verify that the document folder exists 
# Stop the program early if the folder is missing
docs_dir = Path("groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"



# ---------- Step 2: Load the Documents ----------

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load all documents from the groundwork_docs folder
docs = SimpleDirectoryReader("groundwork_docs").load_data()

# Print total number of loaded document
print(f"\nLoaded {len(docs)} documents.\n")

# Print the file name for each document
for doc in docs:
    print("File name:", doc.metadata["file_name"])



# ---------- Step 3: Build the Index and Query Engine ----------

# Build vector index from documents
index = VectorStoreIndex.from_documents(docs)

# Create query engine (top 3 results)
query_engine = index.as_query_engine(similarity_top_k=3)

# Confirmation message 
print("\nIndex built successfully. Ready to answer questions.")


# ---------- Step 4: Query the Assistant ----------

# List of questions to query the assistant 
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

# Loop through each question
for q in questions:
    print("\n===================")

    # Print the question
    print("Question: ", q)

    # Get answer from query engine
    response = query_engine.query(q)

    # Print model answer 
    print("\nAnswer:")
    print(response)

    # Print retrieved source chunks  
    print("\nSource nodes:")

    # Loop through top retrieved documents
    for node_with_score in response.source_nodes:

        # Print document name
        print(f"Document: {node_with_score.node.metadata.get('file_name')}")

        # Print similarity score
        print(f"Similarity Score: {node_with_score.score:.4f}")

        # Print first 200 characters of the retrieved text 
        print(f"Text Snippet: {node_with_score.get_content()[:200]}...")

        # Separator for readability
        print("-" * 30)

# The responses are mostly confident and accurate.
# The model answers questions directly without hedging.
# Most retrieved chunks are relevant to the questions, especially for hours, menu and story.
# I did not see any unexpected or unrelated results.



# ---------- Step 5: Find a Failure ----------

question = "Are there any allergens in your menu, such as walnuts or nuts?"

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nQuestion: ", question)

response = query_engine.query(question)

print("\nAnswer:")
print(response)

print("\nSource nodes:")

# Loop through top retrieved documents
for node_with_score in response.source_nodes:

    # Print document name
    print(f"Document: {node_with_score.node.metadata.get('file_name')}")

    # Print similarity score
    print(f"Similarity Score: {node_with_score.score:.4f}")

    # Print first 200 characters of the retrieved text 
    print(f"Text Snippet: {node_with_score.get_content()[:200]}...")

    # Separator for readability
    print("-" * 30)

"""
- What you asked and why you expected it to be hard:

    I asked if there are allergens in the menu, like walnuts or nuts.
    I think this is hard because the documents do not clearly say anything about allergens. 

- What went wrong — wrong retrieval, missing information, the model guessed anyway?

    The model did not find clear information in the documents.
    It still gave an answer saying there are no allergens, but this is not clearly written in the data. So it guessed the answer.

- When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? 
- What does this suggest about trusting AI-generated responses?

    The model sounded confident. It did not say "maybe" or "not sure", even though the information was not clearly in the documents.
    This means the model can still hallucinate answers when the correct information is not found.
    This shows that we cannot fully trust AI-generated answers in RAG systems if we do not check the retrieved sources.

- What you would change about the system to improve it:

    I would add clearer information in the documents (for example, a section about allergens).
    Also, the system should say when it cannot find the answer instead of guessing.

"""



# ---------- Step 6: Reflection ----------

"""
1. The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. 
How many lines did the equivalent LlamaIndex implementation take in your project? 
What does that tell you about the value of using a framework?

    The LlamaIndex version uses much less code compared to building RAG manually.
    In manual RAG, we had to write many lines for chunking, embeddings, and retrieval.
    With LlamaIndex, it was only a few lines of code.
    This shows that frameworks save time and make development much easier and faster.

2. You have now built a system that answers questions from real documents. 
Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.
    
    A real use case for this system could be in a large company with many departments.
    It can help employees quickly find information from internal documents, policies, or reports.
    This makes it easier to understand what different teams are working on.

3. What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

    One failure mode of RAG is hallucination.
    Even if retrieval works correctly, the model can still generate information that is not in the documents.
    This cannot be fully prevented, so we still need to check the sources carefully.
    
"""