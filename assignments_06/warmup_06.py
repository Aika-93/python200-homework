from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")



# --- RAG Concepts ---

# Concepts Q1

"""
Scenario A:
RAG is the best approach because the assistant needs to search through many internal PDFs that are updated often.
RAG allows the model to retrieve the latest information without retraining.

Scenario B:
Fine-tuning is the best approach because the company wants a very specific writing style that is not common online.
The 3000 examples can help train the model to consistently match the brand voice.

Scenario C:
Prompt engineering is the best approach because the analyst only needs to work with one short report.
A well-written prompt is enough, and no training oe retrieval system is necessary.

"""


# Concepts Q2

"""
A confidently wrong answer is more harmful than uncertainty because people tend to trust answers that sound certain,
even if they are incorrect. This can lead to wrong decisions based on false information.

For example, if someone asks about medication dosage and the model gives a confident but incorrect answer, 
the person might follow it and cause harm to their health.

The tone matters because confident language increases trust, making it harder for users to recognize when the information might be wrong.

"""


# Concepts Q3

steps = [
    "Receive the user's query", # The system takes the user's question as input.
    "Extract text from source documents", # The raw text is pulled from documents.
    "Split text into chunks", # The documents are divided into smaller, managable pieces.
    "Convert text chunks into embeddings", # Each text chunk is converted into a numerical vector representation.
    "Embed the user's query", # The user's question is also converted into an embedding vector.
    "Retrieve the most relevant chunks", # The system finds chunks most similar to the query using vector similarity.
    "Inject retrieved chunks into the prompt", # The relevant chunks are added to the prompt as context.
    "Generate a response from the LLM", # The model produces the final answer using the prompt and retrieved context.
]



# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]



# Keyword RAG Q1

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0])

# The selected document was "loyalty.txt".
# It was chosen because the keyword-based method matched the word "your" from the query.
# Even though this is not very meaningful, it had the highest overlap score among the documents.
# This shows a limitation of keyword search: it can pick irrelevant documents based on common words rather than real meaning.



# Keyword RAG Q2

query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0])

# No document was selected because keyword RAG did not find any overlapping words in the query and documents.
# Keyword RAG did not work well here because it only matches exact words and cannot understand meaning.
# The best matching document would have been "menu.txt" because the question is about drinks without caffeine.



# Keyword RAG Q3

# Prediction: no document will be selected.
# Reason: None of the documents contain the exact keywords "sign", "up", or "rewards", 
# so keyword-based retrieval cannot find any overlap.
# Keyword RAG fails here because it relies only on exact word matching and does not understand meaning. 

query = "How do I sign up for rewards?"

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0])

# Yes my prediction was correct.


# --- Semantic RAG Concepts ---


# Semantic Q1

# 1. A vector embedding is a numerical representation of text that captures its meaning.
# It converts words into numbers so that a model can compare texts mathematically.

# 2. The chunk with cosine similarity 0.85 is more relevant because it is closer to 1.
# A higher cosine similarity means the texts are more similar in meaning, while lower values mean less similarity.

# 3. Semantic search can find relevant chunks even without exact matching words because it compares meaning instead of keywords.
# It uses embeddings to understand relationships between words and concepts, not just exact text matches.



# Semantic Q2

"""
| Feature                    | Keyword RAG                       | Semantic RAG |
|----------------------------|-----------------------------------|--------------|
| What is compared?          | Exact word overlap                | Meaning             |
| What is retrieved?         | Full document                     | Relevant text chunks|
| Can it handle synonyms?    | No                                | Yes                 |
| Storage format             | Plain text dictionary             | Vector embeddings   |
| Relevance score            | Number of overlapping keywords    | Cosine similarity   |
"""


# --- LlamaIndex ---

# LlamaIndex Q1

from llama_index.core import VectorStoreIndex, Document
from pypdf import PdfReader 
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text =[]

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    
    return "\n".join(text)

pdf_dir = Path("brightleaf_pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

documents = []

for pdf_file in pdf_files:
    text = extract_text_from_pdf(pdf_file)
    documents.append(Document(text=text))

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print("\n===================")
    print("Question: ", q)

    response = query_engine.query(q)

    print("\nAnswer:")
    print(response)

    print("\nSource nodes:")

    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
        print("-" * 30)

# Query 1: What employee benefits does Brightleaf offer?
# 1. The retrieved chunks are relevant because they include sections about employee benefits,
# such as health insurance, retirement plans, and professional development programs.
#
# 2. The model's response sounds confident and specific. It directly lists benefits without hedging language like "based on the context",
# or "I'm not sure".
# 
# 3. Nothing unexpected was retrieved.


# Query 2: What are BrightLeaf's security policies?
# 1. The retrieved chunks are relevant because they contain information about network security,
# authentication, encryption, and compliance policies.
# 
# 2. The model's responce is confident and detailed. It clearly lists specific security measures without ubcertainty or hedging phrases.
# 
# 3. Nothing unexpected was retrieved.


# LlamaIndex Q2

question = "What employee benefits does BrightLeaf offer?"

# top_k = 1 
print("\n=======================")
print("RUN: similarity_top_k = 1")

query_engine = index.as_query_engine(similarity_top_k=1)

response = query_engine.query(question)

print("\nAnswer:")
print(response)

print("\nSource nodes:")
for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
    print("-" * 30)

# top_k = 5
print("\n=======================")
print("RUN: similarity_top_k = 5")
query_engine = index.as_query_engine(similarity_top_k=5)

response = query_engine.query(question)

print("\nAnswer:")
print(response)

print("\nSource nodes:")
for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
    print("-" * 30)

# When using similarity_top_k = 1, the response is more focused and based on a single most relevant chunk.
# The answer is accurate but slightly limited on detail.

# When using similarity_top_k = 5, more chunks are retrieved , which makes the answer more detailed.
# However, some additional chunks are only loosely related to the question.

# This shows that increasing the number of retrieved chunks does not always improve quality.
# It can help completness, but may also introduce noise from less relevant information.


# LlamaIndex Q3

question = "What risks could Brightleaf face in the future?"

print("\n===================")
print("Question: ", question)

query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query(question)

print("\nAnswer:")
print(response)

print("\nSource nodes:")
for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.get_content()[:150]}...")
    print("-" * 30)

# Stress test query: "What risks could Brightleaf face in the future?"

# I expected this question to be hard because it is very general and not directly answered in one specific part of the document.
# The retrieved chunks were mostly general company information like overview and introduction, not a specific "risk" section.
# The model still give a good answer by using the context.

# This shows that semantic search can work wit questions, but it does not always retrieve very exact or specific information.

# To improve this system, I would try better chunking or labeling sections more clearly.


# LlamaIndex Q4

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

query_engine = index.as_query_engine(similarity_top_k=3)

# Good Query
q1 = "What employee benefits does BrightLeaf offer?"
response1 = query_engine.query(q1)

print("\n GOOD QUERY")

f1 = faithfulness_evaluator.evaluate_response(query=q1, response=response1)
print("Faithfulness Evaluation:" + str(f1.score))

r1 = relevancy_evaluator.evaluate_response(query=q1, response=response1)
print("Relevancy Result:" + str(r1.score))

# Bad Query

q2 = "What is Brightleaf's favorite food?"
response2 = query_engine.query(q2)

print("\n BAD QUERY")

f2 = faithfulness_evaluator.evaluate_response(query=q2, response=response2)
print("Faithfulness Evaluation:" + str(f2.score))

r2 = relevancy_evaluator.evaluate_response(query=q2, response=response2)
print("Relevancy Result:" + str(r2.score))

# A faithfulness score of 1.0 means the answer is fully supported by the retrieved context.
# A score of 0.0 would mean the answer is not supported by the context at all (hallicunation)

# A relevancy score measures how well the answer matches the user's question.
# Faithfulness checks if the answer is grounded in the documents, while relevancy checks if it actually answers the question.

# The scores were higher for the employee benefits question because the information is clearly present in the document.
# The scores were lower for the "favorite food" question because this information is not in the dataset, 
# so the model had to guess or return weak answers.

# The LLM-as-a-judge" approach means we use another LLM(like GPT-4o-mini) to evaluate the quality of the answer instead of using
# simple exact-match metric.
# This is used in RAG because answers are often free-form text, so exact accuracy is not enough.