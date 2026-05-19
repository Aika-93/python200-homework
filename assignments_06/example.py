from dotenv import load_dotenv
import os
from pathlib import Path
from pypdf import PdfReader
from openai import OpenAI

# Load environment and OpenAI key
if load_dotenv():
    print("✅ Successfully loaded API key")
else:
    print("⚠️ Failed to load API key from .env file")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text.strip())
    return "\n".join(text)

def simple_keyword_retrieval(query, documents, verbose=True):
    """
    Keyword retrieval using token overlap scoring.
    - Removes stopwords and punctuation for cleaner matching.
    - Returns the single best-matching document.
    - `documents`: dictionary with the document names as keys and the text as values, extracted using the `extract_text_from_pdf` function.
    """
    import string

    stopwords = [
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    ]

    # Translator to remove punctuation (so "Solar?" -> "Solar")
    translator = str.maketrans("", "", string.punctuation)

    # Tokenize query: lowercase, remove punctuation and stopwords
    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        # Tokenize document: lowercase, remove punctuation and stopwords
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }

        # Compute simple overlap score
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))

        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    # Sort by overlap score (descending)
    scores.sort(reverse=True)

    # Pick the single best match (if score > 0)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
def ask_llm(query, context, with_rag):
    """Ask the LLM using retrieved context."""
    if with_rag:
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question: {query}\nAnswer:"
        )
    else:
        prompt = (
            f"Question: {query}\nAnswer:"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

use_rag = True # Set to True to use keyword RAG, False otherwise
# Load all PDF text into a dictionary
pdf_dir = Path("brightleaf_pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))
if not pdf_files:
    raise FileNotFoundError("No PDFs found in the pdfs directory.")

docs = {f.name: extract_text_from_pdf(f) for f in pdf_files}
print(f"Loaded {len(docs)} BrightLeaf PDF(s).")

print("\nType 'quit' to exit.\n")
while True:
    query = input("Enter your query: ").strip()
    if query.lower() in {"quit", "exit"}:
        print("Goodbye.")
        break

    results = simple_keyword_retrieval(query, docs, verbose=True)
    context = results[0][1]
    answer = ask_llm(query, context, use_rag)

    print("\n--- Response ---")
    print(answer)
    print("\n" + "=" * 60 + "\n")
    
