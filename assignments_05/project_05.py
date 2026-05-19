# --- Mini-Project — Job Application Helper ---

# Task 1: Setup and System Prompt

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# System prompt: defines assistant behaviour
messages = [
    {"role": "system", 
     "content": (
         "You are a helpful job application coach. You help people improve their resumes, cover letters, "
         "and other job application materials. "
         "Give clear and practical suggestions. Stay focused only on job applications. "
         "Always remind the user to review and edit your suggestions before using them. "
         "You may not know all details about the user's industry, so tell them to use their own judgment."
        )
    }
]

# I used simple and clear language to make the instructions easy for model to follow consistenly.


# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    IMPORTANT:
    Return ONLY raw valid JSON.
    DO NOT include:
    - explanations
    - markdown
    - text before or after JSON

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages, temperature=0)

    try:
        result = json.loads(response)

        # Print side by side
        for item in result:
            print(f"Original: {item['original']}")
            print(f"Improved: {item['improved']}")
            print("-" * 50)

        return result
    
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        return[]
    
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)

# The bullets are weak because they are too vague and do not show clear results.
# The model improved them by using stronger verbs and making them more specific and impactful.


# --- Task 3: Cover Letter Generator ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    
    return get_completion(messages, temperature=0.6)

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

result = generate_cover_letter(job_title, background)
print(result)

# I choose these examples to show different career changes and writing style.
# Few-shot helps the model copy the structure and tone, so the output more consistent.


# --- Task 4: Moderation Check ---

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Your message may violate content guidelines. Please rephrase.")
        return False
    else: 
        return True
    
print(is_safe("Hello, can you help me ?"))
print(is_safe("I want to hurt someone"))


# --- Task 5: The Chatbot Loop ---

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": (
         "You are a helpful job application coach. You help people improve their resumes, cover letters, "
         "and other job application materials. "
         "Give clear and practical suggestions. Stay focused only on job applications. "
         "Always remind the user to review and edit your suggestions before using them. "
         "You may not know all details about the user's industry, so tell them to use their own judgment."
        )
        }
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewrite_bullets(raw_bullets)


        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            print(generate_cover_letter(job_title, background))

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(reply)
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


# --- Task 6: Ethics Reflection ---

# Q1:

# The model can be biased because it learns from internet data that includes different cultures, jobs, and writing styles.
# Because of this, its advice may work better for some people than others.

# Q2: 

# If a job-seeker uses the bot's output without checking it, they may send inccorect or poorly written applications to employers, 
# which can hurt their chances.


# Q3:
#  
# One useful guardrail would be a warning that reminds users to always review and edit AI-generated text before using it for real job applications.  