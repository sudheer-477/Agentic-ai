from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Helper function — reusable for all 5 techniques
def ask(messages, label):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=messages
    )
    print(f"\n{'='*50}")
    print(f"{label}")
    print('='*50)
    print(response.choices[0].message.content)


# ============================================
# 1. USER PROMPT
# Just the question, no instructions, no role
# ============================================
ask(
    messages=[
        {"role": "user", "content": "Explain what an AI agent is."}
    ],
    label="1. USER PROMPT (basic)"
)


# ============================================
# 2. SYSTEM PROMPT
# Sets the AI's role/personality BEFORE the user talks
# ============================================
ask(
    messages=[
        {"role": "system", "content": "You are a strict computer science professor. Explain concepts in a formal, academic tone with precise terminology."},
        {"role": "user", "content": "Explain what an AI agent is."}
    ],
    label="2. SYSTEM PROMPT (sets behavior/role)"
)


# ============================================
# 3. ZERO-SHOT PROMPTING
# No examples given — model answers from its own knowledge
# ============================================
ask(
    messages=[
        {"role": "user", "content": "Classify this review as Positive, Negative, or Neutral: 'The product broke after one day, very disappointed.'"}
    ],
    label="3. ZERO-SHOT (no examples given)"
)


# ============================================
# 4. FEW-SHOT PROMPTING
# Give 2-3 examples first, THEN ask the real question
# Model learns the pattern from examples
# ============================================
ask(
    messages=[
        {"role": "user", "content": """Classify the sentiment as Positive, Negative, or Neutral.

Review: "Amazing quality, fast delivery!"
Sentiment: Positive

Review: "It's okay, nothing special."
Sentiment: Neutral

Review: "Worst purchase ever, total waste of money."
Sentiment: Negative

Review: "The product broke after one day, very disappointed."
Sentiment:"""}
    ],
    label="4. FEW-SHOT (3 examples given before the real question)"
)


# ============================================
# 5. CHAIN OF THOUGHT
# Ask the model to think step by step before answering
# Improves accuracy on reasoning/math problems
# ============================================
ask(
    messages=[
        {"role": "user", "content": """A store had 120 apples. They sold 45 in the morning and 38 in the afternoon. 
Then they received a new delivery of 60 apples.
How many apples does the store have now?

Think through this step by step before giving the final answer."""}
    ],
    label="5. CHAIN OF THOUGHT (step-by-step reasoning)"
)