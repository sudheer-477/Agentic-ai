"""
Day 4: LangChain concepts in one file

Covers everything from today's notes:
  - Runnable interface       (everything below is one)
  - ChatPromptTemplate       (role-tagged prompt with variables)
  - Sequential chain         prompt | llm | parser
  - Transform chain          RunnableLambda
  - Parallel chain           RunnableParallel
  - Branching chain          RunnableBranch

LLM provider: Groq, model: openai/gpt-oss-20b
(llama-3.3-70b-versatile and llama-3.1-8b-instant were deprecated by
Groq on free/dev tiers as of June 17, 2026 -- gpt-oss-20b is current.
Check console.groq.com/docs/models if this changes again.)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_KEY"),
)
parser = StrOutputParser()


# ---------------------------------------------------------------
# 1. SEQUENTIAL CHAIN
# prompt, llm, and parser are each a Runnable. "|" pipes the output
# of one into the input of the next -> together they form one bigger
# Runnable called a RunnableSequence.
# ---------------------------------------------------------------
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Reply with ONLY the translation."),
    ("user", "Translate this to French: {text}"),
])
translate_chain = translate_prompt | llm | parser


# ---------------------------------------------------------------
# 2. TRANSFORM CHAIN
# RunnableLambda wraps a plain Python function so custom logic can
# sit inside a "|" pipe just like any other Runnable.
# ---------------------------------------------------------------
shout = RunnableLambda(lambda text: text.upper() + "!!!")
translate_and_shout_chain = translate_chain | shout


# ---------------------------------------------------------------
# 3. PARALLEL CHAIN
# RunnableParallel runs multiple chains on the SAME input at the
# same time and returns a dict of their results.
# ---------------------------------------------------------------
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the user's text in exactly 5 words."),
    ("user", "{text}"),
])
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reply with exactly one word: positive, negative, or neutral."),
    ("user", "{text}"),
])

analysis_chain = RunnableParallel(
    summary=summary_prompt | llm | parser,
    sentiment=sentiment_prompt | llm | parser,
)


# ---------------------------------------------------------------
# 4. BRANCHING CHAIN
# RunnableBranch checks conditions in order (each is a function that
# returns True/False) and routes to the first matching chain. The
# last argument with no condition is the fallback.
# ---------------------------------------------------------------
question_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question in one sentence."),
    ("user", "{text}"),
])
question_chain = question_prompt | llm | parser
fallback_chain = summary_prompt | llm | parser  # reuse the summarizer as default

router_chain = RunnableBranch(
    (lambda x: "?" in x["text"], question_chain),
    fallback_chain,
)


if __name__ == "__main__":
    print("\n--- 1. Sequential chain ---")
    print(translate_chain.invoke({"text": "good morning"}))

    print("\n--- 2. Transform chain (sequential + RunnableLambda) ---")
    print(translate_and_shout_chain.invoke({"text": "good morning"}))

    print("\n--- 3. Parallel chain ---")
    result = analysis_chain.invoke({
        "text": "I absolutely loved this course, it was so well structured."
    })
    print(result)

    print("\n--- 4. Branching chain ---")
    print(router_chain.invoke({"text": "What is the capital of France?"}))
    print(router_chain.invoke({
        "text": "The weather today is unusually warm for this time of year."
    }))