### Agentic AI Learning Journey
Daily practice files as I learn agentic AI engineering.

## Files

- `myfirst.py` — First Claude API call
- `groq_apicall.py` — Switched to Groq free API  
- `prompting_techniques.py` — Testing 5 prompting techniques
## Day 1 
  api call from claude 

## Day 2
api call from groq free

## Day 3 
1. User prompt — Direct question gave a generic, broad explanation.

2. System prompt — Same question, but with a "strict professor" 
   role, gave a much more formal, technical answer. The system 
   prompt clearly changes tone without changing the question.

3. Zero-shot — Model correctly classified sentiment without 
   any examples, since sentiment classification is a common task 
   it already knows well.

4. Few-shot — Giving 3 examples first made the output format 
   more consistent and predictable compared to zero-shot.

5. Chain of thought — Asking the model to "think step by step" 
   showed its full reasoning process before the final answer, 
   making it easier to verify correctness.

## Day 4.langchaingit 
**Runnable** — the base interface every LangChain component implements
(.invoke(), .batch(), .stream()). It's what lets prompts, models,
and parsers all connect with the same | syntax.
**ChatPromptTemplate** — role-tagged (system / user) prompts with
{variable} placeholders filled in at call time.
**LCEL (LangChain Expression Language)** — composing Runnables with |.
Chain types:

**Sequential** — prompt | llm | parser, output feeds into the next step
**Transform**— RunnableLambda, wraps a plain Python function into the pipe
**Parallel** — RunnableParallel, runs multiple chains on the same input at once
**Branching** — RunnableBranch, routes to a chain based on a condition
LLM provider: Groq (free tier), via langchain-groq.

## Chromadb:
    a database used to store embeddings and retreive using sematic search where these embedings nothing but data stored in vectore dense spase

**chromadb.Client()**-Creates local database in memory

**create_collection()**-Creates a named storage container

**collection.add()**-store data as documents
 
**collection.query()**-search by meaning
