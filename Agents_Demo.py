import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool

# ── Load keys from .env file ──────────────────────────────
load_dotenv()

# ── Tool 1: Calculator ────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Solves math. Pass ONLY numbers and operators.
    Good: '8500 * 50'
    Bad:  'gold_price * 50'
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# ── Tool 2: Web Search ────────────────────────────────────
search = TavilySearch(max_results=2)

# ── Agent ─────────────────────────────────────────────────
agent = create_agent(
    model=ChatGroq(model="qwen/qwen3.6-27b"),
    tools=[calculator, search],
    system_prompt=(
        "You have two tools: search and calculator.\n"
        "Step 1: Use search to find the number.\n"
        "Step 2: Use calculator with that actual number to do math.\n"
        "Never pass words into calculator, only numbers."
    ),
)

# ── Question ──────────────────────────────────────────────
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Search the current price of gold per gram in INR."
                   " Then calculate the cost of 50 grams."
    }]
})

# ── Print Steps ───────────────────────────────────────────
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

print("\n" + "="*50)
for msg in result["messages"]:
    if isinstance(msg, HumanMessage):
        print(f"\nQUESTION:\n  {msg.content}")
    elif isinstance(msg, AIMessage) and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"\nACT → {tc['name']}")
            print(f"  args: {tc['args']}")
    elif isinstance(msg, ToolMessage):
        print(f"\nOBSERVE:")
        print(f"  {str(msg.content)[:300]}")
    elif isinstance(msg, AIMessage):
        print(f"\nFINAL ANSWER:\n  {msg.content}")
print("\n" + "="*50)