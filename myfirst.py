from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

# Connect to Claude
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Send first message
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello! I am starting my agentic AI engineering journey today. Give me one motivational sentence."
        }
    ]
)

# Print Claude reply
print("Claude says:")
print(message.content[0].text)