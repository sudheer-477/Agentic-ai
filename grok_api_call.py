from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

message = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello! I am starting my agentic AI engineering journey today. Give me one motivational sentence."
        }
    ]
)

print("Groq says:")
print(message.choices[0].message.content)