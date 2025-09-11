# agents/llm_setup.py

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")  # Reads the key from .env
)

# Test it
if __name__ == "__main__":
    print("🧪 Testing OpenAI connection...")
    try:
        response = llm.invoke("Explain what a purchase order is in one sentence.")
        print("🤖:", response.content)
    except Exception as e:
        print("❌ Error:", str(e))