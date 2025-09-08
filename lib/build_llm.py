from typing import Union
import os
from enum import Enum
import dotenv
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class Stage(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


MODEL_STAGE_MAP = {
    "gemma3:4b": Stage.LOCAL,
    "phi3:mini": Stage.LOCAL,   # ✅ Phi-3 Mini
    "gemini-2.5-flash-lite": Stage.CLOUD,
    "gemini-2.5-pro": Stage.CLOUD,
    "gemini-2.5-flash": Stage.CLOUD,
    "gpt-4o": Stage.CLOUD,      # ✅ OpenAI GPT
    "gpt-4o-mini": Stage.CLOUD, # ✅ Lighter OpenAI GPT
    "qwen2.5:3b": Stage.LOCAL,
    "deepseek-r1:8b": Stage.LOCAL,
    "llama3.1": Stage.LOCAL,
    "qwen3:8b": Stage.LOCAL,
    "qwen3:4b": Stage.LOCAL
}


def build_llm(model_name: str) -> Union[OllamaLLM, ChatGoogleGenerativeAI, ChatOpenAI]:
    stage = MODEL_STAGE_MAP.get(model_name)
    if stage is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Please add it to MODEL_STAGE_MAP."
        )

    print(f"Building {stage.value} LLM for model: {model_name}")

    if stage == Stage.LOCAL:
        # Local models via Ollama
        llm = OllamaLLM(model=model_name, temperature=0.2)

    else:
        # Cloud models: OpenAI or Google GenAI
        if model_name.startswith("gpt-"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI models")
            llm = ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0)

        elif model_name.startswith("gemini-"):
            api_key = dotenv.get_key(".env", "GOOGLE_API_KEY")

            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required for Google models")
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        
        else:
            raise ValueError(f"Cloud model '{model_name}' not supported")

    print(f"Successfully built {stage.value} LLM: {model_name}")
    return llm


# Example usage:
if __name__ == "__main__":
    llm = build_llm("phi3:mini")  # or "gpt-4o" / "gemini-2.5-flash-lite"
    response = llm.invoke("Return a JSON object with customer_id=7")
    print(f"Response: {response}")
