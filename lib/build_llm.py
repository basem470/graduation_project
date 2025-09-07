from typing import Union
import os
from enum import Enum
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI


class Stage(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


MODEL_STAGE_MAP = {
    "gemma3:4b": Stage.LOCAL,
    "gemini-2.5-flash-lite": Stage.CLOUD,
    "phi3:4b": Stage.LOCAL,
    "gemini-2.5-pro": Stage.CLOUD,
}


def build_llm(model_name: str) -> Union[OllamaLLM, ChatGoogleGenerativeAI]:
    stage = MODEL_STAGE_MAP.get(model_name)
    if stage is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Please add it to MODEL_STAGE_MAP."
        )

    print(f"Building {stage.value} LLM for model: {model_name}")

    if stage == Stage.LOCAL:
        llm = OllamaLLM(model=model_name)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for cloud models")
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    print(f"Successfully built {stage.value} LLM: {model_name}")
    return llm


# Example usage:
if __name__ == "__main__":
    llm = build_llm("gemini-2.5-flash-lite")
    response = llm.invoke("Hello, how are you?")
    print(f"Response: {response}")
