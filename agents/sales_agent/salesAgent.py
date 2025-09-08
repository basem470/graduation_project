from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re
from lib.build_llm import build_llm
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from lib.sql_query_executor import sql_query_executor
from langchain import hub
import sqlite3
from enum import Enum
from lib.parse_value import parse_value
from agents.sales_agent.salesTools import tools
from langchain.agents import create_tool_calling_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

llm = build_llm("gemini-2.5-flash")
# llm = build_llm("qwen3:4b")
# llm = build_llm("qwen3:8b")
# llm = build_llm("qwen2.5:3b")
# llm = build_llm("llama3.1")
# llm = build_llm("deepseek-r1:8b")



load_dotenv()  


# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.2,
#     openai_api_key=os.getenv("OPENAI_API_KEY")  
# )

prompt = hub.pull("hwchase17/react")



agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

sales_agent_exec = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
