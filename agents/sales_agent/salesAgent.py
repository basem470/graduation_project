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
from agents.sales_agent.salesTools import read_sql


llm = build_llm("gemma3:4b")

prompt = hub.pull("hwchase17/react")


read_sql.invoke("intent: get_customer_by_id, value: 1")

tools = [read_sql]
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

sales_agent_exec = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
