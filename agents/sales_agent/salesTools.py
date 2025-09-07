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


DB_PATH = "db/erp.db"  # path to the SQLite database file


class ReadSQLIntent(str, Enum):
    GET_CUSTOMER_BY_ID = "get_customer_by_id"
    GET_CUSTOMER_BY_NAME = "get_customer_by_name"
    GET_LEAD_BY_ID = "get_lead_by_id"
    GET_ALL_ORDERS = "get_all_orders"


@tool("sales_sql_read")
def read_sql(input_str: str) -> str:
    """Parses input string like 'intent: get_customer_by_id, value: 1' and executes SQL queries."""

    # Parse intent and value from string
    intent = parse_value(input_str, "intent")
    print(f"Parsed intent: {intent}")
    value = parse_value(input_str, "value")
    print(f"Parsed value: {value}")
    if not intent or not value:
        return "Invalid input format. Expected 'intent: <intent>, value: <value>'."

    # Convert value to int if it's a digit
    if value.isdigit():
        value = int(value)

    # Validate intent
    try:
        intent_enum = ReadSQLIntent(intent)
    except ValueError:
        return f"Invalid intent: {intent}. Valid intents are: {[e.value for e in ReadSQLIntent]}"

    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Map intent to SQL
    if intent_enum == ReadSQLIntent.GET_CUSTOMER_BY_ID:
        cursor.execute("SELECT * FROM customers WHERE id = ?", (value,))
    elif intent_enum == ReadSQLIntent.GET_CUSTOMER_BY_NAME:
        cursor.execute("SELECT * FROM customers WHERE name LIKE ?", (f"%{value}%",))
    elif intent_enum == ReadSQLIntent.GET_LEAD_BY_ID:
        cursor.execute("SELECT * FROM leads WHERE id = ?", (value,))
    elif intent_enum == ReadSQLIntent.GET_ALL_ORDERS:
        cursor.execute("SELECT * FROM orders")

    rows = cursor.fetchall()

    if cursor.description:
        col_names = [desc[0] for desc in cursor.description]
        results = [dict(zip(col_names, row)) for row in rows]
        conn.close()
        return str(results) if results else "No results found."
    else:
        conn.close()
        return "No results found."
