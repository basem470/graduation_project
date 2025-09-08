import json
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
from langchain.tools import StructuredTool
from typing import Union
import datetime
from typing import Optional
from typing import Union
from datetime import date, timedelta
from typing import Optional, Tuple
import logging
import os

import requests

DB_PATH = "db/erp.db"  # path to the SQLite database file

API_BASE_URL = "http://localhost:8000" 

# =========================
# CUSTOMER TOOLS
# =========================


@tool
def find_customer_by_id_tool(customer_id: Union[int, str]) -> str:
    """
    Find customer by ID.
    Accepts input like find_customer_by_id(1)
    find_customer_by_id(2)

    Args:
        customer_id: The ID of the customer to find (can be int or string)
    
    Returns:
        Customer information in natural language or error message
    """
    
    # Handle both int and string inputs
    if isinstance(customer_id, str):
        if not customer_id or not customer_id.isdigit():
            return "Error: customer_id must be a valid number"
        customer_id = int(customer_id)
    elif isinstance(customer_id, int):
        if customer_id <= 0:
            return "Error: customer_id must be a positive number"
    else:
        return "Error: customer_id must be a number"
    
    # Your database query logic here
    try:
        answer = sql_query_executor(f"SELECT * FROM customers WHERE id = {customer_id}")
        return answer
        
    except Exception as e:
        return f"Error finding customer: {e}"

@tool
def find_customer_by_name_tool(customer_name: str) -> str:
    """
    Find customer by name.
    Accepts input like find_customer_by_name("Acme Corp") or find_customer_by_name("John Smith")
    
    Args:
        customer_name: The name of the customer to find.
    
    Returns:
        Customer information in natural language or an error message.
    """
    
    if not isinstance(customer_name, str) or not customer_name:
        return "Error: customer_name must be a non-empty string."
    
    # Use a LIKE query for partial or case-insensitive matching.
    # The '%' wildcard allows for a flexible search.
    sql_query = f"SELECT * FROM customers WHERE name LIKE '%{customer_name}%'"
    
    # Your database query logic here
    try:
        # Assuming sql_query_executor is a function that runs the query
        answer = sql_query_executor(sql_query)
        
        if not answer:
            return f"No customers found with a name like '{customer_name}'."
        
        return answer
        
    except Exception as e:
        return f"Error finding customer: {e}"
    
@tool
def find_customer_by_email_tool(customer_email: str) -> str:
    """
    Finds customer information by their email address.
    Args:
        customer_email: The email of the customer to find.
    Returns:
        Customer information or an error message.
    """
    if not isinstance(customer_email, str) or not customer_email:
        return "Error: customer_email must be a non-empty string."
    
    sql_query = f"SELECT * FROM customers WHERE email = '{customer_email}'"
    
    try:
        answer = sql_query_executor(sql_query)
        if not answer:
            return f"No customer found with the email '{customer_email}'."
        return answer
    except Exception as e:
        return f"Error finding customer: {e}"
    


@tool
def find_customer_by_phone_tool(customer_phone: str) -> str:
    """
    Finds customer information by their phone number directly from the database.
    Args:
        customer_phone: The phone number of the customer to find.
    Returns:
        Customer information or an error message.
    """
    if not isinstance(customer_phone, str) or not customer_phone.strip():
        return "Error: customer_phone must be a non-empty string."
    
    # sanitize input
    customer_phone = customer_phone.replace("'", "").replace('"', "").replace(" ", "")
    
    sql_query = f"SELECT * FROM customers WHERE phone LIKE '%{customer_phone}%' OR phone = '{customer_phone} LIMIT 3'"
    
    try:
        answer = sql_query_executor(sql_query)
        if not answer:
            return f"No customer found with the phone number '{customer_phone}'."
        return answer
    except Exception as e:
        return f"Error finding customer: {e}"

#today's date
@tool
def get_todays_date():
    """
    Returns the current date in the format "YYYY-MM-DD".
    """
    today = date.today()
    return today.strftime("%Y-%m-%d")

class CustomerDateInput(BaseModel):
    """Input schema for finding customers by date created."""
    start_date: str = Field(description="The start date of the date range in the format YYYY-MM-DD")
    end_date: str = Field(description="The end date of the date range in the format YYYY-MM-DD")





def query_customers_by_date_range(start_date: str, end_date: str):
    sql_query = f"SELECT * FROM customers WHERE created_at BETWEEN '{start_date} 00:00:00' AND '{end_date} 23:59:59'"
    return sql_query_executor(sql_query)


@tool
def find_customers_by_date_created(input: str) -> str:
    """
    Finds customers created within a specific date range.
    Use get_todays_date() tool to get today's date if you were asked for relative dates, like last month, week, year.
    input should be json '{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}'
    Returns:
        A list of customer information or an error message.
        Answer should be natural language, mentioning start_date and end_date
    """
 
    try:
        if isinstance(input, str):
            parsed_input = json.loads(input) 
            start_date = parsed_input['start_date']
            end_date = parsed_input['end_date']
        elif isinstance(input, dict):
            start_date = input['start_date']
            end_date = input['end_date']
        else:
            return "Error: Invalid input format"

        customers = query_customers_by_date_range(start_date, end_date)
        
        if not customers:
            return f"No customers found created between {start_date} and {end_date}"
        
        return f"Found {len(customers)} customers created between {start_date} and {end_date}: {customers}"
        
    except Exception as e:
        return f"Error finding customers: {str(e)}"
    

# =========================
# ORDERS TOOLS
# =========================


@tool
def find_order_by_order_id(order_id: str) -> str:
    """
    Finds order information by its ID.
    Args:
        order_id: The ID of the order to find.
    Returns:
        Order information or an error message.
        Respond with the order details in natural language
    """
    if not isinstance(order_id, str) or not order_id:
        return "Error: order_id must be a non-empty string."
    
    sql_query = f"SELECT o.id as order_id, name, email, o.total, o.status, o.created_at as order_date from customers c join orders o on c.id=o.customer_id where o.id= {order_id}"

    
    try:
        answer = sql_query_executor(sql_query)
        if not answer:
            return f"No order found with the ID '{order_id}'."
        return answer
    except Exception as e:
        return f"Error finding order: {e}"


def query_orders_by_date_range(start_date: str, end_date: str):
    sql_query = f"""
    SELECT o.id as order_id, name as customer_name, o.total, o.status, o.created_at as order_date
    from customers c join orders o on c.id=o.customer_id 
    where
    o.created_at between '{start_date} 00:00:00' AND '{end_date} 23:59:59' 
    order by o.created_at DESC 
    limit 10"""
    return sql_query_executor(sql_query)


@tool
def find_orders_by_date(input: str) -> str:
    """
    Finds orders created within a specific date range.
    Use get_todays_date() tool to get today's date if you were asked for relative dates, like last month, week, year.
    input should be json '{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}'
    Returns:
        A list of order information or an error message.
        Answer should be natural language, mentioning start_date and end_date
    """
    
    try:
        if isinstance(input, str):
            parsed_input = json.loads(input) 
            start_date = parsed_input['start_date']
            end_date = parsed_input['end_date']
        elif isinstance(input, dict):
            start_date = input['start_date']
            end_date = input['end_date']
        else:
            return "Error: Invalid input format"

        orders = query_orders_by_date_range(start_date, end_date)
        
        if not orders:
            return f"No orders found created between {start_date} and {end_date}"
        
        return f"Found {len(orders)} orders created between {start_date} and {end_date}: {orders}"
        
    except Exception as e:
        return f"Error finding orders: {str(e)}"

@tool
def find_order_by_customer_id(customer_id: str) -> str:
    """
    Finds order information by its customer ID.
    Args:
        customer_id: The ID of the customer to find.
    Returns:
        Order information or an error message.
    """
    if not isinstance(customer_id, str) or not customer_id:
        return "Error: customer_id must be a non-empty string."
    
    sql_query = f"SELECT o.id as order_id, name, email, o.total, o.status, o.created_at as order_date from customers c join orders o on c.id=o.customer_id where c.id= {customer_id}"

    
    try:
        answer = sql_query_executor(sql_query)
        if not answer:
            return f"No order found with the ID '{customer_id}'."
        return answer
    except Exception as e:
        return f"Error finding order: {e}"

    

# =========================
# LEADS TOOLS
# =========================

@tool
def find_leads_by_name(name: str) -> str:
    """
    Finds leads by their name.
    Args:
        name: The name of the lead to find.
    Returns:
        Lead information or an error message.
    """
    if not isinstance(name, str) or not name:
        return "Error: name must be a non-empty string."
    
    sql_query = f"SELECT * FROM leads WHERE name = '{name}'"
    
    try:
        answer = sql_query_executor(sql_query)
        if not answer:
            return f"No lead found with the name '{name}'."
        return answer
    except Exception as e:
        return f"Error finding lead: {e}"





tools = [
    # Customers_read
    find_customer_by_id_tool,
    find_customer_by_name_tool,
    find_customer_by_email_tool,
    find_customer_by_phone_tool,
    get_todays_date,
    find_customers_by_date_created,
    # Orders_read
    find_order_by_order_id,
    find_order_by_customer_id,
    find_orders_by_date
     ]