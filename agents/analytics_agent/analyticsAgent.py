import os
import sys
import sqlite3
import json

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Example: Pulling a specific prompt from the hub


# Get the path to the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two directories to reach the project root
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add the project root to the Python path
sys.path.insert(0, project_root)

from lib.build_llm import build_llm
from agents.analytics_agent.gloassary_rag import retrieve_and_generate


# Get the path to the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two directories to reach the project root
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add the project root to the Python path
sys.path.insert(0, project_root)

# Import the existing RAG tool function
from agents.analytics_agent.gloassary_rag import retrieve_and_generate

# Load environment variables
load_dotenv()
DB_PATH = os.getenv("DB_PATH")
if not DB_PATH:
    raise ValueError("DB_PATH is not set in the environment or .env file")



def summarize_schema(db: SQLDatabase, max_tables=None):
    """
    Summarize the schema string returned by db.get_table_info().
    Optionally limit the number of tables shown.
    """
    tables_info = db.get_table_info()  # this is a string
    lines = tables_info.strip().split("\n")
    if max_tables:
        lines = lines[:max_tables]
    return "\n".join(lines)


# --- Initialize Database and LLM ---
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
schema_summary = summarize_schema(db)
llm = build_llm("gemini-2.5-flash")

# --- Define the Tools for the Agent ---

def text_to_sql_tool(question: str):
    """
    Generates and executes a SQL query based on a natural language question.
    """
    sql_prompt_template = ChatPromptTemplate.from_template(
        f"You have access to the following database schema:\n{schema_summary}\n\n"
        "You are an SQL expert. Given the following question, generate a valid SQL query.\n"
        "Your generated queries should be directly runnable, no comments, no explanation, and always start with 'SELECT'.\n"
        "Question: {question}\nSQL Query:"
    )
    prompt = sql_prompt_template.format(question=question)
    ai_response = llm.invoke(prompt)

    # Extract the SQL query, handling potential markdown formatting
    # This part was updated to correctly extract the string from the response object
    sql_query = str(ai_response.content).replace("```sql", "").replace("```", "").strip()
    print(f"\nGenerated SQL query: {sql_query}")

    # Connect to the database and execute the query
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
    except Exception as e:
        results = f"Error executing SQL: {e}"
    finally:
        conn.close()

    return results

def analytics_and_reporting_tool(input_data):
    """
    Generates a JSON specification for a chart or report.
    Can accept either a question string or structured report data.
    """
    try:
        # Check if input is already structured data (dict or JSON string of dict)
        if isinstance(input_data, dict):
            # Input is already a structured report, just validate and return
            required_fields = ['report_name', 'report_type', 'data']
            if all(field in input_data for field in required_fields):
                return input_data
            else:
                return {"error": "Missing required fields: report_name, report_type, data"}
        
        # Try to parse as JSON first (in case it's a JSON string)
        try:
            parsed_data = json.loads(input_data)
            if isinstance(parsed_data, dict):
                return parsed_data
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Treat as a question string and generate chart specification
        question = str(input_data)
        
        json_prompt_template = ChatPromptTemplate.from_template(
            "You have access to the following database schema:\n{schema_summary}\n\n"
            "You are an expert at creating chart specifications. Given the following question, "
            "generate a valid JSON object that describes a chart. The JSON object must have "
            "a 'title', a 'type' (e.g., 'bar', 'pie', 'line'), and a 'data' array containing objects with 'label' and 'value'.\n"
            "Example output:\n{{\n  \"title\": \"Example Chart\",\n  \"type\": \"bar\",\n  \"data\": [{{\"label\": \"A\", \"value\": 10}}]\n}}\n"
            "Question: {question}\n"
            "Generate only valid JSON, no extra text:"
        )
        
        prompt = json_prompt_template.format(question=question, schema_summary=schema_summary)
        ai_response = llm.invoke(prompt)
        
        # Extract the content from the AI response
        if hasattr(ai_response, 'content'):
            json_str = ai_response.content
        else:
            json_str = str(ai_response)
        
        # Clean up the response
        json_str = json_str.strip()
        
        # Remove markdown formatting if present
        if json_str.startswith('```json'):
            json_str = json_str[7:]  # Remove ```json
        if json_str.startswith('```'):
            json_str = json_str[3:]   # Remove ```
        if json_str.endswith('```'):
            json_str = json_str[:-3]  # Remove trailing ```
            
        json_str = json_str.strip()
        
        # Debug print (remove in production)
        print(f"Cleaned JSON string: '{json_str}'")
        
        if not json_str:
            return {"error": "Empty response from AI"}
        
        # Parse and validate JSON
        json_obj = json.loads(json_str)
        
        # Validate required fields
        if not isinstance(json_obj, dict):
            return {"error": "Response is not a valid JSON object"}
            
        required_fields = ['title', 'type', 'data']
        missing_fields = [field for field in required_fields if field not in json_obj]
        
        if missing_fields:
            return {"error": f"Missing required fields: {', '.join(missing_fields)}"}
        
        return json_obj
        
    except json.JSONDecodeError as e:
        return {"error": f"Error decoding JSON: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

# Define the tools with descriptions for the agent
tools = [
    Tool(
        name="text_to_sql_tool",
        func=text_to_sql_tool,
        description="useful for when you need to answer questions that require querying the database with SQL."
    ),
    Tool(
        name="rag_definition_tool",
        func=retrieve_and_generate,
        description="useful for when you need to answer questions about glossary or business terms, like KPIs and metrics."
    ),
    Tool(
        name="analytics_and_reporting_tool",
        func=analytics_and_reporting_tool,
        description="useful for when you need to generate chart or report specifications in JSON format for data visualization."
    )
]

# Create the agent prompt

agent_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant with access to a database. Your main task is to answer user questions using the tools provided.\n"
    "Respond to the user as a helpful data analyst.\n"
    "You have access to the following tools:\n"
    "{tools}\n"
    "The schema of the database is:\n"
    "{schema_summary}\n"
    "Use the following format EXACTLY:\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n"
    "\n"
    "IMPORTANT RULES:\n"
    "- Always start each step with 'Thought:'\n"
    "- When you have enough information to answer, write 'Thought: I now know the final answer'\n"
    "- Then provide 'Final Answer:' with your response\n"
    "- Do NOT provide the final answer without the proper format\n"
    "\n"
    "Begin!\n\n"
    "{agent_scratchpad}\n"
    "Question: {question}\n"
    "Thought:"
)

# Create the ReAct agent
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --- Main Execution Loop ---
questions = [
    "what are the top selling products?",
    "which product generate the most revenue? how many units sold and how much is total revenue?",
    "Who is the customer who made the most orders?",
    "What are the definitions of KPI and CAD?", 
    "Who is the highest paying customer",
    "Generate a bar chart specification showing the total amount of invoices by status ."
    "Show me a summary of all invoices by status",
]

for q in questions:
    print("-" * 50)
    print(f"Question: {q}")
    result = agent_executor.invoke({"question": q, "tools": tools, "schema_summary": schema_summary})
    print("\nFinal Result:", result["output"])
    print("-" * 50)
