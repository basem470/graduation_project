import os
import sys
import sqlite3
import json
import re

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.agents.output_parsers import ReActSingleInputOutputParser

# Get the path to the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two directories to reach the project root
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add the project root to the Python path
sys.path.insert(0, project_root)

from lib.build_llm import build_llm
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
llm = build_llm("gpt-4o")

# --- Define the Tools for the Agent ---

def text_to_sql_tool(input_text: str):
    """
    Generates and executes a SQL query based on either:
    1. A natural language question
    2. A direct SQL query (if it starts with SELECT, INSERT, UPDATE, DELETE)
    """
    try:
        # Clean input
        input_text = input_text.strip()
        
        # Check if input is already a SQL query
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']
        first_word = input_text.split()[0].upper() if input_text.split() else ""
        
        if first_word in sql_keywords:
            # Input is already a SQL query, clean and execute it
            sql_query = clean_sql_query(input_text)
            print(f"\nExecuting provided SQL query: {sql_query}")
        else:
            # Input is a natural language question, generate SQL
            sql_query = generate_sql_from_question(input_text)
            print(f"\nGenerated SQL query: {sql_query}")
        
        # Execute the query
        return execute_sql_query(sql_query)
        
    except Exception as e:
        return f"Error in text_to_sql_tool: {str(e)}"

def clean_sql_query(sql_text: str) -> str:
    """Clean and extract SQL query from text that may contain markdown or other formatting."""
    # Remove markdown code blocks
    sql_text = re.sub(r'```sql\s*', '', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'```\s*', '', sql_text)
    
    # Remove extra whitespace and newlines
    sql_text = ' '.join(sql_text.split())
    
    # Ensure it ends with semicolon if it doesn't
    if not sql_text.rstrip().endswith(';'):
        sql_text = sql_text.rstrip() + ';'
    
    return sql_text

def generate_sql_from_question(question: str) -> str:
    """Generate SQL query from natural language question."""
    sql_prompt_template = ChatPromptTemplate.from_template(
        "You have access to the following database schema:\n{schema_summary}\n\n"
        "You are an SQL expert. Given the following question, generate a valid SQL query.\n"
        "Rules:\n"
        "- Generate ONLY the SQL query, no explanation or comments\n"
        "- Always start with SELECT for data retrieval questions\n"
        "- Use proper JOINs when needed\n"
        "- Include appropriate GROUP BY, ORDER BY, and LIMIT clauses\n"
        "- End with semicolon\n\n"
        "Question: {question}\n\n"
        "SQL Query:"
    )
    
    prompt = sql_prompt_template.format(question=question, schema_summary=schema_summary)
    ai_response = llm.invoke(prompt)
    
    # Extract and clean the SQL query
    sql_query = str(ai_response.content).strip()
    return clean_sql_query(sql_query)

def execute_sql_query(sql_query: str):
    """Execute SQL query and return results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        
        # Check if it's a SELECT query to fetch results
        if sql_query.upper().strip().startswith('SELECT'):
            results = cursor.fetchall()
            # Get column names for better formatting
            column_names = [description[0] for description in cursor.description]
            
            if results:
                # Format results as list of dictionaries for better readability
                formatted_results = []
                for row in results:
                    formatted_results.append(dict(zip(column_names, row)))
                return formatted_results
            else:
                return "No results found."
        else:
            # For INSERT, UPDATE, DELETE operations
            conn.commit()
            return f"Query executed successfully. {cursor.rowcount} rows affected."
            
    except Exception as e:
        return f"Error executing SQL: {str(e)}"
    finally:
        conn.close()

def analytics_and_reporting_tool(input_data):
    """
    Generates a JSON specification for a chart or report.
    Can accept either a question string or structured report data.
    """
    try:
        # Check if input is already structured data (dict or JSON string of dict)
        if isinstance(input_data, dict):
            # Input is already a structured report, just validate and return
            required_fields = ['title', 'type', 'data']
            if all(field in input_data for field in required_fields):
                return input_data
            else:
                return {"error": "Missing required fields: title, type, data"}
        
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
            "generate a valid JSON object that describes a chart visualization.\n\n"
            "The JSON object must have:\n"
            "- 'title': A descriptive title for the chart\n"
            "- 'type': Chart type (e.g., 'bar', 'pie', 'line', 'column')\n"
            "- 'data': Array of objects with 'label' and 'value' properties\n\n"
            "Example output:\n"
            "{{\n"
            '  "title": "Top 5 Products by Sales",\n'
            '  "type": "bar",\n'
            '  "data": [\n'
            '    {{"label": "Product A", "value": 150}},\n'
            '    {{"label": "Product B", "value": 120}}\n'
            '  ]\n'
            "}}\n\n"
            "Question: {question}\n\n"
            "Generate only the JSON object:"
        )
        
        prompt = json_prompt_template.format(question=question, schema_summary=schema_summary)
        ai_response = llm.invoke(prompt)
        
        # Extract and clean JSON from response
        json_str = extract_json_from_response(ai_response.content)
        
        if not json_str:
            return {"error": "Empty or invalid response from AI"}
        
        # Parse and validate JSON
        json_obj = json.loads(json_str)
        
        # Validate required fields
        required_fields = ['title', 'type', 'data']
        missing_fields = [field for field in required_fields if field not in json_obj]
        
        if missing_fields:
            return {"error": f"Missing required fields: {', '.join(missing_fields)}"}
        
        return json_obj
        
    except json.JSONDecodeError as e:
        return {"error": f"Error decoding JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from AI response, handling markdown and other formatting."""
    if not response_text:
        return ""
    
    # Remove markdown formatting
    json_str = response_text.strip()
    
    if json_str.startswith('```json'):
        json_str = json_str[7:]
    if json_str.startswith('```'):
        json_str = json_str[3:]
    if json_str.endswith('```'):
        json_str = json_str[:-3]
    
    json_str = json_str.strip()
    
    # Try to find JSON object boundaries
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = json_str[start_idx:end_idx + 1]
    
    return json_str

# Define the tools with descriptions for the agent
tools = [
    Tool(
        name="text_to_sql_tool",
        func=text_to_sql_tool,
        description=(
            "Use this tool to query the database. You can either:\n"
            "1. Provide a natural language question about the data\n"
            "2. Provide a direct SQL query (starting with SELECT, INSERT, UPDATE, or DELETE)\n"
            "The tool will execute the query and return formatted results."
        )
    ),
    Tool(
        name="rag_definition_tool",
        func=retrieve_and_generate,
        description="Use this tool to get definitions and explanations of business terms, KPIs, metrics, or glossary items."
    ),
    Tool(
        name="analytics_and_reporting_tool",
        func=analytics_and_reporting_tool,
        description=(
            "Use this tool to generate chart or report specifications in JSON format for data visualization. "
            "Provide a description of what kind of chart or report you want to create."
        )
    )
]

# Create the agent prompt
agent_prompt = ChatPromptTemplate.from_template(
    "You are a helpful data analyst with access to a database and analytical tools. "
    "Your task is to answer user questions accurately and provide insights.\n\n"
    "Available tools:\n{tools}\n\n"
    "Database schema:\n{schema_summary}\n\n"
    "Instructions:\n"
    "- For data queries, use the text_to_sql_tool\n"
    "- For business term definitions, use the rag_definition_tool\n"
    "- For chart/report specifications, use the analytics_and_reporting_tool\n"
    "- Always think through your approach before taking action\n"
    "- Provide clear, concise final answers\n\n"
    "Use this exact format:\n"
    "Question: the input question you must answer\n"
    "Thought: think about what you need to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (repeat Thought/Action/Action Input/Observation as needed)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {question}\n"
    "Thought: {agent_scratchpad}"
)

# Create the ReAct agent
class AnalyticsReActAgent:
    def __init__(self):
        self.llm = llm
        self.tools = tools
        self.agent_prompt = agent_prompt
        
        self.agent = create_react_agent(llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=10,  # Prevent infinite loops
            max_execution_time=60  # Timeout after 60 seconds
        )
    
    def query(self, question: str):
        """Execute a query using the analytics agent."""
        try:
            result = self.agent_executor.invoke({
                "question": question,
                "tools": self.tools,
                "schema_summary": schema_summary,
                "tool_names": [tool.name for tool in self.tools]
            })
            return result
        except Exception as e:
            return {"error": f"Agent execution failed: {str(e)}"}

# --- Main Execution ---
def main():
    """Main execution function for testing the agent."""
    analytics_agent = AnalyticsReActAgent()
    
    questions = [
        "What are the top 5 selling products by quantity?",
        "Which product generates the most revenue? How many units were sold and what is the total revenue?",
        "Who is the customer who made the most orders?",
        "What are the definitions of KPI and CAD?",
        "Who is the highest paying customer?",
        "Generate a bar chart specification showing the total amount of invoices by status.",
        "Show me a summary of all invoices by status",
    ]

    for q in questions:
        print("=" * 70)
        print(f"Question: {q}")
        print("=" * 70)
        
        result = analytics_agent.query(q)
        
        if "error" in result:
            print(f"\nError: {result['error']}")
        else:
            print(f"\nFinal Result: {result.get('output', 'No output available')}")
        
        print("=" * 70)