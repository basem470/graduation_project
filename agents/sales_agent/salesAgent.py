# agents/sales/react_agent_schema.py
from typing import List, Optional, Dict, Any, Union
import os
import sqlite3
import logging
import hashlib
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from lib.build_llm import build_llm


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_path: str
    allowed_tables: List[str]
    max_rows_affected: int = 100
    confirmation_timeout: int = 300  # 5 minutes
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        return cls(
            db_path=os.getenv("DB_PATH", "db/erp.db"),
            allowed_tables=[
                "customers", "leads", "orders", "order_items", 
                "tickets", "products", "customer_kv"
            ],
            max_rows_affected=int(os.getenv("MAX_ROWS_AFFECTED", "100")),
            confirmation_timeout=int(os.getenv("CONFIRMATION_TIMEOUT", "300"))
        )


@dataclass
class PendingWrite:
    """Represents a pending write operation awaiting confirmation."""
    token: str
    sql_query: str
    operation_type: str
    affected_data: str
    row_count: int
    timestamp: datetime
    expires_at: datetime


class WriteConfirmationManager:
    """Manages pending write operations and confirmations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pending_writes: Dict[str, PendingWrite] = {}
    
    def generate_token(self, sql_query: str) -> str:
        """Generate a unique token for a write operation."""
        content = f"{sql_query}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def add_pending_write(self, sql_query: str, operation_type: str, 
                         affected_data: str, row_count: int) -> str:
        """Add a pending write operation."""
        token = self.generate_token(sql_query)
        now = datetime.now()
        expires_at = now + timedelta(seconds=self.config.confirmation_timeout)
        
        self.pending_writes[token] = PendingWrite(
            token=token,
            sql_query=sql_query,
            operation_type=operation_type,
            affected_data=affected_data,
            row_count=row_count,
            timestamp=now,
            expires_at=expires_at
        )
        
        return token
    
    def get_pending_write(self, token: str) -> Optional[PendingWrite]:
        """Get a pending write operation by token."""
        write_op = self.pending_writes.get(token)
        if write_op and datetime.now() > write_op.expires_at:
            # Remove expired operation
            del self.pending_writes[token]
            return None
        return write_op
    
    def confirm_write(self, token: str) -> Optional[PendingWrite]:
        """Confirm and remove a pending write operation."""
        write_op = self.get_pending_write(token)
        if write_op:
            del self.pending_writes[token]
        return write_op
    
    def cleanup_expired(self):
        """Remove expired pending writes."""
        now = datetime.now()
        expired_tokens = [
            token for token, write_op in self.pending_writes.items()
            if now > write_op.expires_at
        ]
        for token in expired_tokens:
            del self.pending_writes[token]


class DatabaseSchemaManager:
    """Manages database schema operations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._validate_database()
    
    def _validate_database(self) -> None:
        """Validate that the database file exists."""
        if not os.path.exists(self.config.db_path):
            raise FileNotFoundError(f"Database file not found: {self.config.db_path}")
    
    def get_schema(self) -> str:
        """
        Get the database schema for allowed tables.
        
        Returns:
            Formatted string containing table schemas
        """
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                all_tables = {row[0] for row in cursor.fetchall()}
                
                # Filter to allowed tables only
                available_tables = all_tables.intersection(self.config.allowed_tables)
                
                if not available_tables:
                    logger.warning("No allowed tables found in database")
                    return "No accessible tables found."
                
                schema_parts = []
                for table_name in sorted(available_tables):
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    
                    if columns:
                        col_info = []
                        for col in columns:
                            col_name, col_type = col[1], col[2]
                            col_info.append(f"{col_name} ({col_type})")
                        
                        col_list = ", ".join(col_info)
                        schema_parts.append(f"Table {table_name}: {col_list}")
                
                return "\n".join(schema_parts)
                
        except sqlite3.Error as e:
            logger.error(f"Database error while getting schema: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while getting schema: {e}")
            raise


class SQLQueryTool:
    """Handles SQL query execution with safety measures."""
    
    def __init__(self, config: DatabaseConfig, schema: str, confirmation_manager: WriteConfirmationManager):
        self.config = config
        self.schema = schema
        self.confirmation_manager = confirmation_manager
        self.db = SQLDatabase.from_uri(f"sqlite:///{config.db_path}")
    
    def _get_operation_type(self, query: str) -> str:
        """Extract operation type from SQL query."""
        query_upper = query.strip().upper()
        if query_upper.startswith('SELECT'):
            return 'SELECT'
        elif query_upper.startswith('INSERT'):
            return 'INSERT'
        elif query_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif query_upper.startswith('DELETE'):
            return 'DELETE'
        else:
            return 'UNKNOWN'
        
    def _validate_write_query(self,query: str) -> tuple[bool, str]:
        """Validate write query for safety."""
        query_upper = query.strip().upper()

        # Patterns that are always forbidden
        forbidden_patterns = [
            r'\bDROP\s+TABLE\b',
            r'\bDROP\s+DATABASE\b',
            r'\bTRUNCATE\b',
            r'\bALTER\s+TABLE.*\bDROP\b',
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, query_upper):
                return False, f"Dangerous operation detected: {pattern}"

        # DELETE without WHERE
        if query_upper.startswith("DELETE"):
            if not re.search(r'\bWHERE\b', query_upper):
                return False, "DELETE without WHERE clause is not allowed"

        # UPDATE without WHERE
        if query_upper.startswith("UPDATE"):
            if not re.search(r'\bWHERE\b', query_upper):
                return False, "UPDATE without WHERE clause is not allowed"

        return True, "Query is valid"
    
    def _get_affected_data_preview(self, query: str) -> tuple[str, int]:
        """Get preview of data that would be affected by the operation."""
        try:
            operation_type = self._get_operation_type(query)
            
            if operation_type == 'UPDATE':
                # For UPDATE, show current data that would be changed
                # Convert UPDATE to SELECT to show current state
                update_match = re.match(
                    r'UPDATE\s+(\w+)\s+SET\s+.*?\s+WHERE\s+(.*)',
                    query.strip(),
                    re.IGNORECASE
                )
                if update_match:
                    table_name, where_clause = update_match.groups()
                    preview_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
                    current_data = self.db.run(preview_query)
                    
                    # Count affected rows
                    count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
                    count_result = self.db.run(count_query)
                    row_count = int(count_result.strip()) if count_result.strip().isdigit() else 0
                    
                    return f"Current data that will be updated:\n{current_data}", row_count
            
            elif operation_type == 'DELETE':
                # For DELETE, show data that would be removed
                delete_match = re.match(
                    r'DELETE\s+FROM\s+(\w+)\s+WHERE\s+(.*)',
                    query.strip(),
                    re.IGNORECASE
                )
                if delete_match:
                    table_name, where_clause = delete_match.groups()
                    preview_query = f"SELECT * FROM {table_name} WHERE {where_clause}"
                    current_data = self.db.run(preview_query)
                    
                    # Count affected rows
                    count_query = f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}"
                    count_result = self.db.run(count_query)
                    row_count = int(count_result.strip()) if count_result.strip().isdigit() else 0
                    
                    return f"Data that will be deleted:\n{current_data}", row_count
            
            elif operation_type == 'INSERT':
                # For INSERT, show the data that would be added
                return f"Data to be inserted:\n{query}", 1
            
            return "No preview available", 0
            
        except Exception as e:
            logger.error(f"Error getting affected data preview: {e}")
            return f"Error getting preview: {str(e)}", 0
    
    def execute_read_query(self, query: str) -> str:
        """Execute read-only SQL query."""
        try:
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                return "Error: Only SELECT queries are allowed in read mode."
            
            result = self.db.run(query)
            return result if result else "No results found."
            
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return f"Query execution failed: {str(e)}"
    
    def preview_write_query(self, query: str) -> str:
        """Preview a write operation without executing it."""
        try:
            # Clean up expired confirmations
            self.confirmation_manager.cleanup_expired()
            
            operation_type = self._get_operation_type(query)
            
            if operation_type == 'SELECT':
                return "This is a SELECT query. Use the regular SQL tool for read operations."
            
            # Validate the write query
            is_valid, validation_message = self._validate_write_query(query)
            if not is_valid:
                return f"❌ Invalid query: {validation_message}"
            
            # Get preview of affected data
            affected_data, row_count = self._get_affected_data_preview(query)
            
            # Check row limit
            if row_count > self.config.max_rows_affected:
                return f"❌ Operation would affect {row_count} rows, which exceeds the limit of {self.config.max_rows_affected} rows."
            
            # Create pending write operation
            token = self.confirmation_manager.add_pending_write(
                query, operation_type, affected_data, row_count
            )
            
            preview_message = f"""
📋 **WRITE OPERATION PREVIEW**
Operation: {operation_type}
Rows affected: {row_count}
Query: {query}

{affected_data}

⚠️ **CONFIRMATION REQUIRED**
To execute this operation, you must confirm by using the execute_confirmed_write tool with token: {token}
This confirmation will expire in {self.config.confirmation_timeout // 60} minutes.

Type "CONFIRM {token}" or ask me to execute the confirmed write with token {token}.
"""
            return preview_message
            
        except Exception as e:
            logger.error(f"Error previewing write query: {e}")
            return f"Error previewing query: {str(e)}"
    
    def execute_confirmed_write(self, token: str) -> str:
        """Execute a confirmed write operation."""
        try:
            # Get and confirm the pending write
            write_op = self.confirmation_manager.confirm_write(token)
            
            if not write_op:
                return f"❌ Invalid or expired confirmation token: {token}"
            
            # Execute the write operation
            result = self.db.run(write_op.sql_query)
            
            # Log the operation
            logger.info(f"Write operation executed - Token: {token}, Operation: {write_op.operation_type}, Rows: {write_op.row_count}")
            
            success_message = f"""
✅ **WRITE OPERATION COMPLETED**
Operation: {write_op.operation_type}
Rows affected: {write_op.row_count}
Query executed: {write_op.sql_query}

Result: {result if result else 'Operation completed successfully'}
"""
            return success_message
            
        except Exception as e:
            logger.error(f"Error executing confirmed write: {e}")
            return f"❌ Error executing write operation: {str(e)}"


class SalesReactAgent:
    """Main sales ReAct agent class with write preview capabilities."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.config = DatabaseConfig.from_env()
        self.schema_manager = DatabaseSchemaManager(self.config)
        self.schema = self.schema_manager.get_schema()
        self.confirmation_manager = WriteConfirmationManager(self.config)
        self.sql_tool = SQLQueryTool(self.config, self.schema, self.confirmation_manager)
        self.llm = build_llm(model_name)
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        self.chat_history = self.memory 
        self.agent_executor = self._setup_agent()
    
    def _create_read_sql_tool(self) -> Tool:
        """Create the SQL database read tool."""
        description = f"""
        Use this tool to query the database using SQL SELECT statements only.
        Pass only valid and runnable SQL SELECT code that can be executed directly.
        
        Available tables and their schemas:
        {self.schema}
        
        Restrictions:
        - Only SELECT queries are allowed
        - You can only access these tables: {', '.join(self.config.allowed_tables)}
        - Input must be valid SQL syntax
        
        Example queries:
        - SELECT * FROM customers LIMIT 5;
        - SELECT COUNT(*) FROM orders WHERE amount > 1000;
        - SELECT c.name, COUNT(o.id) as order_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id;
        """
        
        return Tool(
            name="SQL_Database_Read",
            func=self.sql_tool.execute_read_query,
            description=description
        )
    
    def _create_write_preview_tool(self) -> Tool:
        """Create the write preview tool."""
        description = f"""
        Use this tool to preview write operations (INSERT, UPDATE, DELETE) before execution.
        This tool shows what data will be affected and provides a confirmation token.
        
        IMPORTANT: Always use this tool BEFORE any write operation.
        
        Available tables: {', '.join(self.config.allowed_tables)}
        
        Safety features:
        - Shows affected data before execution
        - Requires explicit confirmation
        - Prevents operations affecting more than {self.config.max_rows_affected} rows
        - Blocks dangerous operations without WHERE clauses
        
        Example usage:
        - UPDATE customers SET status = 'active' WHERE city = 'New York'
        - INSERT INTO customers (name, email) VALUES ('John Doe', 'john@example.com')
        - DELETE FROM leads WHERE status = 'rejected' AND created_at < '2024-01-01'
        """
        
        return Tool(
            name="Preview_Write_Operation",
            func=self.sql_tool.preview_write_query,
            description=description
        )
    
    def _create_execute_confirmed_write_tool(self) -> Tool:
        """Create the execute confirmed write tool."""
        description = """
        Use this tool to execute a previously previewed write operation.
        You must provide the confirmation token received from the Preview_Write_Operation tool.
        
        Input: confirmation token (8-character string)
        
        This tool will only work with valid, non-expired tokens from recent preview operations.
        """
        
        return Tool(
            name="Execute_Confirmed_Write",
            func=self.sql_tool.execute_confirmed_write,
            description=description
        )
    
    def _create_react_prompt(self) -> PromptTemplate:
        """Create the ReAct prompt template with memory."""
        template = """You are a helpful AI assistant that can query and modify a sales database to answer questions and perform operations.

You have access to the following tools:

{tools}

IMPORTANT INSTRUCTIONS FOR WRITE OPERATIONS:
1. ALWAYS use Preview_Write_Operation tool FIRST before any INSERT, UPDATE, or DELETE operation
2. Show the user what data will be affected and ask for confirmation
3. Only use Execute_Confirmed_Write after the user confirms with the provided token
4. Never perform write operations directly without preview and confirmation

CONVERSATION MEMORY:
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"]
        )
    
    def _setup_agent(self) -> AgentExecutor:
        """Initialize and configure the ReAct agent."""
        tools = [
            self._create_read_sql_tool(),
            self._create_write_preview_tool(),
            self._create_execute_confirmed_write_tool()
        ]
        prompt = self._create_react_prompt()
        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
    
    def _format_chat_history(self) -> str:
        """Format chat history for the prompt."""
        if not self.chat_history.buffer:
            return "No previous conversation."
        
        formatted_history = []
        for message in self.chat_history.buffer[-6:]:  # Keep last 6 messages
            if hasattr(message, 'content'):
                role = "Human" if message.type == "human" else "Assistant"
                formatted_history.append(f"{role}: {message.content}")
        
        return "\n".join(formatted_history) if formatted_history else "No previous conversation."
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute a query using the agent.
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary containing result and metadata
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Format chat history
            chat_history_str = self._format_chat_history()
            
            # Invoke agent with chat history
            result = self.agent_executor.invoke({
                "input": question,
                "chat_history": chat_history_str
            })
            
            # Extract the output
            output = result.get("output", str(result))
            
            # Use save_context to update conversation memory
            self.chat_history.save_context(
                {"input": question},
                {"output": output}
            )
            
            return {
                "success": True,
                "result": output,
                "question": question,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return {
                "success": False,
                "result": None,
                "question": question,
                "error": str(e)
            }
    
    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        """Get list of pending write confirmations."""
        self.confirmation_manager.cleanup_expired()
        return [
            {
                "token": write_op.token,
                "operation": write_op.operation_type,
                "row_count": write_op.row_count,
                "expires_at": write_op.expires_at.isoformat(),
                "sql": write_op.sql_query
            }
            for write_op in self.confirmation_manager.pending_writes.values()
        ]


def get_test_queries() -> List[str]:
    """Get list of test queries for validation."""
    return [
        # Read queries
        "Get the first 5 customers created in March 2024",
        "List all orders with amount greater than 1000", 
        "Show the last 3 tickets created",
        "Find all products with quantity less than 10",
        "Get customer details for email 'john.doe@example.com'",
        
        # Write queries (will require confirmation)
        "Update all customers in New York to have status 'active'",
        "Insert a new customer named 'Test Customer' with email 'test@example.com'",
        "Delete all leads with status 'rejected' created before 2024",
        
        # Complex queries
        "Show total number of orders per customer",
        "List all customers who never placed an order"
    ]


def run_tests(agent: SalesReactAgent, queries: Optional[List[str]] = None) -> None:
    """Run test queries against the agent."""
    test_queries = queries or get_test_queries()
    
    print("=" * 60)
    print("SALES REACT AGENT - TEST RESULTS")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: {query}")
        print("-" * 40)
        
        result = agent.query(query)
        
        if result["success"]:
            print(f"✅ Success")
            print(f"Result: {result['result']}")
        else:
            print(f"❌ Failed")
            print(f"Error: {result['error']}")
        
        # Show pending confirmations after each query
        pending = agent.get_pending_confirmations()
        if pending:
            print(f"\n📋 Pending Confirmations: {len(pending)}")
            for p in pending:
                print(f"  Token: {p['token']} | Operation: {p['operation']} | Rows: {p['row_count']}")


def interactive_mode(agent: SalesReactAgent):
    """Run the agent in interactive mode."""
    print("🤖 Sales Agent Interactive Mode")
    print("Type 'quit' to exit, 'pending' to see pending confirmations")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'pending':
                pending = agent.get_pending_confirmations()
                if pending:
                    print(f"\n📋 Pending Confirmations ({len(pending)}):")
                    for p in pending:
                        print(f"  🔸 Token: {p['token']}")
                        print(f"    Operation: {p['operation']} | Rows: {p['row_count']}")
                        print(f"    Expires: {p['expires_at']}")
                        print(f"    SQL: {p['sql'][:100]}...")
                else:
                    print("📋 No pending confirmations")
                continue
            
            if not user_input:
                continue
            
            print(f"\n🤖 Agent: Processing...")
            result = agent.query(user_input)
            
            if result["success"]:
                print(f"🤖 Agent: {result['result']}")
            else:
                print(f"❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    try:
        # Initialize agent
        agent = SalesReactAgent()
        
        # Choose mode
        mode = input("Choose mode: (1) Test mode, (2) Interactive mode [1]: ").strip() or "1"
        
        if mode == "2":
            interactive_mode(agent)
        else:
            run_tests(agent)
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        print(f"Error: {e}")