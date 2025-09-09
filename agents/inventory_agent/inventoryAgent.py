from typing import List, Optional, Dict, Any
import os
import json
import math
import sqlite3
import hashlib
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
from lib.build_llm import build_llm

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
                "products", "stock", "stock_movements", "suppliers",
                "purchase_orders", "po_items", "approvals",
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
        
class InventorySQL:
    """
    Small helper aroung sqlite3 for multi-statement transactions
    (needed when creating a PO then its items).
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def run_tx_create_po(self, supplier_id: int, items: List[Dict[str, Any]]) -> int:
        """
        Creates a purchase order and its items in one transaction.
        items : [{product_id, quantity, unit_cost}]
        Returns: new purchase_orders.id
        """

        conn = sqlite.connect(self.db_path)
        try: 
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO purchase_orders (supplier_id, status) VALUES (?, 'draft')",
                (supplier_id,)
            )
            po_id = cur.lastrowid
            for item in items:
                cur.execute(
                    "INSERT INTO po_items (po_id, product_id, quantity, unit_cost) VALUES (?, ?, ?, ?)",
                    (po_id, int(item["product_id"]), int(item["quanitity"]), float(item["unit_cost"]))
                )
            conn.commit()
            return po_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

@dataclass
class InventoryConfig:
    usage_window_days: int = 30
    default_lead_time_days: int = 7
    default_safety_days: int = 3
    default_moq: int = 10 # Minimum Order Quantity

class InventoryAgent:
    """
    Inventory & Supply-Chain ReAct agent
    - read-only SQL tool
    - inventory helpers (low stock, usage/forecast)
    - reorder planner
    - propose PO into approvals (pending)'
    - commit approved PO into purchase_orders + po_items (transaction)
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # === safety + schema ===
        self.config = DatabaseConfig.from_env()
        self.schema_manager = DatabaseSchemaManager(self.config)
        self.schema = self.schema_manager.get_schema()
        self.confirmation_manager = WriteConfirmationManager(self.config)
        self.sql_tool = SQLQueryTool(self.config, self.schema, self.confirmation_manager)

        # === inventory helpers ===
        self.inv_sql = InventorySQL(self.config.db_path)
        self.inv_config = InventoryConfig()

        # === LLM + memory ===
        self.llm = build_llm(model_name)
        self.memory = ConversationBufferMemory(memory_key="chat_histroy", 
                                               return_messages=True, 
                                               input_key="input", 
                                               output_key="output")
        self.chat_history = self.memory

        # === assemble tools + agent ===
        tools = [
            self._create_read_sql_tool(), 
            self._tool_low_stock_report(),
            self._tool_usage_and_forecast(), 
            self._tool_reorder_planner(), 
            self._tool_propose_po(),                        # writes to approval (pending) via preview/confirm
            self._tool_commit_po_from_approval(),           # transactional write, also protected by preview/confirm
            self._create_write_preview_tool(),              # generic write preview (INSERT/UPDATE/DELETE)
            self._create_execute_confirmed_write_tool(),    # generic write confirm
        ]
        prompt = self.create_react_prompt()
        agent = create_react_agent(llm=self.llm, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=tools, 
            memory=self.memory, 
            verbose=True, 
            handle_parsing_errors=True, 
            max_iterations=10, 
            early_stopping_method="generate"
        )
