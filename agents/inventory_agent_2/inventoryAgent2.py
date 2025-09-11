from typing import List, Optional, Dict, Any
import os
import json
import math
import sqlite3
import logging
import hashlib
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SQLDatabase
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
                    with sqlite3.connect(self.config.db_path) as conn:
                        cur = conn.cursor()
                        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}")
                        row_count = cur.fetchone()[0] or 0
                    
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
                    with sqlite3.connect(self.config.db_path) as conn:
                        cur = conn.cursor()
                        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}")
                        row_count = cur.fetchone()[0] or 0
                    
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
    Small helper around sqlite3 for multi-statement transactions
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

        conn = sqlite3.connect(self.db_path)
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
                    (po_id, int(item["product_id"]), int(item["quantity"]), float(item["unit_cost"]))
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

class InventoryAgent2:
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
        self.memory = ConversationBufferMemory(memory_key="chat_history", 
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
            self._tool_propose_po(),
            self._tool_confirm_po_proposal(),                                                # writes to approval (pending) via preview/confirm
            self._tool_commit_po_from_approval(),           # transactional write, also protected by preview/confirm
            self._create_write_preview_tool(),              # generic write preview (INSERT/UPDATE/DELETE)
            self._create_execute_confirmed_write_tool(),    # generic write confirm
        ]
        prompt = self._create_react_prompt()
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

    # ============ generic SQL tools ============
    def _create_read_sql_tool(self) -> Tool:
        desc = f"""
        Read-only SELECT over inventory tables.
        Allowed tables: {', '.join(self.config.allowed_tables)}
        Schema: 
        {self.schema}
        """
        return Tool(name="Inventory_SQL_Read", func=self.sql_tool.execute_read_query, description=desc)
    
    def _create_write_preview_tool(self) -> Tool:
        desc = "Preview any INSERT/UPDATE/DELETE first; returns a token to confirm later."
        return Tool(name="Preview_Write_Operation", func=self.sql_tool.preview_write_query, description=desc)
    
    def _create_execute_confirmed_write_tool(self) -> Tool:
        desc = "Execute a previously previewed write using the token."
        return Tool(name="Execute_Confirmed_Write", func=self.sql_tool.execute_confirmed_write, description=desc)
    
    # ============ inventory domain tools ============
    def _tool_low_stock_report(self) -> Tool:
        def low_stock(_: str) -> str:
            q = """
            SELECT p.id AS product_id, p.sku, p.name, s.qty_on_hand, s.reorder_point
            FROM products p
            JOIN stock s ON s.product_id = p.id
            WHERE s.qty_on_hand <= s.reorder_point
            ORDER BY s.qty_on_hand ASC
            """
            return self.sql_tool.execute_read_query(q)
        return Tool(
            name="low_stock_report_tool",
            func=low_stock,
            description="List products where qty_on_hand <= reorder_point."
        )
    
    def _tool_usage_and_forecast(self) -> Tool:
        def usage_forecast(days_str: str) -> str:
            """
            Input: optional integer string for usage window (days). Defaults to 30.
            Computes avg daily usage (based on negative stock_movements.change_qty) and 
            a naive next-7-days forecast. 
            """

            try:
                wnd = int(days_str) if days_str.strip() else self.inv_config.usage_window_days
            except:
                wnd = self.inv_config.usage_window_days

            since_dt = (datetime.now() - timedelta(days=wnd)).strftime("%Y-%m-%d %H:%M:%S")
            q = f"""
            SELECT p.id AS product_id, p.sku, p.name, SUM(CASE WHEN sm.change_qty < 0 THEN -sm.change_qty ELSE 0 END) AS total_out
            FROM products p
            LEFT JOIN stock_movements sm ON sm.product_id = p.id AND sm.created_at >= '{since_dt}'
            GROUP BY p.id, p.sku, p.name
            """
            res = self._run_sql_fetchall(q)

            # avg daily usage + simple forecast (avg per day * 7)
            rows = []
            for r in res:
                total_out = r["total_out"] if r["total_out"] is not None else 0
                avg_daily = total_out / max(wnd, 1)
                forecast_7 = round(avg_daily * 7, 2)
                rows.append({
                    "product_id": r["product_id"],
                    "sku": r["sku"],
                    "name": r["name"],
                    "avg_daily_usage": round(avg_daily, 3), 
                    "forecast_next_7_days": forecast_7
                })
            return json.dumps(rows, indent=2)
        
        return Tool(
            name="usage_and_forecast_tool", 
            func=usage_forecast,
            description="Input optional number of days (default 30). Returns average daily usage & 7-day forecast per product (JSON)"
        )
    
    def _tool_reorder_planner(self) -> Tool:
        def plan(params_json: str) -> str:
            """
            Input: JSON (optional): 
            {
                "lead_time_days": 7,
                "safety_days": 3,
                "moq": 10
            }
            For each product, recommends qty = max(avg_daily * (lead + safety) - qty_on_hand, 0) rounded up to MOQ.
            """
            try: 
                params = json.loads(params_json) if params_json.strip() else {}
            except:
                params = {}
            lead = int(params.get("lead_time_days", self.inv_config.default_lead_time_days))
            safety = int(params.get("safety_days", self.inv_config.default_safety_days))
            moq =  int(params.get("moq", self.inv_config.default_moq))

            # 1. Usage (Last N days)
            wnd = self.inv_config.usage_window_days
            since_dt = (datetime.now() - timedelta(days=wnd)).strftime("%Y-%m-%d %H:%M:%S")
            usage_q = f"""
            WITH usage AS (
                SELECT p.id AS product_id, SUM(CASE WHEN sm.change_qty < 0 THEN -sm.change_qty ELSE 0 END) AS total_out
                FROM products p
                LEFT JOIN stock_movements sm ON sm.product_id = p.id AND sm.created_at >= '{since_dt}'
                GROUP BY p.id
            )
            SELECT p.id AS product_id, p.sku, p.name, IFNULL(u.total_out, 0) AS total_out, s.qty_on_hand, s.reorder_point
            FROM products p
            JOIN stock s ON s.product_id = p.id
            LEFT JOIN usage u ON u.product_id = p.id
            """
            rows = self._run_sql_fetchall(usage_q)

            suggestions = []
            for r in rows:
                total_out = r["total_out"] or 0
                avg_daily = total_out / max(wnd, 1)
                target = avg_daily * (lead + safety) # Demand to cover
                need = max(target - (r["qty_on_hand"] or 0), 0.0)
                # Round up to MOQ
                rec_qty = 0 if need <= 0 else int(math.ceil(need / moq) * moq)
                suggestions.append({
                    "product_id": r["product_id"],
                    "sku": r["sku"],
                    "name": r["name"],
                    "avg_daily_usage": round(avg_daily, 3), 
                    "recommended_qty": rec_qty
                })

            # Filter zero recommendations for clarity
            suggestions = [x for x in suggestions if x["recommended_qty"] > 0]
            return json.dumps(suggestions, indent=2)
        return Tool(
            name="reorder_planner_tool",
            func=plan,
            description="Input JSON with lead_time_days/safety_days/moq; returns JSON list of recommended_qty per product."
        )
    
    def _tool_propose_po(self) -> Tool:
        def propose(po_json: str) -> str:
            """
            Input: JSON: 
            {
                "supplier_id": 1,
                "items": [{"product_id": 1, "quantity": 50, "unit_cost": 9.5}, ...]
            }
            PREVIEW ONLY: creates a confirmation token; once confirmed, it will INSERT a row into approvals
            with payload_json describing the PO proposal (status="pending").
            """
            try: 
                payload = json.loads(po_json)
                supplier_id = int(payload["supplier_id"])
                items = payload["items"]
            except Exception as e:
                return f"Invalid JSON: {e}"
            
            # Human-readable preview
            affected = "Proposed PO -> approvals.pending\n" + json.dumps(payload, indent=2)
            token = self.confirmation_manager.add_pending_write(
                sql_query="", # not used here; we will execute Python code on confirm
                operation_type="PROPOSE_PO", 
                affected_data=affected, 
                row_count=len(items)
            )
            return (
                f"📋 WRITE OPERATION PREVIEW\nOperation: PROPOSE_PO\n"
                f"Rows affected (items): {len(items)}\n"
                f"{affected}\n\n"
                f"CONFIRMATION REQUIRED: token {token}\n"
            )
        return Tool(
            name="propose_po_tool", 
            func=propose, 
            description="Preview a new PO proposal (insert a pending row into approvals on confirm)."
        )
    
    def _tool_confirm_po_proposal(self) -> Tool:
        def confirm(token_and_requester: str) -> str:
            # expects: "token=ABCDEFGH requested_by=ahmed"
            try: 
                token = token_and_requester.split("token=")[1].split()[0]
                requested_by = token_and_requester.split("requested_by=")[1].split()[0]
            except:
                return "Usage: token=ABCDEFGH requested_by=yourname"
            
            write_op = self.confirmation_manager.confirm_write(token)
            if not write_op or write_op.operation_type != "PROPOSE_PO":
                return f"Invalid or expired token for proposal: {token}"
            
            # recover the proposal payload we put in "affected_data"
            # it was "Proposed PO -> approvals.pending\n{JSON}"
            try:
                payload_json = write_op.affected_data.split("\n", 1)[1]
            except Exception:
                return "Could not recover proposal payload."
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            approval_id = self._execute_returning_id(
                "INSERT INTO approvals (module, payload_json, status, requested_by, created_at) VALUES (?, ?, 'pending', ?, ?)",
                ("inventory", payload_json, requested_by, now)
            )
            return f"✅ Proposal saved to approvals. approval_id={approval_id}"
        return Tool(
            name="confirm_po_proposal_tool",
            func=confirm,
            description="Consumes the PROPOSE_PO token and inserts a pending row into approvals. Input: 'token=... requested_by=...'."
        )
    
    def _tool_commit_po_from_approval(self) -> Tool:
        def commit(input_str: str) -> str:
            """
            Two modes:
            1) 'preview approval_id=123' -> shows what will be written, returns a token.
            2) 'confirm token=ABCDEFGH decided_by=alice' -> execute sthe trannsaction:
                - INSERT into purchase_orders + po_items
                - UPDATE approvals.status='approved', decided_by, decided_at
            """
            s = input_str.strip()
            if s.lower().startswith("preview"):
                # parse approval_id
                try:
                    approval_id = int(s.split("approval_id=")[1])
                except:
                    return "Usage: preview approval_id=123"
            
                row = self._fetchone("SELECT id, payload_json FROM approvals WHERE id=? AND status='pending'", (approval_id,))
                if not row:
                    return f"No pending approval with id={approval_id}"
                payload = json.loads(row["payload_json"])
                supplier_id = int(payload["supplier_id"])
                items = payload["items"]

                preview_text = f"Will create purchase_order (supplier_id={supplier_id} and {len(items)} po_items.)"
                token = self.confirmation_manager.add_pending_write(
                    sql_query=f"APPROVAL:{approval_id}",
                    operation_type="COMMIT_APPROVED_PO", 
                    affected_data=preview_text + "\n" + json.dumps(payload, indent=2), 
                    row_count=len(items)
                )
                return (
                    f"📋WRITE OPERATION PREVIEW\nOperation: COMMIT_APPROVED_PO\n"
                    f"Rows affected (items): {len(items)}\n"
                    f"{preview_text}\n"
                    f"CONFIRMATION REQUIRED: token {token}"
                )
            
            elif s.lower().startswith("confirm"):
                try:
                    token = s.split("token=")[1].split()[0]
                    decided_by = s.split("decided_by=")[1].split()[0]
                except:
                    return "Usage: confirm token=ABCDEFGH decided_by=alice"
                
                write_op = self.confirmation_manager.confirm_write(token)
                if not write_op:
                    return f"Invalid or expired token: {token}"
                
                # get approval_id back from write_op.sql_query
                approval_id = int(write_op.sql_query.split(":")[1])
                row = self._fetchone("SELECT payload_json FROM approvals WHERE id=?", (approval_id,))
                if not row:
                    return f"Approval id={approval_id} not found."
                
                payload = json.loads(row["payload_json"])
                supplier_id = int(payload["supplier_id"])
                items = payload["items"]

                # Run the transactional insert for PO + items
                po_id = self.inv_sql.run_tx_create_po(supplier_id, items)

                # Mark approval as approved
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._execute(
                    "UPDATE approvals SET status='approved', decided_by=?, decided_at=? WHERE id=?",
                    (decided_by, now, approval_id)
                )

                return (
                    f"✅ PO committed. purchase_orders.id={po_id}\n"
                    f"approval {approval_id} -> approved by {decided_by} at {now}"
                )
            else:
                return "Usage:\n- preview approval_id=123\n- confirm token=ABCDEFGH decided_by=alice"
        return Tool(
            name="commit_po_from_approval_tool",
            func=commit,
            description="Preview/commit an approved PO from approvals payload. See usage in docstring."
        )
    
    # ============ ReAct prompt ============
    def _create_react_prompt(self) -> PromptTemplate:
        template = """
        You are the Inventory & Supply-Chain agent for a small ERP.
        You can read inventory tables, compute usage/forecasts, plan reorders, and safely propose/commit purchase orders.

        TOOLS:
        {tools}

        RULES FOR WRITES:
        1) ALWAYS preview write operations first (either Preview_Write_Operation or the propose/commit tools that yield a token).
        2) Execute only after user confirms with the token.
        3) Never run destructive SQL without WHERE.

        CONVERSATION MEMORY:
        {chat_history}

        Use the exact ReAct format:

        Question: the user input
        Thought: reasoning
        Action: one of [{tool_names}]
        Action Input: input to the action
        Observation: tool result
        ... (repeat as needed)
        Thought: I now know the final answer
        Final Answer: answer

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """

        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"]
        )
    
    # ============ DB Helpers ============
    def _run_sql_fetchall(self, query: str, params: Optional[tuple]=None) -> List[sqlite3.Row]:
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params or ())
        rows = cur.fetchall()
        conn.close()
        return rows
    
    def _fetchone(self, query: str, params: tuple) -> Optional[sqlite3.Row]:
        conn = sqlite3.connect(self.config.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        conn.close()
        return row
    
    def _execute(self, query: str, params: tuple=()) -> None:
        conn = sqlite3.connect(self.config.db_path)
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        conn.close()

    def _execute_returning_id(self, query: str, params: tuple=()) -> int:
        conn = sqlite3.connect(self.config.db_path)
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    # ============ Agent entrypoint ============
    def _format_chat_history(self) -> str:
        if not self.chat_history.buffer:
            return "No previous conversation."
        formatted = []
        for m in self.chat_history.buffer[-6:]:
            role = "Human" if m.type == "human" else "Assistant"
            formatted.append(f"{role}: {m.content}")
        return "\n".join(formatted) if formatted else "No previous conversation."
    
    def query(self, question: str) -> Dict[str, Any]:
        try: 
            ch = self._format_chat_history()
            result = self.agent_executor.invoke({"input": question, "chat_history": ch})
            output = result.get("output", str(result))
            self.chat_history.save_context({"input": question}, {"output": output})
            return {"success": True, "result": output, "question": question, "error": None}
        except Exception as e:
            return {"success": False, "result": None, "question": question, "error": str(e)}
        
def get_test_queries() -> List[str]:
    return [
        "Show low stock items.",
        "Compute usage and forecast for the last 30 days", 
        "Plan reorders with lead_time_days=7, safety_days=3, moq=10",
        "Propose a PO: supplier 1, items [{product_id: 1, quantity: 50, unit_cost: 9.5}]",
    ]

def run_tests(agent: InventoryAgent2):
    tests = [
        "Show me a low_stock_report_tool",
        "Run usage_and_forecast_tool for 30", 
        'Use reorder_planner_tool with {"lead_time_days":7, "safety_days":3, "moq":10}',
        'Use propose_po_tool with {"supplier_id": 1, "items": [{"product_id": 1, "quantity": 50, "unit_cost": 9.5}]}'
    ]
    for t in tests:
        print("\n--- TEST ---")
        print("Q:", t) 
        res = agent.query(t)
        print("OUT:", res["result"])

if __name__ == "__main__":
    agent = InventoryAgent2()
    run_tests(agent)
    print("\nTip:")
    print("- After proposing a PO, run: commit_po_from_approval_tool with 'preview approval_id=123'")
    print("- Then: commit_po_from_approval_tool with 'confirm token=ABCDEFGH decided_by=Ahmed'")
