# agents/finance/tools.py

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv  
import sqlite3
import os

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# --- Setup DB ---
# ✅ Use correct path based on your system
DB_PATH = r"db/erp.db"
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"❌ Database not found at: {DB_PATH}")

DB_URI = f"sqlite:///{DB_PATH.replace(os.sep, '/')}"  # Convert \ → /
db = SQLDatabase.from_uri(DB_URI)

# --- RAG: Policy Lookup ---
embedding_model = OpenAIEmbeddings(api_key=api_key)  # ✅ Pass API key here
vectorstore = Chroma(
    persist_directory="data/chroma_db_openai",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

from rag.tools_rag import rag_search_tool, finance_rag

@tool
def policy_rag_tool(query: str) -> str:
    """Searches finance policy documents for rules."""
    return finance_rag(query)

@tool
def finance_sql_read(query: str) -> str:
    """Reads from finance tables: invoices, payments, ledger."""
    try:
        return db.run(query)
    except Exception as e:
        return f"SQL Read Error: {e}"

@tool
def finance_sql_write(query: str) -> str:
    """Writes to ledger_entries and ledger_lines."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()
        return "✅ Ledger updated."
    except Exception as e:
        return f"❌ Failed to write: {e}"

@tool
def anomaly_detector_tool() -> str:
    """Detects suspicious invoices using ML."""
    try:
        from agents_finance_ML import train_anomaly_model, detect_anomalies
        model = train_anomaly_model(DB_PATH)
        anomalies = detect_anomalies(DB_PATH, model)
        if not anomalies:
            return "No anomalies detected."
        return f"🚨 Suspicious invoices detected: {anomalies[:5]}"
    except Exception as e:
        return f"❌ Anomaly detection failed: {e}"

@tool
def flag_invoice_for_approval(invoice_number: str, reason: str) -> str:
    """Creates a pending approval for a suspicious invoice."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO approvals (module, payload_json, status, requested_by) 
            VALUES (?, ?, 'pending', ?)
        """, ("finance", f'{{"invoice": "{invoice_number}", "reason": "{reason}"}}', "system"))
        conn.commit()
        conn.close()
        return f"✅ Invoice {invoice_number} flagged for approval: {reason}"
    except Exception as e:
        return f"❌ Failed to flag: {e}"

# --- All Tools ---
finance_tools = [
    policy_rag_tool,
    finance_sql_read,
    finance_sql_write,
    anomaly_detector_tool,
    flag_invoice_for_approval
]