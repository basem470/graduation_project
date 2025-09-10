# agents/finance/tools.py
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from agents_llm_setup import llm # env variable based LLM setup
from langchain_openai import OpenAIEmbeddings
import sqlite3
import os
from agents_finance_ML import train_anomaly_model, detect_anomalies # Import ML functions that i have created from the package

# --- Setup DB ---
db_path = os.path.abspath(r"D:\Sprints.ai-Assingment Group - Final Project\data\erp.db")
db_uri = f"sqlite:///{db_path}"
db = SQLDatabase.from_uri(db_uri)

# --- RAG: Policy Lookup ---
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="data/chroma_db_openai", embedding_function=embedding_model)
retriever = vectorstore.as_retriever()

@tool
def policy_rag_tool(query: str) -> str:
    """Searches finance policy documents for rules."""
    results = retriever.get_relevant_documents(query)
    return "\n\n".join([r.page_content for r in results]) if results else "No policy found."

# --- SQL Tool: Read Invoices ---
@tool
def finance_sql_read(query: str) -> str:
    """Reads from finance tables: invoices, payments, ledger."""
    return db.run(query)

# --- SQL Tool: Write to Ledger ---
@tool
def finance_sql_write(query: str) -> str:
    """Writes to ledger_entries and ledger_lines."""
    try:
        from sqlite3 import connect
        conn = connect(db_path.replace("sqlite:///", ""))
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()
        return "✅ Ledger updated."
    except Exception as e:
        return f"❌ Failed to write: {e}"

# --- Anomaly Detection Tool ---
@tool
def anomaly_detector_tool() -> str:
    """Detects suspicious invoices using ML."""
    model = train_anomaly_model(db_path)
    anomalies = detect_anomalies(db_path, model)
    if not anomalies:
        return "No anomalies detected."
    return f"🚨 Suspicious invoices detected: {anomalies[:3]}"

# --- Flag Invoice for Approval ---
@tool
def flag_invoice_for_approval(invoice_number: str, reason: str) -> str:
    """Creates a pending approval for a suspicious invoice."""
    try:
        from sqlite3 import connect
        conn = connect(db_path.replace("sqlite:///", ""))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO approvals (module, payload_json, status, requested_by) VALUES (?, ?, 'pending', ?)",
            ("finance", f'{{"invoice": "{invoice_number}", "reason": "{reason}"}}', "system")
        )
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