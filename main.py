# agents/router_agent/router_agent.py

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from agents_llm_setup import llm
from dotenv import load_dotenv
import os

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# Load environment variables
load_dotenv()

# -------------------------------
# 1. LLM Setup
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------
# 2. Import All Agents (Fixed Paths)
# -------------------------------
try:
    from agents.sales_agent.salesAgent import SalesReactAgent
    sales_agent_instance = SalesReactAgent()
except Exception as e:
    print(f"⚠️ Failed to load Sales Agent: {e}")
    sales_agent_instance = None

try:
    from agents.finance_agent.agent_finance_agent import finance_agent
    print("✅ Finance Agent loaded")
except Exception as e:
    print(f"⚠️ Failed to load Finance Agent: {e}")
    finance_agent = None

try:
    from agents.inventory_agent.agents_inventory_agent import inventory_agent
    print("✅ Inventory Agent loaded")
except Exception as e:
    print(f"⚠️ Failed to load Inventory Agent: {e}")
    inventory_agent = None

try:
    from agents.analytics_agent.analyticsAgent import AnalyticsReActAgent
    print("✅ Analytics Agent loaded")
except Exception as e:
    print(f"⚠️ Failed to load Analytics Agent: {e}")
    analytics_agent = None

# -------------------------------
# 3. Intent Classification Tool
# -------------------------------
def classify_intent(query: str) -> str:
    """Classify query into the right agent."""
    q = query.lower()
    if any(k in q for k in ["customer", "order", "lead", "ticket"]):
        return "SALES_AGENT"
    elif any(k in q for k in ["invoice", "payment", "ledger", "refund", "suspicious"]):
        return "FINANCE_AGENT"
    elif any(k in q for k in ["stock", "inventory", "supplier", "po", "purchase order"]):
        return "INVENTORY_AGENT"
    elif any(k in q for k in ["revenue", "kpi", "forecast", "drop", "trend", "why did"]):
        return "ANALYTICS_AGENT"
    else:
        return "SALES_AGENT"

intent_tool = Tool(
    name="Intent Classifier",
    func=classify_intent,
    description="Use this first to route the query to the correct agent."
)

# -------------------------------
# 4. SQL Tool (for approvals, messages)
# -------------------------------
try:
    from langchain_community.utilities import SQLDatabase
    DB_PATH = r"db/erp.db"
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ Database not found at: {DB_PATH}")

    DB_URI = f"sqlite:///{DB_PATH.replace(os.sep, '/')}"
    db = SQLDatabase.from_uri(DB_URI)

    def run_sql(query: str) -> str:
        try:
            return db.run(query)
        except Exception as e:
            return f"SQL Error: {e}"

    sql_tool = Tool(
        name="RouterSQL",
        func=run_sql,
        description="Use this to check approvals, messages, or statuses."
    )
except Exception as e:
    print(f"⚠️ SQL Tool failed: {e}")
    sql_tool = Tool(
        name="RouterSQL",
        func=lambda x: "Database currently unavailable.",
        description="Fallback SQL tool"
    )

# -------------------------------
# 5. Create Router Agent (FIXED: Use prompt= not state_modifier)
# -------------------------------
tools = [intent_tool, sql_tool]

system_prompt = """
You are the <Router Agent> for Helios Dynamics ERP.
Your job is to:
1. Use 'Intent Classifier' to route queries to the right agent.
2. If needed, use 'RouterSQL' to check pending approvals, messages, or statuses.
3. Never guess — always use tools.

Available agents:
- SALES_AGENT: Customers, orders, leads, tickets
- FINANCE_AGENT: Invoices, payments, fraud detection, ledger
- INVENTORY_AGENT: Stock levels, purchase orders, supplier contracts
- ANALYTICS_AGENT: Revenue trends, KPIs, forecasting, reports

Always respond clearly with routing decisions or direct answers.
"""

router_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt  # ✅ Use 'prompt', NOT 'state_modifier'
)