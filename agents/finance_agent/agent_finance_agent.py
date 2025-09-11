# agents/finance/agent.py
from langgraph.prebuilt import create_react_agent
from agents_llm_setup import llm
from agents.finance_agent.agent_finance_tool import finance_tools

# --- Create Finance Agent ---
finance_agent = create_react_agent(
    model=llm,
    tools=finance_tools,
    prompt="You are a Finance Agent. Use tools to process invoices, update ledgers, detect anomalies, and enforce policies."
)