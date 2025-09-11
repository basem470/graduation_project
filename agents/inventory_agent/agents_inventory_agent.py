# agents/inventory_agent.py


from langgraph.prebuilt import create_react_agent
from agents_llm_setup import llm
from agents.inventory_agent.agents_inventory_tool import inventory_tools

# System prompt for inventory management
system_prompt = """You are an Inventory & Supply Chain Agent for Helios Dynamics ERP.

 YOUR RESPONSIBILITIES:
- Monitor stock levels and reorder points
- Forecast demand using ML models
- Create purchase orders with suppliers
- Receive goods and update inventory
- Enforce supplier contracts and policies
- Prevent stockouts and overstocking

 AVAILABLE TOOLS:
1. get_low_stock_items - Find items below reorder point
2. forecast_tool - Predict demand using PyTorch LSTM
3. create_purchase_order - Create new purchase orders
4. receive_po - Record received goods
5. doc_rag_tool - Search supplier contracts and policies
6. get_product_stock - Check specific product stock levels

 RULES:
- Always check stock levels before creating POs
- Use forecasting for demand planning
- Verify supplier terms using RAG search
- Maintain accurate inventory records
- Flag critical stock situations immediately"""

# --- Create Inventory Agent ---
inventory_agent = create_react_agent(
    model=llm,
    tools=inventory_tools,
    prompt=system_prompt
)

def handle_inventory_query(query: str) -> str:
    """Convenience function to handle inventory queries"""
    try:
        result = inventory_agent.invoke({
            "messages": [("user", query)]
        })
        return result["messages"][-1].content
    except Exception as e:
        return f"❌ Inventory Agent Error: {str(e)}"