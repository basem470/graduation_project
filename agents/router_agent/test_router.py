# test_router.py

from langchain_core.messages import HumanMessage
from agents.router_agent import router_agent  # Make sure router_agent.py is in agents/router.py

# List of realistic test queries
queries = [
    "Create a new order for John Doe with product A100",
    "Is invoice INV-1001 suspicious?",
    "Show me low-stock items",
    "Explain why revenue dropped last week",
    "Are there any pending approvals?"
]

print("🚀 Testing Router Agent\n")
print("=" * 80)

for q in queries:
    print(f"\n💬 Query: {q}")
    try:
        # Format input as list of messages
        inputs = {"messages": [HumanMessage(content=q)]}
        
        # Invoke the router agent
        response = router_agent.invoke(inputs)
        
        # Extract final answer
        final_output = response["messages"][-1].content
        
        # Print result
        print(f"🤖 Response: {final_output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 80)