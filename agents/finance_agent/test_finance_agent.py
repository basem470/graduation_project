# test_finance.py
from agent_finance_agent import finance_agent

print("🚀 Testing Finance Agent...\n")

def run(query: str):
    print(f"\n💸 Query: {query}")
    try:
        result = finance_agent.invoke({
            "messages": [("user", query)]
        })
        print(f"🤖 Response: {result['messages'][-1].content}")
    except Exception as e:
        print(f"❌ Error: {e}")

queries = [
    "What is the refund policy?",
    "Is invoice INV-1001 suspicious?",
    "Are there any pending approvals?",
    "Get status of invoice INV-000174",
    "Flag invoice INV-1001 as suspicious due to duplicate billing"
]

print("🚀 Testing Finance Agent...\n")

for q in queries:
    run(q)