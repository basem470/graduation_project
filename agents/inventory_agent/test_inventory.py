# test_inventory_fixed.py
from agents_inventory_agent import inventory_agent, handle_inventory_query

# More realistic test queries
test_queries = [
    # 1. Basic stock check (should work)
    "Show me products that need reordering",
    
    # # 2. Simple forecast (use fallback)
    # "Predict demand for product SKU-0134 for next 30 days",
    
    # # 3. Stock level check
    # "What is the current stock level for product SKU-0134?",
    
    # 4. RAG search
    "What is the validity period of the proposal according to the supplier contracts?",
    
    # # 6. Simple query
    # "How does the inventory system work?",
]

print("🚀 Testing Fixed Inventory Agent\n")
print("=" * 80)

for i, query in enumerate(test_queries, 1):
    print(f"📦 Query {i}: {query}")
    try:
        response = handle_inventory_query(query)
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print("-" * 80)
    print()