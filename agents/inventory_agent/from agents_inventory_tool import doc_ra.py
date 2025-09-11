from agents_inventory_tool import doc_rag_tool

def test_doc_rag_tool():
    query = "What are the penalties for late delivery by a supplier?"
    print(f"Testing doc_rag_tool with query: {query}")
    result = doc_rag_tool(query)
    print("RAG Tool Result:")
    print(result)

if __name__ == "__main__":
    test_doc_rag_tool()