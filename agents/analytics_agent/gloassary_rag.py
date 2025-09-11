import os 
import sys
from typing import List

# Get the path to the current file's directory 
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two directories to reach the project root 
project_root = os.path.dirname(os.path.dirname(current_dir))
# Add the project root to the Python path 
sys.path.insert(0, project_root)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from lib.build_llm import build_llm

# --- Configuration ---
PERSIST_DIRECTORY = "agents/analytics_agent/rag/"
METRICS_FILE_PATH = "data/docs/metrics/definitions.md"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def inspect_source_file(file_path: str) -> None:
    """
    Inspects the source file to understand what content is available.
    """
    print(f"\n--- Inspecting Source File: {file_path} ---")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        print(f"File size: {len(content)} characters")
        print(f"First 500 characters:")
        print(content[:500])
        print("...")
        print(f"Last 500 characters:")
        print(content[-500:])
        
        # Look for specific terms
        terms_to_check = ["KPI", "CAD", "payable", "power of attorney"]
        print(f"\nChecking for specific terms:")
        for term in terms_to_check:
            count = content.upper().count(term.upper())
            print(f"  {term}: {count} occurrences")
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    print("-" * 50)


def embed_once(file_path: str, force_recreate: bool = False) -> None:
    """
    Embeds the markdown file into ChromaDB.
    
    Args:
        file_path: Path to the source file
        force_recreate: If True, recreates the database even if it exists
    """
    if os.path.exists(PERSIST_DIRECTORY) and not force_recreate:
        print(f"Vector database already exists at '{PERSIST_DIRECTORY}'. Skipping embedding.")
        return

    if force_recreate and os.path.exists(PERSIST_DIRECTORY):
        import shutil
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f"Removed existing database at '{PERSIST_DIRECTORY}'.")

    print("Creating embeddings...")

    # Load the document content
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    if not raw_text.strip():
        print("Error: The source file is empty.")
        return

    # Split into chunks with better parameters for definitions
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Smaller chunks for definitions
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for definitions
    )
    documents = text_splitter.create_documents([raw_text])
    print(f"Split document into {len(documents)} chunks.")

    # Show some sample chunks
    print("Sample chunks:")
    for i, doc in enumerate(documents[:3]):
        print(f"Chunk {i+1}: {doc.page_content[:100]}...")

    # Initialize embeddings
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    print(f"Embedding model '{OLLAMA_EMBED_MODEL}' initialized.")

    # Create ChromaDB
    try:
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        db.persist()
        print("Embedding complete. Vector DB stored.")
    except Exception as e:
        print(f"Error creating embeddings: {e}")


def test_retrieval_detailed(query: str, k: int = 5) -> List[Document]:
    """
    Tests the retrieval process with more detailed output.
    """
    print(f"\n--- Detailed Retrieval Test for Query: '{query}' ---")

    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

        # Get all documents to understand what's in the database
        all_docs = db.get()
        print(f"Total documents in database: {len(all_docs['documents']) if all_docs else 0}")

        # Perform similarity search
        retrieved_docs = db.similarity_search_with_score(query, k=k)

        print(f"Found {len(retrieved_docs)} relevant documents with scores:")
        for i, (doc, score) in enumerate(retrieved_docs):
            print(f"\nDocument {i+1} (Score: {score:.4f}):")
            print(f"Content: {doc.page_content}")
            print("-" * 30)
        
        return [doc for doc, score in retrieved_docs]
    
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []


def retrieve_and_generate(query: str) -> None:
    """
    Retrieves documents and uses an LLM to generate an answer with improved prompting.
    """
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

        # Get more documents and use score threshold
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.1}
        )
        
        llm = build_llm("gemini-2.5-flash")

        prompt_template = """
        You are an expert business analyst helping with metric definitions. 
        Use the following context to answer the user's question about business metrics and definitions.

        Context: {context}
        
        Question: {question}
        
        Instructions:
        - If you find relevant information in the context, provide a detailed answer
        - If the exact term isn't found but similar terms exist, mention those
        - If no relevant information is found, clearly state that the information isn't available in the knowledge base
        - Always be specific about what information you're basing your answer on

        Answer:
        """
        prompt = PromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"\nQuery: {query}")
        answer = rag_chain.invoke(query)
        print("Generated Answer:")
        print(answer)
        print("-" * 50)
        return answer

    except Exception as e:
        print(f"Error in retrieve_and_generate: {e}")


if __name__ == "__main__":
    print("=== RAG System Debugging and Testing ===")
    
    # Always inspect the source file first
    inspect_source_file(METRICS_FILE_PATH)
    
    # Force recreate the database to ensure fresh embeddings with your actual content
    print("\nRecreating embeddings from your actual definitions file...")
    embed_once(METRICS_FILE_PATH, force_recreate=False)

    # Test queries with detailed retrieval
    test_queries = [
        "What is the definition of KPI?",
        "What is CAD?", 
        "Explain payable",
        "What's a power of attorney",
    ]

    print("\n" + "="*60)
    print("DETAILED RETRIEVAL TESTING")
    print("="*60)

    for query in test_queries:
        test_retrieval_detailed(query)

    print("\n" + "="*60)
    print("RAG CHAIN TESTING")
    print("="*60)

    for query in test_queries:
        retrieve_and_generate(query)