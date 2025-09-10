# tools/rag.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import sqlite3
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ROOT = r"/"
DOCS_DIR = "data/docs"
PERSIST_DIR = "data/finance_agent_rag/"


# --- Load OpenAI API Key ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

print("✅ OPENAI_API_KEY loaded")

class RAGT:
    def __init__(self, collection_name: str = "erp_documents"):
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small"
        )
        self.collection_name = collection_name
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.init_vectorstore()
    
    def init_vectorstore(self):
        """Initialize or load the vector store"""
        os.makedirs(PERSIST_DIR, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=PERSIST_DIR
        )
    
    def load_documents_from_filesystem(self):
        """Load documents from the specified file paths"""
        documents = []
        
        # Define the specific files you want to process
        target_files = [
            r"data\docs\policies\refund_policy.pdf"
        ]
        
        for file_path in target_files:
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue
            
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            "source": "file",
                            "file_path": file_path,
                            "file_type": "pdf",
                            "category": self.get_category_from_path(file_path)
                        })
                    documents.extend(docs)
                    print(f"✅ Loaded PDF: {file_path}")
                
                elif file_path.endswith('.md'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            "source": "file",
                            "file_path": file_path,
                            "file_type": "markdown",
                            "category": self.get_category_from_path(file_path)
                        })
                    documents.extend(docs)
                    print(f"✅ Loaded Markdown: {file_path}")
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        return documents
    
    def get_category_from_path(self, file_path: str) -> str:
        """Extract category from file path"""
        if "contracts" in file_path:
            return "contracts"
        elif "metrics" in file_path:
            return "metrics"
        elif "policies" in file_path:
            return "policies"
        return "general"
    
    def load_glossary_from_db(self):
        """Load glossary terms from SQLite database"""
        conn = sqlite3.connect(os.path.join("db", "erp.db"))
        cursor = conn.cursor()
        
        documents = []
        
        try:
            cursor.execute("SELECT term, definition, module FROM glossary")
            for term, definition, module in cursor.fetchall():
                text = f"Term: {term}\nDefinition: {definition}\nModule: {module}"
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": "glossary",
                        "term": term,
                        "module": module,
                        "category": "glossary"
                    }
                ))
            print(f"✅ Loaded {len(documents)} glossary terms from database")
        except Exception as e:
            print(f"❌ Error loading glossary from DB: {e}")
        finally:
            conn.close()
        
        return documents
    
    def initialize_vectorstore(self):
        """Initialize the vector store with all documents"""
        print("🔄 Initializing vector store...")
        
        all_documents = []
        
        # Load from filesystem
        file_docs = self.load_documents_from_filesystem()
        all_documents.extend(file_docs)
        
        # Load from database glossary
        glossary_docs = self.load_glossary_from_db()
        all_documents.extend(glossary_docs)
        
        if not all_documents:
            print("⚠️ No documents found to load")
            return 0
        
        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(all_documents)
        
        # Add to vector store
        self.vectorstore.add_documents(split_docs)
        
        print(f"✅ Added {len(split_docs)} document chunks to vector store")
        return len(split_docs)
    
    def search(self, query: str, k: int = 5, filter_metadata: dict = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using OpenAI embeddings"""
        if self.vectorstore is None:
            self.init_vectorstore()
        
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata
            )
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get('source', 'unknown'),
                    "category": doc.metadata.get('category', 'general')
                })
            
            return formatted_results
        except Exception as e:
            print(f"❌ Error in RAG search: {e}")
            return []
    
    def search_contracts(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search specifically in contracts"""
        return self.search(
            query=query,
            k=k,
            filter_metadata={"category": "contracts"}
        )
    
    def search_policies(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search specifically in policies"""
        return self.search(
            query=query,
            k=k,
            filter_metadata={"category": "policies"}
        )
    
    def search_metrics(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search specifically in metrics"""
        return self.search(
            query=query,
            k=k,
            filter_metadata={"category": "metrics"}
        )
    
    def search_glossary(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search specifically in glossary"""
        return self.search(
            query=query,
            k=k,
            filter_metadata={"category": "glossary"}
        )

# Global RAG instance
rag_tool = RAGT()

def rag_search_tool(query: str, category: str = "all", k: int = 3) -> str:
    """
    RAG search tool for agents to use with OpenAI embeddings
    
    Args:
        query: Search query
        category: "contracts", "policies", "metrics", "glossary", or "all"
        k: Number of results to return
    
    Returns:
        Formatted search results
    """
    try:
        if category == "contracts":
            results = rag_tool.search_contracts(query, k)
        elif category == "policies":
            results = rag_tool.search_policies(query, k)
        elif category == "metrics":
            results = rag_tool.search_metrics(query, k)
        elif category == "glossary":
            results = rag_tool.search_glossary(query, k)
        else:
            results = rag_tool.search(query, k)
        
        if not results:
            return "No relevant documents found for your query."
        
        output = [f"🔍 Found {len(results)} results for: '{query}'"]
        output.append("")
        
        for i, result in enumerate(results, 1):
            output.append(f"📄 Result {i}:")
            output.append(f"   Category: {result.get('category', 'unknown')}")
            output.append(f"   Source: {result.get('source', 'unknown')}")
            
            # Truncate content for readability
            content = result['content']
            if len(content) > 300:
                content = content[:300] + "..."
            
            output.append(f"   Content: {content}")
            
            # Show file path if available
            file_path = result['metadata'].get('file_path')
            if file_path:
                output.append(f"   File: {os.path.basename(file_path)}")
            
            output.append("")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Error in RAG search: {str(e)}"

def initialize_rag_system():
    """Initialize the complete RAG system"""
    print("🚀 Initializing RAG System with OpenAI Embeddings")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # Create subdirectories for your documents
    for subdir in ["contracts", "policies", "metrics"]:
        os.makedirs(os.path.join(DOCS_DIR, subdir), exist_ok=True)
    
    # Initialize vector store with documents
    total_loaded = rag_tool.initialize_vectorstore()
    
    print("=" * 60)
    print(f"✅ RAG System Initialized with {total_loaded} document chunks")
    return total_loaded
# tools/rag.py (add these functions at the end)
def smart_rag_search(query: str, agent_type: str = "general") -> str:
    """
    Smart RAG search that handles empty/generic queries intelligently
    """
    # Default queries for different agent types
    default_queries = {
        "inventory": "supplier contract terms delivery payment penalties lead time",
        "finance": "finance policies accounting procedures audit compliance refund",
        "sales": "sales policies customer agreements pricing terms commission",
        "general": "company policies procedures guidelines terms conditions"
    }
    
    # Use appropriate default if query is empty or too generic
    if not query.strip() or len(query.strip()) < 3:
        default_query = default_queries.get(agent_type, default_queries["general"])
        query = default_query
    
    # Determine appropriate categories based on agent type
    category_priority = {
        "inventory": ["contracts", "policies", "all"],
        "finance": ["policies", "glossary", "all"],
        "sales": ["policies", "contracts", "all"],
        "general": ["all", "policies", "glossary"]
    }
    
    categories = category_priority.get(agent_type, ["all"])
    
    # Try each category until we get results
    for category in categories:
        results = rag_search_tool(query, category=category, k=3)
        if "no relevant documents" not in results.lower():
            return results
    
    return f"No relevant documents found for '{query}'. Please try a different search term or ensure relevant documents are uploaded."

# Helper function for finance agent  
def finance_rag(query: str) -> str:
    return smart_rag_search(query, "finance")



if __name__ == "__main__":
    initialize_rag_system()
    
    while True:
        query = input("Enter your query: ")
        print(smart_rag_search(query, "general"))
        print("-" * 60)