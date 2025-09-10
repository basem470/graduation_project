# agents/inventory/tools.py
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from agents_llm_setup import llm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import sqlite3
import os

# --- Setup DB ---
db_path = os.path.abspath(r"db/erp.db")
db_uri = f"sqlite:///{db_path}"
db = SQLDatabase.from_uri(db_uri)

# --- Tool 1: Get Low-Stock Items ---
@tool
def get_low_stock_items() -> str:
    """Returns items where qty_on_hand < reorder_point."""
    query = """
    SELECT p.sku, p.name, s.qty_on_hand, s.reorder_point
    FROM stock s
    JOIN products p ON s.product_id = p.id
    WHERE s.qty_on_hand < s.reorder_point
    ORDER BY s.qty_on_hand ASC
    """
    return db.run(query)

# --- Tool 2: Create Purchase Order ---
@tool
def create_purchase_order(supplier_name: str, product_sku: str, quantity: int) -> str:
    """Creates a new PO for a supplier."""
    try:
        conn = sqlite3.connect(db_path.replace("sqlite:///", ""))
        cursor = conn.cursor()
        
        # Get supplier_id
        cursor.execute("SELECT id FROM suppliers WHERE name = ?", (supplier_name,))
        supplier_row = cursor.fetchone()
        if not supplier_row:
            return f"❌ Supplier {supplier_name} not found."
        supplier_id = supplier_row[0]

        # Get product_id
        cursor.execute("SELECT id FROM products WHERE sku = ?", (product_sku,))
        product_row = cursor.fetchone()
        if not product_row:
            return f"❌ Product {product_sku} not found."
        product_id = product_row[0]

        # Insert PO
        cursor.execute(
            "INSERT INTO purchase_orders (supplier_id, status) VALUES (?, 'draft')",
            (supplier_id,)
        )
        po_id = cursor.lastrowid
        
        # Get unit cost
        cursor.execute(
            "SELECT default_cost FROM supplier_products WHERE supplier_id = ? AND product_id = ?",
            (supplier_id, product_id)
        )
        cost_row = cursor.fetchone()
        unit_cost = cost_row[0] if cost_row else 0.0

        cursor.execute(
            "INSERT INTO po_items (po_id, product_id, quantity, unit_cost) VALUES (?, ?, ?, ?)",
            (po_id, product_id, quantity, unit_cost)
        )
        
        conn.commit()
        conn.close()
        return f"✅ PO #{po_id} created for {quantity}x {product_sku} from {supplier_name}. Unit cost: ${unit_cost:.2f}"
    except Exception as e:
        return f"❌ Failed to create PO: {e}"

# --- Tool 3: Receive Goods ---
@tool
def receive_po(po_id: int, product_sku: str, received_qty: int) -> str:
    """Records received goods into inventory."""
    try:
        conn = sqlite3.connect(db_path.replace("sqlite:///", ""))
        cursor = conn.cursor()

        # Get product_id
        cursor.execute("SELECT id FROM products WHERE sku = ?", (product_sku,))
        product_row = cursor.fetchone()
        if not product_row:
            return f"❌ Product {product_sku} not found."
        product_id = product_row[0]

        # Verify PO exists and has this product
        cursor.execute(
            "SELECT 1 FROM po_items WHERE po_id = ? AND product_id = ?",
            (po_id, product_id)
        )
        if not cursor.fetchone():
            return f"❌ PO #{po_id} does not contain product {product_sku}"

        # Insert receipt
        cursor.execute(
            "INSERT INTO po_receipts (po_id, product_id, received_qty, received_at) VALUES (?, ?, ?, datetime('now'))",
            (po_id, product_id, received_qty)
        )

        # Update stock
        cursor.execute(
            "UPDATE stock SET qty_on_hand = qty_on_hand + ? WHERE product_id = ?",
            (received_qty, product_id)
        )

        # Update PO status if all items received
        cursor.execute(
            """UPDATE purchase_orders SET status = 'received' 
            WHERE id = ? AND NOT EXISTS (
                SELECT 1 FROM po_items poi 
                LEFT JOIN (
                    SELECT po_id, product_id, SUM(received_qty) as total_received
                    FROM po_receipts GROUP BY po_id, product_id
                ) pr ON poi.po_id = pr.po_id AND poi.product_id = pr.product_id
                WHERE poi.po_id = ? AND poi.quantity > COALESCE(pr.total_received, 0)
            )""",
            (po_id, po_id)
        )

        conn.commit()
        conn.close()
        return f"✅ Received {received_qty} units of {product_sku} for PO #{po_id}. Stock updated."
    except Exception as e:
        return f"❌ Failed to receive PO: {e}"
# --- Tool 4: Document RAG Search ---
# --- Tool 4: Document RAG Search (FIXED) ---
@tool
def doc_rag_tool(query: str) -> str:
    """Searches supplier contracts and policies."""
    try:
        # Import correct modules
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ OPENAI_API_KEY not found in .env file"

        # Paths
        DOCS_DIR = os.path.abspath(r"data/docs/contracts")
        PERSIST_DIR = os.path.abspath(r"data/chroma_db_inventory")

        # Initialize embeddings
        embedding_model = OpenAIEmbeddings(api_key=api_key)

        # Check if vector store already exists
        if os.path.exists(PERSIST_DIR) and any(os.scandir(PERSIST_DIR)):
            print("✅ Loading existing inventory vector store...")
            retriever = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embedding_model
            ).as_retriever(search_kwargs={"k": 3})
        else:
            print("🔄 Creating new vector store for inventory docs...")

            # Create list of documents
            documents = []
            if not os.path.exists(DOCS_DIR):
                return f"❌ Documents directory not found: {DOCS_DIR}"

            for filename in os.listdir(DOCS_DIR):
                filepath = os.path.join(DOCS_DIR, filename)
                if not os.path.isfile(filepath):
                    continue
                try:
                    if filename.endswith(".pdf"):
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(filepath)
                        docs = loader.load()
                    elif filename.endswith(".txt") or filename.endswith(".md"):
                        from langchain_community.document_loaders import TextLoader
                        loader = TextLoader(filepath, encoding='utf-8')
                        docs = loader.load()
                    else:
                        continue

                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            "source": filename,
                            "category": "contract",
                            "agent": "inventory"
                        })
                    documents.extend(docs)
                except Exception as e:
                    print(f"⚠️ Failed to load {filename}: {e}")

            if not documents:
                return "No valid documents found in contracts folder."

            # Split and embed
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)

            # Create vector store
            vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=embedding_model,
                persist_directory=PERSIST_DIR
            )
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Perform retrieval
        results = retriever.invoke(query)
        if not results:
            return "No relevant policy or contract found."

        # Format response
        context = "\n\n".join([
            f"📄 [{r.metadata.get('source', 'unknown')}]\n{r.page_content}"
            for r in results
        ])
        return f"Found {len(results)} relevant document(s):\n\n{context}"

    except ImportError as e:
        return f"❌ Missing package: {e.name}. Run: pip install {e.name}"
    except Exception as e:
        return f"❌ RAG Tool Error: {str(e)}"

# --- Tool 5: Get Product Stock Level ---
@tool
def get_product_stock(sku: str) -> str:
    """Returns current stock level for a specific product."""
    query = f"""
    SELECT p.sku, p.name, s.qty_on_hand, s.reorder_point
    FROM stock s
    JOIN products p ON s.product_id = p.id
    WHERE p.sku = '{sku}'
    """
    result = db.run(query)
    if "no rows" in result.lower():
        return f"Product {sku} not found in inventory."
    return result
@tool
def forecast_demand(sku: str, days: int = 30) -> str:
    """Predicts demand for a product."""
    # In demo, return hardcoded value
    if sku.upper() == "SKU-0134":
        return f"Predicted demand for {sku} over {days} days: 85 units"
    return f"Forecasting not available for {sku}"

# --- All Tools ---
inventory_tools = [
    get_low_stock_items,
    create_purchase_order,
    receive_po,
    doc_rag_tool,
    get_product_stock,
    forecast_demand
]