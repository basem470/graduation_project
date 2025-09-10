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
@tool
def doc_rag_tool(query: str) -> str:
    """Searches supplier contracts and policies."""
    try:
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Paths
        DOCS_DIR = os.path.abspath(r"data/docs/contracts")
        PERSIST_DIR = os.path.abspath(r"data/inventory_chroma")

        # Load or create vector store
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=OpenAIEmbeddings())
        else:
            documents = []
            for filename in os.listdir(DOCS_DIR):
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(DOCS_DIR, filename))
                elif filename.endswith(".txt"):
                    loader = TextLoader(os.path.join(DOCS_DIR, filename))
                else:
                    continue
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectordb = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
            vectordb.persist()

        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        response = qa_chain.run(query)
        return response
    except Exception as e:
        return f"❌ RAG Tool Error: {e}"


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