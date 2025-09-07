from typing import Any, Dict, List, Optional
import sqlite3, json, os
from enum import Enum
from langchain_core.tools import tool 
from lib.parse_value import parse_value

DB_PATH = os.getenv("DB_PATH", "db/erp.db")

# Read intents
class InvReadIntent(str, Enum):
    GET_STOCK_BY_PRODUCT_ID = "get_stock_by_product_id"
    GET_STOCK_BY_SKU = "get_stock_by_sku"
    GET_STOCK_BY_NAME = "get_stock_by_name"
    LIST_UNDER_REORDER = "list_under_reorder"
    LIST_PRODUCTS = "list_products"
    LIST_SUPPLIERS = "list_suppliers"
    GET_PO_BY_ID = "get_po_by_id"

# Write intents
class InvWriteIntent(str, Enum):
    CREATE_DRAFT_PO = "create_draft_po"

def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

# row list -> list[dict]
def _rows_to_dicts(rows):
    return [dict(r) for r in rows]

# READ TOOL
@tool("inventory_sql_read")
def inventory_sql_read(input_str: str) -> str:
    """
    Parse 'intent: <intent>, key: <value>, ...' run safe SELECTs over inventory tables.

    Valid read intents: 
    - get_stock_by_product_id       (requires: product_id)
    - get_stock_by_sku              (requires: sku)
    - get_stock_by_name             (requires: name)
    - list_under_reorder
    - list_products
    - list_suppliers
    - get_po_by_id                  (requires: po_id)

    Example:
    "intent: list_under_reorder"
    "intent: get_stock_by_sku, sku: SKU-0001"
    "intent: get_po_by_id, po_id: 41"
    """

    intent = parse_value(input_str, "intent")
    if not intent: 
        return "Missing 'intent'."
    
    try: 
        read_intent = InvReadIntent(intent)
    except ValueError:
        return f"Invalid intent {intent}. Valid: {[i.value for i in InvReadIntent]}"
    
    with _connect as con:
        cur = con.cursor()

        if read_intent == InvReadIntent.GET_STOCK_BY_PRODUCT_ID:
            product_id = parse_value(input_str, "product_id")
            if not product_id or not product_id.isdigit():
                return "Provide integer 'product_id'."
            cur.execute("""
                        SELECT p.id as product_id, p.sku, p.name, s.qty_on_hand, s.reorder_point, p.price, p.description
                        FROM products p
                        JOIN stock s ON p.id = s.product_id
                        WHERE p.id = ?
                        """, (int(product_id),))
            return json.dumps(_rows_to_dicts(cur.fetchall()), indent=2) or "No results."
        
        if read_intent == InvReadIntent.GET_STOCK_BY_SKU:
            sku = parse_value(input_str, "sku")
            if not sku:
                return "Provide 'sku'."
            cur.execute("""
                        SELECT p.id as product_id, p.sku, p.name, s.qty_on_hand, s.reorder_point, p.price, p.description
                        FROM products p
                        JOIN stock s ON p.id = s.product_id
                        WHERE p.sku = ?
                        """, (sku,))
            return json.dumps(_rows_to_dicts(cur.fetchall()), indent=2) or "No results."
        
        if read_intent == InvReadIntent.GET_STOCK_BY_NAME:
            name = parse_value(input_str, "name")
            if not name:
                return "Provide 'name'."
            cur.execute("""
                        SELECT p.id as product_id, p.sku, p.name, s.qty_on_hand, s.reorder_point, p.price, p.description
                        FROM products p
                        JOIN stock s ON p.id = s.product_id
                        WHERE p.name LIKE ?
                        """, (f"%{name}%",))
            return json.dumps(_rows_to_dicts(cur.fetchall()), indent=2) or "No results."
        
        if read_intent == InvReadIntent.LIST_UNDER_REORDER:
            cur.execute("""
                        SELECT p.id as product_id, p.sku, p.name, s.qty_on_hand, s.reorder_point
                        FROM products p
                        JOIN stock s ON p.id = s.product_id
                        WHERE s.qty_on_hand < s.reorder_point
                        ORDER BY (s.reorder_point - s.qty_on_hand) DESC
                        """)
            return json.dumps(_rows_to_dicts(cur.fetchall()), indent=2) or "No items under reorder."
        
        if read_intent == InvReadIntent.LIST_PRODUCTS:
            cur.execute("SELECT id, sku, name, price, description FROM products ORDER BY id")
            return json.dumps(_rows_to_dicts(cur.fetchall()), indent=2)
        
        if read_intent == InvReadIntent.LIST_SUPPLIERS:
            cur.execute("SELECT id, name, email, phone FROM suppliers ORDER BY id")
            return json.dumps(_rows_to_dicts(cur.fetchall()), indent=2)
        

        
