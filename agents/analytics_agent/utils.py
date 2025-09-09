
import sqlite3
import os

def seed_db():
    query = """INSERT or ignore INTO glossary (term, definition, module) VALUES
('ARR (Annual Recurring Revenue)', 'Predictable revenue generated from subscriptions or long-term contracts over a year.', 'Finance'),
('Lead-to-Cash', 'The end-to-end process from a potential customer''s initial interest to receiving payment for a sale.', 'Sales'),
('SKU (Stock Keeping Unit)', 'A unique identifier for a specific product, allowing it to be tracked for inventory purposes.', 'Operations'),
('BOM (Bill of Materials)', 'A comprehensive list of all raw materials, components, and sub-assemblies required to manufacture a product.', 'Manufacturing'),
('CRM (Customer Relationship Management)', 'A system for managing interactions with current and prospective customers.', 'Sales'),
('Demand Forecasting', 'The process of predicting future customer demand to inform production and inventory decisions.', 'Supply Chain'),
('LTV (Lifetime Value)', 'The total revenue a company can expect from a single customer account throughout their relationship.', 'Sales'),
('EBITDA', 'Earnings Before Interest, Taxes, Depreciation, and Amortization; a measure of a company''s financial performance.', 'Finance'),
('WIP (Work in Progress)', 'Partially finished goods that are still in the manufacturing process.', 'Operations'),
('KPI (Key Performance Indicator)', 'A quantifiable measure used to track and analyze a company''s progress towards its business goals.', 'Strategy'),
('SCM (Supply Chain Management)', 'The management of the flow of goods and services, including all processes that transform raw materials into final products.', 'Supply Chain'),
('Net Promoter Score (NPS)', 'A metric used to gauge customer loyalty and satisfaction by asking how likely they are to recommend the company.', 'Customer Service'),
('Unit Economics', 'The revenue and costs associated with a company''s business model on a per-unit basis.', 'Finance'),
('COGS (Cost of Goods Sold)', 'The direct costs attributable to the production of the goods sold by a company.', 'Finance'),
('Customer Churn', 'The rate at which customers stop doing business with a company over a given period.', 'Sales'),
('CAPEX (Capital Expenditure)', 'Funds used by a company to acquire, upgrade, and maintain physical assets such as buildings and machinery.', 'Finance'),
('A/P (Accounts Payable)', 'Money owed by a company to its suppliers for goods or services purchased on credit.', 'Finance'),
('A/R (Accounts Receivable)', 'Money owed to a company by its customers for goods or services delivered.', 'Finance'),
('Kanban', 'A visual system for managing workflow that originated in lean manufacturing.', 'Operations'),
('ERP (Enterprise Resource Planning)', 'A system that integrates all aspects of a company’s operations into a single database and platform.', 'IT/Operations');"""
    DB_PATH = os.getenv("DB_PATH", "db/erp.db")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(query)
    conn.commit()
    conn.close()
    