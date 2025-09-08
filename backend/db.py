import sqlite3

DB_PATH = "db/erp.db" 

def sql_query_executor(query: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        raise Exception(str(e))

if __name__ == "__main__":
    phone = "+201955504107"
    query = f"SELECT * FROM customers WHERE phone LIKE '%{phone}%'"
    
    result = sql_query_executor(query)
    print(result)
