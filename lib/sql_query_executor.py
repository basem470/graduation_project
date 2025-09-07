import sqlite3
import os


def sql_query_executor(query: str):
    """Execute a SQL query and return the results as a list of dictionaries."""
    db_path = os.getenv("DB_PATH", "db/erp.db")

    try:
        conn = sqlite3.connect(db_path)
        # Enable row factory to get column names
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        # Convert sqlite3.Row objects to dictionaries
        results_list = [dict(row) for row in results]
        conn.close()

        return results_list

    except Exception as e:
        if "conn" in locals():
            conn.close()
        raise e


if __name__ == "__main__":
    test_query = "SELECT name FROM sqlite_master WHERE type='table';"
    print(sql_query_executor(test_query))
