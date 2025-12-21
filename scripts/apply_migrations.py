import sqlite3
import os
from pathlib import Path

# Config
DB_PATH = Path(__file__).parent.parent / "chatrel_csm.db"
MIGRATIONS_DIR = Path(__file__).parent.parent / "database" / "migrations"

def apply_migrations():
    print(f"Applying migrations to {DB_PATH}...")
    
    if not DB_PATH.exists():
        print("Database not found, creating...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all SQL files
    migration_files = sorted(list(MIGRATIONS_DIR.glob("*.sql")))
    
    for sql_file in migration_files:
        print(f"Applying {sql_file.name}...")
        with open(sql_file, "r") as f:
            sql_script = f.read()
            try:
                cursor.executescript(sql_script)
                print(f"  -> Success")
            except Exception as e:
                print(f"  -> Failed: {e}")
    
    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    apply_migrations()
