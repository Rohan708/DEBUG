import sqlite3

# Create the `CodeSnippets` Table in SQLite
def initialize_database(db_name="code_data.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CodeSnippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            label TEXT NOT NULL
        )
    ''')
    # Add some sample data for testing (optional)
    sample_data = [
        ("def add(a, b): return a b", "buggy"),  # Buggy example
        ("def add(a, b): return a + b", "bug-free")  # Bug-free example
    ]
    cursor.executemany('INSERT INTO CodeSnippets (code, label) VALUES (?, ?)', sample_data)
    conn.commit()
    conn.close()
    print("Database initialized and table created successfully!")

# Run this function to initialize the database
if __name__ == "__main__":
    initialize_database()