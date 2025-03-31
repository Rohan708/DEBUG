import os
import re
import json
import sqlite3

# Step 1: Data Collection - Gather code snippets from a directory of files
def collect_code_snippets(directory):
    code_snippets = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):  # Example for Python files
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_snippets.append(f.read())
    return code_snippets

# Step 2: Preprocessing - Normalize and clean the code snippets
def preprocess_code(code_snippet):
    # Remove extra spaces and newlines
    code = re.sub(r'\s+', ' ', code)
    # Remove comments
    code = re.sub(r'#.*', '', code)
    # Optionally: Remove docstrings
    code = re.sub(r'(\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")', '', code)
    return code.strip()

# Step 3: Labeling - Create labels for buggy and bug-free code (Manual or predefined)
def label_data(code_snippets):
    labeled_data = []
    for code in code_snippets:
        # Example labeling: You can replace this with a manual labeling mechanism
        label = input(f"Label for this code snippet (buggy or bug-free):\n{code}\n> ")
        labeled_data.append({"code": code, "label": label})
    return labeled_data

# Step 4: Save Data - Store the labeled data in a SQLite database
def save_data_to_db(labeled_data, db_name='code_data.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CodeSnippets (
            id INTEGER PRIMARY KEY,
            code TEXT,
            label TEXT
        )
    ''')
    # Insert data
    for entry in labeled_data:
        cursor.execute('INSERT INTO CodeSnippets (code, label) VALUES (?, ?)', (entry['code'], entry['label']))
    conn.commit()
    conn.close()

# Main function to run the pipeline
def main():
    directory = 'path_to_code_directory'  # Change this to your code directory path
    print("Collecting code snippets...")
    code_snippets = collect_code_snippets(directory)
    
    print("Preprocessing code snippets...")
    processed_snippets = [preprocess_code(snippet) for snippet in code_snippets]
    
    print("Labeling code snippets...")
    labeled_data = label_data(processed_snippets)
    
    print("Saving data to database...")
    save_data_to_db(labeled_data)
    print("Data pipeline completed!")

if __name__ == "__main__":
    main()