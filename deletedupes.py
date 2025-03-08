import sqlite3

def inspect_duplicates():
    """Inspect the database for duplicate titles."""
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    # Count total rows
    cursor.execute("SELECT COUNT(*) FROM movies")
    total = cursor.fetchone()[0]
    print(f"Total rows: {total}")
    
    # Find duplicates by title
    cursor.execute("""
        SELECT title, COUNT(*) as count 
        FROM movies 
        GROUP BY title 
        HAVING count > 1
    """)
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"Found {len(duplicates)} titles with duplicates:")
        for title, count in duplicates:
            print(f"Title: '{title}' appears {count} times")
    else:
        print("No duplicates found.")
    
    conn.close()
    return duplicates

def deduplicate_database():
    """Remove duplicates, keeping the first entry for each title."""
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    
    # Create a temporary table with unique titles (first occurrence)
    cursor.execute("""
        CREATE TABLE movies_temp AS 
        SELECT * FROM movies 
        WHERE id IN (
            SELECT MIN(id) 
            FROM movies 
            GROUP BY title
        )
    """)
    
    # Drop the old table and rename the temp table
    cursor.execute("DROP TABLE movies")
    cursor.execute("ALTER TABLE movies_temp RENAME TO movies")
    
    conn.commit()
    
    # Verify new total
    cursor.execute("SELECT COUNT(*) FROM movies")
    new_total = cursor.fetchone()[0]
    print(f"Database deduplicated. New total rows: {new_total}")
    
    conn.close()

def get_all_movies():
    """Print all movies for verification."""
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM movies")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()
    return len(rows)

if __name__ == "__main__":
    # Step 1: Inspect duplicates
    print("Inspecting duplicates...")
    duplicates = inspect_duplicates()
    
    # Step 2: Deduplicate if needed
    if duplicates:
        print("\nDeduplicating database...")
        deduplicate_database()
    else:
        print("\nNo deduplication needed.")
    
    # Step 3: Verify
    print("\nCurrent database contents:")
    total = get_all_movies()
    print(f"Total unique movies: {total}")