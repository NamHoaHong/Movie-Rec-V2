import sqlite3

def setup_database():
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            summary TEXT,
            genre TEXT
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    setup_database()