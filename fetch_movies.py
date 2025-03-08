import requests
import sqlite3
import json
from db_setup import setup_database

RAPIDAPI_KEY = "86c0f51769mshea7c377a92d9068p1e25d1jsna7f3aa8555b2"
RAPIDAPI_HOST = "imdb236.p.rapidapi.com"

def fetch_most_popular_movies():
    url = "https://imdb236.p.rapidapi.com/imdb/most-popular-movies"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    response = requests.get(url, headers=headers)
    print(f"\nFetching full list from 'most-popular-movies' endpoint")
    print(f"Status: {response.status_code}")
    data = response.json()
    print("JSON Response (full list):")
    print(json.dumps(data, indent=2))  # Print the entire response
    if isinstance(data, list) and len(data) > 0:
        print(f"Got {len(data)} movies")
        return data  # Return the full list of movies
    else:
        print("No movie list found.")
        return []

def fetch_top250_movies():
    url = "https://imdb236.p.rapidapi.com/imdb/top250-movies"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    response = requests.get(url, headers=headers)
    print(f"\nFetching full list from 'most-popular-movies' endpoint")
    print(f"Status: {response.status_code}")
    data = response.json()
    print("JSON Response (full list):")
    print(json.dumps(data, indent=2))  # Print the entire response
    if isinstance(data, list) and len(data) > 0:
        print(f"Got {len(data)} movies")
        return data  # Return the full list of movies
    else:
        print("No movie list found.")
        return []

def save_to_database(movies):
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    new_movies = 0
    for movie in movies:
        if movie is None:  # Skip if fetch failed
            continue
        title = movie.get("primaryTitle", "No title")
        summary = movie.get("description") or "No description"
        genres = movie.get("genres", [])
        genre = ", ".join(genres) if isinstance(genres, list) else "Unknown"
        cursor.execute("SELECT COUNT(*) FROM movies WHERE title = ?", (title,))
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO movies (title, summary, genre)
                VALUES (?, ?, ?)
            ''', (title, summary, genre))
            new_movies += 1
            print(f"Added to DB: '{title}'")
        else:
            print(f"Skipped duplicate: '{title}'")
    conn.commit()
    conn.close()
    return new_movies

def get_all_movies():
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM movies")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()
    return len(rows)

if __name__ == "__main__":
    setup_database()

    movies = fetch_most_popular_movies()

    movies2 = fetch_top250_movies()

    # Save to database
    new_count = save_to_database(movies)
    new_count2 = save_to_database(movies2)

    print(f"\nSaved {new_count} new movies to database.")
    print(f"\nSaved {new_count2} new movies to database.")

    # Print current database contents
    print("\nTotal movies in database:")
    total = get_all_movies()
    print(f"Database now contains {total} movies.")