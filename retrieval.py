import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import spacy

# Load the SentenceTransformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def load_movies():
    """Load all movies from the SQLite database."""
    conn = sqlite3.connect('movies.db')
    cursor = conn.cursor()
    cursor.execute("SELECT title, summary, genre FROM movies")
    movies = [{'title': row[0], 'summary': row[1], 'genre': row[2]} for row in cursor.fetchall()]
    conn.close()
    return movies

def build_faiss_index(movies):
    """Build a FAISS index from movie summary embeddings."""
    # Extract summaries, replacing None with a default string
    summaries = [movie['summary'] if movie['summary'] is not None else "No description available" 
                 for movie in movies]
    
    # Generate embeddings
    embeddings = model.encode(summaries, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]  # Embedding size (e.g., 384 for MiniLM)
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings)  # Add embeddings to index
    
    return index, embeddings, movies

def extract_genres_and_adjectives(query):
    genres = ["Action", "Comedy", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Thriller", "Western"]
    
    # Identify genres in query
    matched_genres = [g for g in genres if g.lower() in query.lower()]
    
    # Extract adjectives using spaCy
    doc = nlp(query)
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]
    
    return matched_genres, adjectives

def fetch_movies_from_db(db_path: str = "movies.db") -> list[dict]:
    """Fetch all movies from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT title, summary, genre FROM movies")
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to list of dicts
    movies = [{"title": row[0], "summary": row[1], "genre": row[2]} for row in rows]
    return movies

def preprocess_and_retrieve(query: str, index, movies: list[dict], db_path: str = "movies.db") -> tuple[dict, float]:
    """
    Retrieve one movie: filter by genre first, then weight adjectives if matches exist;
    otherwise, use adjectives only.
    
    Args:
        query (str): User query, e.g., "exciting action movies"
        index: FAISS index
        movies (list[dict]): List of movie dictionaries
        db_path (str): Path to SQLite database
    
    Returns:
        tuple[dict, float]: Best matching movie and its distance
    """
    # Extract genres and adjectives
    matched_genres, adjectives = extract_genres_and_adjectives(query)
    
    # Fetch all movies from DB
    all_movies = fetch_movies_from_db(db_path)
    
    # Step 1: Filter movies by matched genres
    genre_filtered_movies = []
    genre_filtered_indices = []
    for idx, movie in enumerate(all_movies):
        genres = [g.strip() for g in movie["genre"].split(",")]
        if any(g.capitalize() in genres for g in matched_genres):  # Match any genre
            genre_filtered_movies.append(movie)
            genre_filtered_indices.append(idx)
    
    if genre_filtered_movies:
        # Step 2: Genre matches exist, prioritize adjectives among filtered movies
        print(f"\nFound {len(genre_filtered_movies)} movies with genres: {matched_genres}")
        # Create a query focused on adjectives
        adjective_query = " ".join(adjectives) if adjectives else query
        query_embedding = model.encode([adjective_query], convert_to_numpy=True)
        
        # Subset the FAISS index for genre-matched movies
        filtered_embeddings = np.array([index.reconstruct(i) for i in genre_filtered_indices])
        distances, indices = index.search(query_embedding, k=1)  # Top 1 match
        
        best_local_index = indices[0][0]
        best_global_index = genre_filtered_indices[best_local_index]
        best_movie = movies[best_global_index]
        distance = distances[0][0]
    else:
        # Step 3: No genre matches, use adjectives only on all movies
        print(f"\nNo movies found with genres: {matched_genres}, using adjectives only")
        adjective_query = " ".join(adjectives) if adjectives else query
        query_embedding = model.encode([adjective_query], convert_to_numpy=True)
        
        # Search full index
        distances, indices = index.search(query_embedding, k=1)  # Top 1 match
        best_index = indices[0][0]
        best_movie = movies[best_index]
        distance = distances[0][0]
    
    return best_movie, distance

if __name__ == "__main__":
    # Load movies from database
    movies = load_movies()
    print(f"Loaded {len(movies)} movies from database.")

    # Debug: Check for None summaries
    none_summaries = [m for m in movies if m['summary'] is None]
    if none_summaries:
        print(f"Found {len(none_summaries)} movies with None summaries:")
        for m in none_summaries:
            print(f"Title: {m['title']}, Genre: {m['genre']}")

    # Build FAISS index
    index, embeddings, movies = build_faiss_index(movies)
    print(f"FAISS index built with {index.ntotal} embeddings.")

    # Test retrieval with sample queries
    test_queries = [
        "happy movie",
        "scary film",
        "exciting action movie",
        "sad drama"
    ]
    
    for query in test_queries:
        movie, distance = preprocess_and_retrieve(query, index, movies)
        print(f"Query: '{query}'")
        print(f"Retrieved Movie: {movie['title']}")
        print(f"Summary: {movie['summary']}")
        print(f"Genre: {movie['genre']}")
        print(f"Distance: {distance:.4f}")  # Lower distance = better match