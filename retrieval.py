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

def preprocess_query(query):
    matched_genres, adjectives = extract_genres_and_adjectives(query)
    
    # Boost genres (repeat them 2 times)
    genre_boost = " ".join(matched_genres) * 2 if matched_genres else ""
    
    # Modified query
    modified_query = f"{query} {genre_boost}"
    
    return modified_query.strip()


def retrieve_movie(query, index, movies):
    """Retrieve the most relevant movie based on a query."""
    # Encode the query
    query = preprocess_query(query)
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search FAISS index for the top match
    distances, indices = index.search(query_embedding, k=1)  # k=1 for top 1 match
    
    # Get the best matching movie
    best_index = indices[0][0]
    return movies[best_index], distances[0][0]

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
        movie, distance = retrieve_movie(query, index, movies)
        print(f"\nQuery: '{query}'")
        print(f"Retrieved Movie: {movie['title']}")
        print(f"Summary: {movie['summary']}")
        print(f"Genre: {movie['genre']}")
        print(f"Distance: {distance:.4f}")  # Lower distance = better match