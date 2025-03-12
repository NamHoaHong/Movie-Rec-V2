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

def preprocess_and_retrieve(query: str, index, movies: list[dict], top_k: int = 1) -> tuple[dict, float]:
    # Extract genres and adjectives
    matched_genres, adjectives = extract_genres_and_adjectives(query)
    
    # Use the provided movies list
    all_movies = movies
    
    # Step 1: Filter by genre if applicable
    genre_filtered_movies = []
    genre_filtered_indices = []
    for idx, movie in enumerate(all_movies):
        genres = [g.strip() for g in movie["genre"].split(",")]
        if any(g.capitalize() in genres for g in matched_genres):
            genre_filtered_movies.append(movie)
            genre_filtered_indices.append(idx)
    
    if genre_filtered_movies:
        print(f"Found {len(genre_filtered_movies)} movies with genres: {matched_genres}")
        adjective_query = " ".join(adjectives) if adjectives else query
        query_embedding = model.encode([adjective_query])
        filtered_embs = np.array([index.reconstruct(i) for i in genre_filtered_indices])
        faiss_index = faiss.IndexFlatL2(filtered_embs.shape[1])
        faiss_index.add(filtered_embs)
        distances, indices = faiss_index.search(query_embedding, k=top_k)
        best_local_index = indices[0][0]
        best_global_index = genre_filtered_indices[best_local_index]
        best_movie = movies[best_global_index]
        distance = distances[0][0]
    else:
        print(f"No genre matches for {matched_genres}, using full search")
        query_embedding = model.encode([query])
        distances, indices = index.search(query_embedding, k=top_k)
        best_movie = movies[indices[0][0]]
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