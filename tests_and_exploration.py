"""
tests_and_exploration.py

Exploration and testing script for the Book Recommender System.
Includes loading data, generating embeddings, creating FAISS index,
and running a test query.
"""
import os
import pandas as pd

#  Local Modules 
from model import get_model, generate_embeddings
from indexer import create_faiss_index
from utils import text_to_string
from search import search_books


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load books CSV and prepare the full_text and main_category fields.
    """
    print(f"[INFO] Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    df["full_text"] = df.apply(text_to_string, axis=1)
    df["main_category"] = df["categories"].apply(
        lambda x: x.split(",")[0] if isinstance(x, str) else "Unknown"
    )
    return df


def generate_sample_embeddings(df: pd.DataFrame, sample_size: int = 5):
    """
    Generate embeddings for a small sample of books.
    """
    print(f"[INFO] Generating embeddings for {sample_size} samples...")
    model = get_model()
    texts = df["full_text"].iloc[:sample_size].tolist()
    embeddings = generate_embeddings(model, texts)
    return model, texts, embeddings


def test_similarity_search(model, embeddings, texts, df, top_k: int = 3):
    """
    Create a FAISS index and run similarity search on a single sample.
    """
    print("[INFO] Creating FAISS index and running similarity search...")
    index = create_faiss_index(embeddings, model.get_sentence_embedding_dimension())
    results, _ = search_books(texts[0], model, index, df.iloc[:len(texts)], top_k=top_k)
    print(f"\nTop {top_k} similar books for:\n\"{texts[0][:100]}...\"\n")
    print(results[["title", "authors", "similarity_score"]])


if __name__ == "__main__":
    books_path = "books.csv"

    if not os.path.exists(books_path):
        raise FileNotFoundError(f"{books_path} not found. Make sure the CSV file is in the working directory.")

    df_books = load_and_prepare_data(books_path)

    print("\n[INFO] Top 10 categories in dataset:")
    print(df_books["main_category"].value_counts().head(10))

    model, sample_texts, sample_embeddings = generate_sample_embeddings(df_books)
    test_similarity_search(model, sample_embeddings, sample_texts, df_books)