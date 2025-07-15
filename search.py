import numpy as np
import faiss as fa

def search_books(prompt, model, index, df, top_k=5):
    # Convert the prompt to an embedding 
    query_embedding = model.encode(prompt).astype('float32').reshape(1,-1)

    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve and return matching rows / books
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0]
    return results.sort_values(by="similarity_score", ascending=True).reset_index(drop=True), df["description"].iloc[indices[0]].tolist()
