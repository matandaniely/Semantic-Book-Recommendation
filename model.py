from sentence_transformers import SentenceTransformer
import numpy as np

def get_model(model_name="all-MiniLM-L6-v2"):       # Load a pre-trained model
    return SentenceTransformer(model_name)

def generate_embeddings(model, texts):
    dim = model.get_sentence_embedding_dimension()  # Get the dimension of the embeddings
    embeddings = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"Processing row {i} of {len(texts)}")
        embeddings[i] = model.encode(text, show_progress_bar=False)
    return embeddings
