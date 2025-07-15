import faiss

def create_faiss_index(embeddings, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path):
    faiss.write_index(index, path)

def load_index(path):
    return faiss.read_index(path)
