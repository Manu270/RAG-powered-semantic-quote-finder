
from embed_model import load_embedding_model
from vector_store import load_faiss_index
import numpy as np

def query_quotes(query, top_k=5):
    model = load_embedding_model()
    index, df = load_faiss_index()
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = df.iloc[I[0]]
    return results
