
import faiss
import numpy as np
import pandas as pd
import pickle

def build_faiss_index(embeddings, df, save_path="faiss_index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{save_path}.index")
    with open(f"{save_path}.pkl", "wb") as f:
        pickle.dump(df, f)

def load_faiss_index(index_path="faiss_index"):
    index = faiss.read_index(f"{index_path}.index")
    with open(f"{index_path}.pkl", "rb") as f:
        df = pickle.load(f)
    return index, df
