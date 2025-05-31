
from data_loader import load_and_prepare_quotes
from embed_model import load_embedding_model
from vector_store import build_faiss_index
import numpy as np

df = load_and_prepare_quotes()
model = load_embedding_model()
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
build_faiss_index(np.array(embeddings), df)
print("✅ FAISS index built and saved successfully.")
