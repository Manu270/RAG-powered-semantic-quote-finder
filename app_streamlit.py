
import streamlit as st
from rag_pipeline import query_quotes

st.set_page_config(page_title="Semantic Quote Search", layout="centered")
st.title("📚 RAG-Powered Semantic Quote Finder")

query = st.text_input("Enter a quote topic or question (e.g., quotes about hope by Einstein):")

if query:
    with st.spinner("Searching..."):
        results = query_quotes(query, top_k=5)
        for i, row in results.iterrows():
            st.markdown(f"**\"{row['quote']}\"**")
            st.markdown(f"- *{row['author']}*")
            st.markdown(f"Tags: `{', '.join(row['tags'])}`")
            st.markdown("---")
