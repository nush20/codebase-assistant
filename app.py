"""Streamlit interface for the local codebase assistant."""

import os

import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_TOP_K
from rag_pipeline import answer_question, ingest_repo

load_dotenv()
st.set_page_config(page_title="Local Codebase Assistant", page_icon="🔎", layout="wide")

for key, default in {
    "repo_name": None, "chat_history": [], "last_ingestion": None, "retrieved_sources": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.title("Codebase Assistant")
    repo_path = st.text_input("Local repository path", placeholder="/path/to/repository")
    top_k = st.number_input("Retrieved chunks", min_value=1, max_value=20, value=DEFAULT_TOP_K)
    st.caption("Gemini API key: " + ("configured ✅" if os.getenv("GEMINI_API_KEY") else "missing ⚠️"))
    if st.button("Index Repository", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading, chunking, embedding, and indexing…"):
                result = ingest_repo(repo_path)
            st.session_state.repo_name = result.repo_name
            st.session_state.last_ingestion = result
            st.session_state.chat_history = []
            st.success("Repository indexed.")
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.last_ingestion:
        result = st.session_state.last_ingestion
        st.write(f"**Repository:** {result.repo_name}")
        st.write(f"**Files:** {result.files_processed}  \n**Chunks:** {result.chunks_created}")
        if result.skipped_file_count:
            st.caption(f"Skipped files: {result.skipped_file_count}")
        if result.indexing_errors:
            with st.expander("Indexing warnings"):
                for error in result.indexing_errors:
                    st.write(error)

st.title("Ask questions about a local codebase")
st.write("Index a repository, then ask grounded questions. Answers cite the retrieved files, symbols, and line ranges.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved source chunks"):
                for item in message["sources"]:
                    c = item.chunk
                    st.markdown(f"**{c.file_path} — {c.unit_name}, lines {c.start_line}-{c.end_line}** · score `{item.score:.4f}`")
                    st.code(c.text, language=c.language.lower())

question = st.chat_input("Ask about the indexed repository…", disabled=not st.session_state.repo_name)
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving code and asking Gemini…"):
                answer = answer_question(st.session_state.repo_name, question, int(top_k))
            st.markdown(answer.answer)
            if answer.retrieved_sources:
                with st.expander("Retrieved source chunks"):
                    for item in answer.retrieved_sources:
                        c = item.chunk
                        st.markdown(f"**{c.file_path} — {c.unit_name}, lines {c.start_line}-{c.end_line}** · score `{item.score:.4f}`")
                        st.code(c.text, language=c.language.lower())
        st.session_state.retrieved_sources = answer.retrieved_sources
        st.session_state.chat_history.append({"role": "assistant", "content": answer.answer, "sources": answer.retrieved_sources})
    except Exception as exc:
        message = f"Unable to answer: {exc}"
        st.error(message)
        st.session_state.chat_history.append({"role": "assistant", "content": message})
