import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

from utils.chunking import split_text
from utils.embedding import generate_embedding_batches, generate_query_embedding
from utils.llm import generate_answer
from utils.pdf_reader import extract_pdf_text
from utils.retriever import retrieve_chunks
from utils.vector_store import add_embeddings, create_vector_store


MAX_PDF_SIZE = 200 * 1024 * 1024

load_dotenv(Path(__file__).parent / ".env")
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")
st.title("AI Document Assistant")
st.caption("Upload PDF documents and ask questions grounded in their contents.")
st.caption("PDF upload limit: 200 MB per file.")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.processed_signature = None
    st.session_state.question_embeddings = {}

api_key = os.getenv("GOOGLE_API_KEY", "").strip()
if not api_key:
    st.info("Add your Gemini API key to the .env file to enable document processing and chat.")
    client = None
else:
    client = genai.Client(api_key=api_key)

upload_column, chat_column = st.columns([1, 2], gap="large")

with upload_column:
    st.subheader("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each file can be up to 200 MB. Scanned PDFs may need OCR.",
    )

    oversized_files = []
    if uploaded_files:
        st.caption("Selected files")
        for uploaded_file in uploaded_files:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.write(f"📄 {uploaded_file.name} — {size_mb:.2f} MB")
            if uploaded_file.size > MAX_PDF_SIZE:
                oversized_files.append(uploaded_file.name)
        if oversized_files:
            st.error("Each PDF must be 200 MB or smaller. Remove: " + ", ".join(oversized_files))

    process_clicked = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files or client is None,
    )

    if process_clicked and uploaded_files:
        if oversized_files:
            st.error("No documents were processed. Choose PDF files that are 200 MB or smaller.")
        else:
            file_signature = hashlib.sha256()
            for uploaded_file in uploaded_files:
                file_signature.update(uploaded_file.name.encode("utf-8"))
                file_signature.update(uploaded_file.size.to_bytes(8, "big"))
                buffer = uploaded_file.getbuffer()
                for offset in range(0, len(buffer), 1024 * 1024):
                    file_signature.update(buffer[offset : offset + 1024 * 1024])
            signature = file_signature.hexdigest()

            if signature == st.session_state.processed_signature:
                st.success("These exact files are already processed. Ask a question below.")
            else:
                # Do not accidentally keep answering from an older upload after a failed replacement.
                st.session_state.index = None
                st.session_state.chunks = None
                st.session_state.processed_signature = None
                st.session_state.question_embeddings = {}
                all_chunks = []
                progress = st.progress(0, text="Preparing documents…")
                try:
                    for file_number, uploaded_file in enumerate(uploaded_files, start=1):
                        progress.progress(
                            (file_number - 1) / (len(uploaded_files) + 1),
                            text=f"Reading {uploaded_file.name}…",
                        )
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getbuffer())
                            temp_path = temp_file.name
                        try:
                            document_text = extract_pdf_text(temp_path)
                        finally:
                            os.remove(temp_path)

                        file_chunks = split_text(document_text)
                        for chunk_number, chunk_text in enumerate(file_chunks, start=1):
                            all_chunks.append(
                                {
                                    "text": chunk_text,
                                    "source": uploaded_file.name,
                                    "section": chunk_number,
                                }
                            )
                        del document_text
                    if not all_chunks:
                        st.error("No readable text was found. This may be an empty or scanned PDF; OCR is not included.")
                    else:
                        index = None
                        texts = [item["text"] for item in all_chunks]
                        batch_count = (len(texts) + 99) // 100
                        for batch_number, (_, vectors) in enumerate(
                            generate_embedding_batches(client, texts), start=1
                        ):
                            if index is None:
                                index = create_vector_store(vectors)
                            else:
                                add_embeddings(index, vectors)
                            fraction = 0.6 + 0.4 * batch_number / batch_count
                            progress.progress(
                                fraction,
                                text=f"Creating embeddings: batch {batch_number} of {batch_count}…",
                            )
                        st.session_state.index = index
                        st.session_state.chunks = all_chunks
                        st.session_state.processed_signature = signature
                        st.session_state.question_embeddings = {}
                        progress.progress(1.0, text="Documents are ready.")
                        st.success(f"Ready to chat. Prepared {len(all_chunks)} sections from {len(uploaded_files)} PDF(s).")
                except Exception as error:
                    message = str(error)
                    if "password" in message.lower() or "encrypted" in message.lower():
                        st.error("A PDF is password-protected. Remove its password and upload it again.")
                    elif "429" in message or "RESOURCE_EXHAUSTED" in message:
                        st.error("Gemini rate limit reached. Wait a little and try processing again.")
                    elif "503" in message or "UNAVAILABLE" in message:
                        st.error("Gemini is temporarily unavailable. Please try again later.")
                    elif "timeout" in message.lower() or "timed out" in message.lower():
                        st.error("The request timed out. Try again or process smaller PDFs.")
                    else:
                        st.error(f"Could not process the PDF: {message}")
                finally:
                    progress.empty()

with chat_column:
    st.subheader("2. Ask about your PDFs")
    with st.form("question_form"):
        question = st.text_input(
            "Your question",
            placeholder="For example: What is this document about?",
        )
        ask_clicked = st.form_submit_button(
            "Ask question",
            type="primary",
            disabled=client is None or st.session_state.index is None,
        )

    if st.session_state.index is None:
        st.caption("Process at least one PDF to start asking questions.")
    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question before submitting.")
        elif client is not None:
            try:
                with st.spinner("Finding relevant passages and preparing an answer…"):
                    normalized_question = question.strip()
                    if normalized_question not in st.session_state.question_embeddings:
                        st.session_state.question_embeddings[normalized_question] = generate_query_embedding(
                            client, normalized_question
                        )
                    broad_question = any(
                        phrase in normalized_question.lower()
                        for phrase in ("summar", "overview", "main point", "about this", "important topic", "each section")
                    )
                    relevant_chunks = retrieve_chunks(
                        st.session_state.index,
                        st.session_state.question_embeddings[normalized_question],
                        st.session_state.chunks,
                        k=5,
                        broad=broad_question,
                    )
                    answer = generate_answer(client, relevant_chunks, normalized_question)
                st.markdown("#### Answer")
                st.write(answer)
            except Exception as error:
                message = str(error)
                if "429" in message or "RESOURCE_EXHAUSTED" in message:
                    st.error("Gemini rate limit reached. Wait a little and try again.")
                elif "503" in message or "UNAVAILABLE" in message:
                    st.error("Gemini is temporarily unavailable. Please try again later.")
                elif "timeout" in message.lower() or "timed out" in message.lower():
                    st.error("The request timed out. Please try again.")
                else:
                    st.error(f"Could not generate an answer: {message}")
