# AI Document Assistant

A beginner-friendly PDF question-answering app built with Python, Streamlit, Gemini, and FAISS. Upload one or more PDFs (up to 200 MB each) and ask questions grounded in their contents.

This project is based on the open-source [Ask to Zuzu – PDF Q&A Chatbot](https://github.com/aankitsharma/ask-to-zuzu-rag-chatbot) by Aankit Sharma. It has been customized for clearer document handling, PDF-grounded responses, and easier local use. Local syntax and packaging checks were performed; a successful Gemini response and a 200 MB PDF upload have not been tested here.

## Features

- Upload one or more PDF files, up to 200 MB per file.
- Show selected filenames and file sizes.
- Extract text, split it into overlapping sections, and create Gemini embeddings in batches.
- Search with FAISS and send only a small set of relevant excerpts to Gemini.
- Ask specific questions or broad questions such as “What is this document about?” and “Summarize this PDF.” Broad requests include excerpts spread across the documents.
- Directly handle unrelated questions and related questions whose answer is missing, using separate messages.
- Include source filenames in answers when practical.
- Avoid rebuilding embeddings when the same uploaded files are submitted again in the current app session. Reuse embeddings for repeated identical questions.
- Keep the API key in a local `.env` file.

## PDF-only answers

The model receives only retrieved excerpts from the uploaded PDFs and is instructed not to use outside knowledge. It first decides whether a question relates to the document topic. For unrelated questions, it returns the out-of-scope message. For related questions with no supporting detail in the retrieved excerpts, it returns the not-found message. Answers must be based on supplied text and can include source filenames.

This is prompt-guided behavior, not a mathematical guarantee: a language model can still misclassify scope or make mistakes. Retrieval may also miss a relevant passage, especially in long or poorly structured documents.

## What RAG means

Retrieval-Augmented Generation (RAG) means finding useful document passages first, then giving those passages to a language model with the question. Here, each text section and question is converted into an embedding (a list of numbers representing meaning). FAISS finds nearby section embeddings. Gemini then responds using the retrieved PDF text.

## Project workflow

1. Select PDF files in the uploader.
2. The app validates file sizes and extracts selectable text from each PDF.
3. The text is divided into 1,000-character chunks with 200 characters of overlap.
4. Gemini creates embeddings in batches of up to 100 chunks; FAISS stores the vectors in memory.
5. The question is embedded and matched against the document vectors.
6. For broad summary questions, the app adds representative passages from across the uploaded documents.
7. Gemini answers from the selected passages or returns the appropriate scope/missing-information message.

## Technologies

Python, Streamlit, Google Gemini API (`google-genai`), `gemini-embedding-001`, `gemini-3.8-flash`, FAISS, LangChain text splitters, PyMuPDF, NumPy, and python-dotenv.

## Windows installation

Install Python 3.10 or newer. Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, use Command Prompt and run `.venv\Scripts\activate.bat` instead.

Open `.env` in a text editor and replace the placeholder with a Gemini API key from [Google AI Studio](https://aistudio.google.com/):

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Keep `.env` private. It is excluded by `.gitignore` and must not be shared.

## Run the app

From the project folder with the virtual environment active:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Example questions

- What is this document about?
- Summarize the main points.
- What does the PDF say about [topic]?
- Which document mentions [subject]?
- What is the definition of [term] in the document?
- Who won yesterday’s cricket match? (Should be rejected as unrelated if the PDF is about another subject.)

## Project structure

```text
ai-document-assistant/
├── .streamlit/
│   └── config.toml       # 200 MB Streamlit upload limit
├── app.py                # Streamlit interface and workflow
├── utils/
│   ├── chunking.py       # Split text into overlapping sections
│   ├── embedding.py      # Create embeddings in API-sized batches
│   ├── llm.py            # Scope check and grounded answer prompt
│   ├── pdf_reader.py     # Extract selectable PDF text
│   ├── retriever.py      # FAISS similarity and broad retrieval
│   ├── storage.py        # Optional local index save/load helpers
│   └── vector_store.py   # Build and extend the FAISS index
├── step1_*.py … step8_*.py # Learning scripts for the RAG steps
├── sample.pdf
├── requirements.txt
├── .env.example
└── README.md
```

## Limitations

- The 200 MB setting is per file. Multiple large uploads can still require substantial memory, disk space, API time, and quota.
- A 200 MB PDF test was not performed. Large PDFs may take a long time to extract and embed; API rate limits can stop processing.
- Scanned/image-only PDFs need OCR, which is not included.
- The index is held in memory and must be rebuilt in a new app session.
- Broad summaries use a limited set of passages and may omit details from long PDFs.
- Scope classification and answer grounding are model-guided and can make mistakes; review important answers against the PDF.
- Gemini access, quotas, and model availability depend on Google API settings.

## Credits and attribution

Original repository: [Aankit Sharma – ask-to-zuzu-rag-chatbot](https://github.com/aankitsharma/ask-to-zuzu-rag-chatbot). This project is a customized version of that existing open-source work; it is not presented as original authorship. See the original repository for its license and history.
