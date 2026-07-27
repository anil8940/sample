## LangGraph RAG agent

This FastAPI app uses a LangGraph workflow with two nodes: `retrieve` searches
Qdrant and `answer` uses the retrieved chunks to produce a grounded response.

Start the stack (Docker pulls Qdrant automatically):

```bash
docker compose up --build
```

Ingest plain text into the knowledge base:

```bash
curl -X POST http://localhost:8000/documents -H "Content-Type: application/json" -d '{"source":"handbook","texts":["Your document text goes here."]}'
```

Ask the RAG agent:

```bash
curl -X POST http://localhost:8000/rag/ask -H "Content-Type: application/json" -d '{"question":"What does the handbook say?"}'
```

You can also use **Upload PDF** in the chat header. The app extracts selectable
text page by page, adds it to Qdrant, and then answers chat questions using the
indexed PDFs. Scanned/image-only PDFs require OCR before upload; the default
upload limit is 15 MB (configure `MAX_UPLOAD_SIZE_MB` to change it).

Set `GOOGLE_API_KEY` in `.env` before starting the application. The RAG service
uses Gemini's hosted `gemini-embedding-2-preview` model, so there is no local
embedding-model download. Qdrant's data is persisted in `qdrant_storage`.

Gemini embeddings use the API's free tier while it is available, subject to its
rate limits. The default collection is `documents-gemini`; ingest documents
again after switching embedding models because vectors from different models
cannot be compared.
