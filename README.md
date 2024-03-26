# Nowreports AI Component

Built with Python, Milvus, and Postgres.

## Main Functions

- Flask web server for the web interface.
- 10-K financial report processing for AI ingestion (parse2.py):
  - Data cleanup and preprocessing.
  - Primary chunking for NLP tagging as one of: NarrativeText, Title, or Table (they go into different processing flows).
  - Secondary chunking via sliding window method with 30% overlap. Extra processing for duplicate data removal (chunks with fuzzy match over 98% are removed).
  - Labelling of each chunk with its Title component.
  - Embedding calculation for retrieval on runtime.
- Cross-checker (automated testing):
  - DB sync check (number of chunks written, Postgres and Milvus).

## Models Used

- Mixtral 8x7b as the client-facing question-answering LLM. Runs via Mixtral AI API.
- alexakkol/BAAI-bge-base-en-nowr-1-2 (on Hugging Face) for embeddings generation. Fine-tuned by myself with financial data and synthetic data generation. Runs locally.
- gpt-3.5-turbo-1106 for synthetic training data generation for the embedding model.

## Files

- `main.py` is the entry point for the document processing flow.
- `parse2.py` contains document processing logic.
- `ai.py` contains the AI calls.
- `db.py` contains the database calls (to Milvus and Postgres, naming convention `mv_*` and `pg_*`).
- `webserver.py` is the Flask web server.

## Work in Progress

Hybrid retrieval and switch to a finetuned bge-m3 embedding model.
