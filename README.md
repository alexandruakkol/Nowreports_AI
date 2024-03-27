# Nowreports AI Module

## What is Nowreports?

Nowreports is a web application composed of three modules: Portal, API and AI, each having its own Github repository. Nowreports leverages company financial reports in order to allow users to inquire about the way a certain business works and is performing. State-of-the-art AI techniques are used to process data and create models that will allow the user to chat with the AI just like they would with the executive board of their chosen company. This enables users to obtain valuable insights into the internal business processes that may not be as easily accessible through other sources.

The AI module was built with Python, Milvus, and Postgres.

## Main Functions

- Web server for nowreports.com, built with Flask. Deployed in production using Gunicorn.
- 10-K financial report processing for AI ingestion (parse2.py):
  - Data cleanup and preprocessing.
  - Initial chunking and NLP tagging each node as one of: NarrativeText, Title, or Table (they go into different processing flows).
  - Secondary chunking via sliding window method with 30% overlap. Extra processing for duplicate data removal (chunks with fuzzy match over 98% are removed).
  - Labelling of each chunk with its Title component.
  - Embedding calculation for retrieval on runtime.
- Cross-checker (automated testing):
  - DB sync check (number of chunks written, Postgres and Milvus).

The availability and performance of the webserver, Milvus vector database (on Docker) and API module are being monitored by Uptime Robot through the testing endpoints.

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

Hybrid retrieval and switch to a finetuned bge-m3 embedding model are in the testing phase. Large improvements in AI performance are on their way.

