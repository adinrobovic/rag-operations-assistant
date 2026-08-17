# RAG Operations Assistant

A Python-based Retrieval-Augmented Generation (RAG) application that answers questions from an operations manual using OpenAI and ChromaDB.

## Demo
![RAG Operations Assistant Demo](assets/rag-demo.PNG)

## What It Does

- Loads operational documentation
- Splits the document into sections
- Creates embeddings with OpenAI
- Stores and searches them with ChromaDB
- Retrieves the most relevant section for a question
- Generates grounded answers with source citations
- Rejects unrelated questions using a relevance threshold
- Reuses saved embeddings unless the source document changes
- Includes retrieval and answer-quality tests

## Tech Stack

- Python
- OpenAI API
- ChromaDB
- Vector embeddings
- Semantic search
- SHA-256 hashing

## How It Works

Document
   ↓
Chunking
   ↓
Embeddings
   ↓
ChromaDB
   ↓
User Question
   ↓
Semantic Search
   ↓
Relevant Context
   ↓
LLM Answer + Source

## Evaluation

Current test results:

Retrieval Tests: 4/4 passed
Answer-Quality Tests: 3/3 passed

## Example

Question:

What should happen if a server overheats?

Response:

Verify the temperature through the monitoring dashboard, notify the operations lead,
and inspect airflow around the affected rack.

Source:
operations_manual.txt — Server Overheating Procedure

## Setup

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd qts-ai-assistant

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file:

OPENAI_API_KEY=your_api_key_here

Run the project:

```bash
python app.py
```

## What I Learned

This project gave me hands-on experience with RAG, embeddings, vector databases, OpenAI APIs, semantic search, prompt grounding, error handling, and AI evaluation.

## Future Improvements

- API access
- Deployment
- Additional document formats
- Simple user interface