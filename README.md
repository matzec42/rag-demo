# rag-demo

A minimal Retrieval-Augmented Generation (RAG) pipeline built with LlamaIndex and OpenAI. Built as a proof of concept to explore how RAG works in practice — including retrieval behavior, prompt constraints, and the effect of conflicting or poisoned documents on model output.

## What It Does

- Loads documents from a local `docs/` directory
- Embeds and indexes them using OpenAI embeddings
- Accepts natural language questions via the terminal
- Retrieves relevant chunks and answers using only the provided context
- Refuses to answer questions outside the scope of the documents

## Setup

**Requirements:** Python 3.9+, an OpenAI API key with available credits.

```bash
# Clone and enter the project
mkdir rag-demo && cd rag-demo

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install llama-index llama-index-llms-openai llama-index-embeddings-openai

# Set your OpenAI API key (required each terminal session)
export OPENAI_API_KEY=your-key-here
```

## Usage

Add `.txt` files to the `docs/` directory, then run:

```bash
python3 rag.py
```

Ask questions in the terminal. The model will only answer based on the contents of your documents.

## Key Observations

- Without explicit prompt constraints, LlamaIndex defaults to a chat prompt template that allows the model to draw on prior training knowledge — not just your documents
- Conflicting information across documents is handled by retrieval ranking, which is non-deterministic and can surface incorrect chunks depending on query phrasing
- Prompt design directly affects retrieval behavior and answer quality

## Stack

- [LlamaIndex](https://www.llamaindex.ai/) — document indexing and RAG pipeline
- [OpenAI](https://platform.openai.com/) — embeddings and LLM (gpt-3.5-turbo / gpt-4)
