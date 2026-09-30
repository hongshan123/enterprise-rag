# Aventro Motors Markdown RAG

This project indexes the 50 Aventro Motors Markdown files in `data/md` with a multilingual Sentence Transformers model and Chroma. `search` retrieves local excerpts; `ask` sends those excerpts and your question to a model API.

## Setup

In PowerShell, from this directory:

```powershell
conda env create --prefix .\.venv --file .\environment.yml
Copy-Item .\.env.example .\.env
```

Edit `.env` and set `OPENAI_API_KEY` to your DeepSeek API key. The example already selects `deepseek-flash`, `https://api.deepseek.com`, and Chat Completions mode. For the OpenAI API, set `OPENAI_MODEL` to your model, leave `OPENAI_BASE_URL` blank, and set `OPENAI_API_MODE=responses`. Keep `.env` private; Git ignores it.

## Build and query

```powershell
.\.venv\python.exe .\rag.py index
.\.venv\python.exe .\rag.py search "What is ABS?"
.\.venv\python.exe .\rag.py ask "What is ABS?"
```

Start the local web page and open `http://127.0.0.1:8000` in a browser:

```powershell
.\.venv\python.exe .\web.py
```

The web server listens only on your computer. It reads `.env` on the server; the API key is not sent to the browser.

The first `index` command downloads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` from Hugging Face. The Chroma index is stored in `vector_store/`. Rerun `index` after changing the source Markdown files. `ask` requires a valid model API key and model name; `search` works without either.

Optional settings in `.env`: `RAG_SOURCE_DIR`, `RAG_VECTOR_DIR`, `RAG_EMBEDDING_MODEL`. The default source path is relative to this project. Run `python rag.py --help` for command options.

## Dataset attribution

The Markdown corpus comes from [udayallu/RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus). Its MIT license and copyright notice are preserved in `data/LICENSE.upstream`.
