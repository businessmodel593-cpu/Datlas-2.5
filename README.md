# Datlas 2.5

Datlas 2.5 is a local Web-RAG application with a from-scratch causal language model training path. The independent model uses a Qwen2-shaped architecture initialized with random weights; no Qwen model weights are loaded. The local Qwen tokenizer is reused.

`pretrain_datlas.py` streams the dataset's `user_prompt` and `agent_reasoning` columns, masks prompt tokens from the loss, and saves resumable checkpoints. Training is a substantial GPU job; this CPU-only workspace can validate the setup but cannot complete meaningful 438M-parameter pretraining. The dataset is specialized synthetic agent-reasoning data, not a broad general-language corpus, so it will not alone produce a capable general-purpose LLM.

## Repository Structure

```text
datlas/
  models/          # Core transformer architecture (attention, blocks, full model)
  training/        # Pretraining utilities
pretrain_datlas.py # Scratch Qwen2-shaped pretraining engine
  crawler.py       # Tavily search plus DuckDuckGo/Bing fallback and crawling
  vector_store.py  # Local vector store for retrieval
  cache.py         # TTL-based result cache
  rag.py           # Chunking, ranking, and prompt construction
app.py             # FastAPI serving and Web-RAG request flow
requirements.txt
pytest.ini
tests/
```

## Quick Start (Local)

```bash
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then visit http://localhost:8000/docs for interactive API docs.

The dataset is downloaded locally under the ignored `datasets/` directory. To fetch it again:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="BlueIsGreen/Edge-Agent-Reasoning-WebSearch-260K", repo_type="dataset", local_dir="datasets/Edge-Agent-Reasoning-WebSearch-260K", allow_patterns=["*.parquet", "README.md"])'
```

Validate the tokenizer, dataset stream, and parameter count without allocating model weights:

```bash
python pretrain_datlas.py --dry-run
```

Run full pretraining on a CUDA machine with substantial VRAM (24 GB recommended):

```bash
python pretrain_datlas.py --epochs 3 --sequence-length 1024 --batch-size 1 --gradient-accumulation-steps 16
```

The application now loads only `./datlas-2.5-independent-model` by default. It returns a clear unavailable status until training creates that directory; override with `DATLAS_MODEL_DIR` if needed. Dataset shards, checkpoints, and model weights are git-ignored. Replies default to 32 new tokens; adjust with `DATLAS_MAX_NEW_TOKENS`.

Chat retrieves 8 web sources by default; each `/ask` request can set `max_sources` from 5 to 20. Search runs providers concurrently and over-fetches candidates to fill the requested count. DuckDuckGo, Bing, and Wikipedia work without API keys; Internet Archive is queried only for historical/archive questions. Tavily requires `TAVILY_API_KEY`. Google requires both `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`. Britannica is queried through a targeted Bing search. Latest/current/news queries request date-aware Google/Tavily results and bypass the answer cache. No search API can guarantee complete coverage of the web.

Set `DATLAS_TRUST_REMOTE_CODE=true` only for a model whose custom loading code you trust.

### Example Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is RAG and why is it useful?","max_sources":12}'
```

## Features

- custom local transformer model
- concurrent Web-RAG search through Tavily, Google CSE, DuckDuckGo, Bing/Britannica, Wikipedia, and Internet Archive
- live web crawling with Crawl4AI fallback
- configurable 5–20 source retrieval
- deterministic Datlas 2.5 identity response
- local FAISS-backed vector retrieval
- TTL-based result caching
- locally generated answers grounded in retrieved web context, with citations
- health and chat endpoints
- `/generate` and `/v1/completions` endpoints when a local model is configured
- custom transformer architecture and training scaffold

## Notes

No trained model weights or Tavily API key are committed. The dataset license is MIT. Full scratch pretraining has not been run in this CPU-only workspace; no trained `datlas-2.5-independent-model` checkpoint exists until the documented GPU training command completes. Search providers do not guarantee complete web coverage or instantaneous indexing.

## License

See LICENSE file.
