import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from datlas.cache import ResultCache
from datlas.crawler import fetch_web_context, is_time_sensitive_query
from datlas.rag import (
    build_citation_list,
    build_country_count_answer,
    build_country_count_notes,
    select_context_excerpt,
)

app = FastAPI(
    title="Datlas 2.5 Web RAG",
    description="Source-grounded web chat with optional local transformer generation",
)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model_dir_setting = Path(os.getenv("DATLAS_MODEL_DIR", "datlas-2.5-independent-model"))
MODEL_DIR = model_dir_setting if model_dir_setting.is_absolute() else Path(__file__).resolve().parent / model_dir_setting
MODEL_NAME = os.getenv("MODEL_NAME", MODEL_DIR.name)
MAX_RAG_TOKENS = int(os.getenv("DATLAS_MAX_NEW_TOKENS", "32"))
RAG_CONTEXT_CHAR_BUDGET = int(os.getenv("DATLAS_CONTEXT_CHAR_BUDGET", "3500"))
MAX_SOURCE_TEXT_CHARS = int(os.getenv("DATLAS_SOURCE_TEXT_CHARS", "1400"))
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, int(os.getenv("DATLAS_CPU_THREADS", "2"))))
model = None
tokenizer = None
MODEL_READY = False
MODEL_LOAD_ERROR = None
if MODEL_DIR.is_dir():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=os.getenv("DATLAS_TRUST_REMOTE_CODE", "false").lower() == "true",
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=os.getenv("DATLAS_TRUST_REMOTE_CODE", "false").lower() == "true",
            local_files_only=True,
        )
        model.eval()
        MODEL_READY = True
    except Exception as exc:
        MODEL_LOAD_ERROR = f"Could not load local model from {MODEL_DIR}: {exc}"
else:
    MODEL_LOAD_ERROR = f"Model directory not found: {MODEL_DIR}"

result_cache = ResultCache(ttl_seconds=600)


def generate_response(messages: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    if not MODEL_READY or model is None or tokenizer is None:
        # Clean fallback generation instead of hard crash
        user_msg = messages[-1]['content'] if messages else ""
        return f"[Fallback Mode - Local model unavailable] Answer for: '{user_msg}'. (Note: Configure DATLAS_MODEL_DIR or run pretrain_datlas.py to load real weights.)"


    if getattr(tokenizer, "chat_template", None):
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_input = "\n\n".join(
            f"### {message['role'].title()}:\n{message['content']}"
            for message in messages
        ) + "\n\n### Assistant:"

    inputs = tokenizer(formatted_input, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: value.to(input_device) for name, value in inputs.items()}
    generation_options = {
        "max_new_tokens": max(1, min(int(max_new_tokens), 1024)),
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_options["temperature"] = float(temperature)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_options)
    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.7


class AskRequest(BaseModel):
    question: str
    max_sources: int = Field(default=8, ge=5, le=20)


class ChatResponse(BaseModel):
    response: str
    model: str = "datlas-2.5"


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "generation_ready": MODEL_READY,
        "model_dir": str(MODEL_DIR),
        "model_error": MODEL_LOAD_ERROR,
    }


@app.post("/generate", response_model=ChatResponse)
def generate(req: ChatRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR or "Local model is not ready")
    try:
        text = generate_response(
            [{"role": "user", "content": req.prompt}],
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return ChatResponse(response=text.strip())
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - API safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(req: AskRequest):
    """Answer from matching retrieved passages and cite their original pages."""
    try:
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        normalized_question = " ".join(question.lower().replace("2.5", "2 5").strip(" .?!").split())
        if (
            normalized_question in {"what is datlas", "what is datlas 2 5", "who are you", "what are you", "what is your name"}
            or "tell me about yourself" in normalized_question
        ):
            return {
                "question": question,
                "answer": (
                    f"I am Datlas 2.5, running locally on {MODEL_NAME}. "
                    "I use live web search to ground answers and include source citations."
                ),
                "sources": [],
                "citations": [],
            }
        # MODEL_READY check bypassed for graceful fallback support


        cache_key = f"{question.lower()}::sources={req.max_sources}"
        cached = None if is_time_sensitive_query(question) else result_cache.get(cache_key)
        if cached is not None:
            return cached

        web_results = asyncio.run(fetch_web_context(question, max_results=req.max_sources))
        if not web_results:
            return {
                "question": question,
                "answer": "I could not retrieve verified web sources for this question, so I can't provide a grounded answer.",
                "sources": [],
                "citations": [],
            }

        chars_per_source = min(
            MAX_SOURCE_TEXT_CHARS,
            max(200, RAG_CONTEXT_CHAR_BUDGET // len(web_results)),
        )
        source_blocks = []
        for index, item in enumerate(web_results, start=1):
            excerpt = select_context_excerpt(
                question,
                str(item.get("text") or ""),
                chars_per_source,
            )
            source_blocks.append(
                f"[{index}] TITLE: {item.get('title') or 'Web source'}\n"
                f"URL: {item.get('url') or ''}\nCONTENT:\n{excerpt}"
            )
        retrieved_web_text = "\n\n".join(source_blocks)
        count_notes = build_country_count_notes(question, web_results)
        numeric_guidance = "\n\nNumeric distinctions:\n" + "\n".join(count_notes) if count_notes else ""
        system_prompt = (
            "You are Datlas. Answer the user's specific question directly in one to three concise sentences, "
            "using only the verified web context below. Synthesize; do not reproduce entire passages. Cite "
            "each factual claim with its source number, such as [1]. For country-count questions, distinguish "
            "internationally recognized sovereign countries from territories and partially recognized entities; "
            "never call a broader list total the number of countries unless the source defines it that way. "
            "Preserve numeric qualifiers exactly: if a source says N of M listed states are UN members, "
            "report N as the UN-member-country count and describe M only as the broader list total. "
            "If sources disagree or the context does not resolve the question, explain the distinction.\n\n"
            f"{retrieved_web_text}{numeric_guidance}"
        )
        answer = generate_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_new_tokens=MAX_RAG_TOKENS,
            temperature=0.0,
        )
        answer = build_country_count_answer(question, web_results) or answer
        if not answer:
            raise HTTPException(status_code=502, detail="The local model returned an empty answer")

        source_entries = [
            {
                "title": item.get("title") or "Web source",
                "url": item.get("url") or "",
                "snippet": str(item.get("text") or "")[:300],
            }
            for item in web_results
        ]

        payload: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "sources": source_entries,
            "citations": build_citation_list(source_entries),
        }
        if not is_time_sensitive_query(question):
            result_cache.set(cache_key, payload)
        return payload
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - API safety
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/completions")
def openai_compat(req: dict):
    """Simple OpenAI-compatible completions endpoint."""
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail=MODEL_LOAD_ERROR or "Local model is not ready")
    prompt = str(req.get("prompt", ""))
    max_tokens = int(req.get("max_tokens", 64))
    text = generate_response([{"role": "user", "content": prompt}], max_new_tokens=max_tokens)
    return {
        "id": "datlas-cmpl",
        "object": "text_completion",
        "choices": [{"text": text.strip(), "index": 0, "finish_reason": "stop"}],
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
