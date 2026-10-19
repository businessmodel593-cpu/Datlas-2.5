import base64
import asyncio
from types import SimpleNamespace

import torch

from datlas.crawler import (
    fetch_web_context,
    is_time_sensitive_query,
    search_bing,
    search_google,
    search_internet_archive,
    search_tavily,
)
from datlas.models.transformer import DatlasModel
from datlas.rag import (
    build_citation_list,
    build_country_count_answer,
    build_country_count_notes,
    chunk_text,
    rank_documents,
    select_context_excerpt,
)
from datlas.vector_store import LocalVectorStore, PersistentLocalVectorStore
from datlas.cache import ResultCache


def test_model_forward_shapes():
    model = DatlasModel(vocab_size=64, d_model=32, n_layers=2, n_heads=4, max_seq_len=32)
    idx = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx)

    assert logits.shape == (2, 8, 64)
    assert loss is None


def test_scratch_qwen_architecture_starts_under_one_billion_parameters():
    from pretrain_datlas import build_config
    from transformers import AutoModelForCausalLM

    class TokenizerShape:
        bos_token_id = 151643
        eos_token_id = 151645
        pad_token_id = None

        def __len__(self):
            return 151665

    config = build_config(TokenizerShape(), sequence_length=1024)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 438_009_856
    assert parameter_count < 1_000_000_000


def test_chunking_and_ranking():
    text = "Alpha beta gamma. Alpha beta delta. Gamma beta omega."
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 0
    ranked = rank_documents("alpha beta", chunks)
    assert ranked
    assert any("alpha" in chunk.lower() or "beta" in chunk.lower() for chunk in ranked)


def test_context_excerpt_prefers_matching_sentences_within_budget():
    text = (
        "Navigation and unrelated site chrome. "
        "Retrieval augmented generation retrieves relevant documents. "
        "It uses that context to ground an answer. More unrelated text about gardening."
    )

    excerpt = select_context_excerpt("What is retrieval augmented generation?", text, max_chars=100)

    assert len(excerpt) <= 100
    assert "Retrieval augmented generation" in excerpt
    assert "Navigation" not in excerpt


def test_ask_refuses_to_dump_context_without_model(monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "MODEL_READY", False)
    monkeypatch.setattr(app_module, "MODEL_LOAD_ERROR", "Model directory not found: test-model")
    client = TestClient(app_module.app)

    response = client.post("/ask", json={"question": "How does retrieval generation provide citations?"})

    assert response.status_code == 503
    assert "test-model" in response.json()["detail"]
    assert client.get("/health").json()["generation_ready"] is False
    assert client.post("/generate", json={"prompt": "hello"}).status_code == 503


def test_ask_validates_source_count_bounds():
    import app as app_module
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)

    assert client.post("/ask", json={"question": "test", "max_sources": 4}).status_code == 422
    assert client.post("/ask", json={"question": "test", "max_sources": 21}).status_code == 422


def test_datlas_identity_is_deterministic_without_model(monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "MODEL_READY", False)
    monkeypatch.setattr(app_module, "fetch_web_context", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("identity must not search")))
    client = TestClient(app_module.app)

    response = client.post("/ask", json={"question": "What is Datlas 2.5?"})

    assert response.status_code == 200
    assert "I am Datlas 2.5" in response.json()["answer"]
    assert response.json()["sources"] == []


def test_ask_synthesizes_web_context_with_loaded_local_model(monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    captured = {}

    async def fake_fetch(query, max_results=3):
        captured["max_sources"] = max_results
        return [{
            "title": "Africa reference",
            "url": "https://docs.example.org/africa",
            "text": "It lists 56 sovereign states (54 of which are member states of the United Nations).",
        }]

    class FakeTokenizer:
        chat_template = "test-template"
        pad_token_id = None
        eos_token_id = 2

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            captured["messages"] = messages
            return "rendered chat prompt"

        def __call__(self, text, return_tensors):
            assert text == "rendered chat prompt"
            return {"input_ids": torch.tensor([[11, 12]])}

        def decode(self, token_ids, skip_special_tokens):
            captured["generated_ids"] = token_ids.tolist()
            return "There are 56 countries in Africa. [1]"

    class FakeModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, **kwargs):
            captured["generation_options"] = kwargs
            return torch.tensor([[11, 12, 13, 14]])

    monkeypatch.setattr(app_module, "MODEL_READY", True)
    monkeypatch.setattr(app_module, "model", FakeModel())
    monkeypatch.setattr(app_module, "tokenizer", FakeTokenizer())
    monkeypatch.setattr(app_module, "fetch_web_context", fake_fetch)
    monkeypatch.setattr(app_module, "result_cache", ResultCache(ttl_seconds=600))
    client = TestClient(app_module.app)

    response = client.post("/ask", json={"question": "How many countries are in Africa?", "max_sources": 12})

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == (
        "The source identifies 54 United Nations member countries in Africa. "
        "Its broader list includes 56 sovereign states; those counts use different criteria [1]."
    )
    assert response.json()["sources"][0]["url"] == "https://docs.example.org/africa"
    assert captured["messages"][0]["role"] == "system"
    assert "56 sovereign states" in captured["messages"][0]["content"]
    assert "distinguish internationally recognized sovereign countries" in captured["messages"][0]["content"]
    assert "Preserve numeric qualifiers exactly" in captured["messages"][0]["content"]
    assert "report 54 UN member countries" in captured["messages"][0]["content"]
    assert captured["messages"][1]["content"] == "How many countries are in Africa?"
    assert captured["max_sources"] == 12
    assert captured["generated_ids"] == [13, 14]
    assert captured["generation_options"]["max_new_tokens"] == 32
    assert captured["generation_options"]["do_sample"] is False


def test_country_count_notes_preserve_source_qualifiers():
    sources = [{"text": "It lists 56 sovereign states (54 of which are member states of the United Nations)."}]
    notes = build_country_count_notes("How many countries are in Africa?", sources)

    assert notes == [
        "Source [1] states that 56 sovereign states are listed, 54 of which are UN member countries. "
        "For a question asking how many countries, report 54 UN member countries and clarify that 56 is the broader list total."
    ]
    assert "54 United Nations member countries" in build_country_count_answer("How many countries are in Africa?", sources)
    assert build_country_count_notes("What is RAG?", [{"text": "No country counts here."}]) == []


def test_bing_search_decodes_original_result_url(monkeypatch):
    target_url = "https://docs.example.org/rag"
    encoded_url = base64.urlsafe_b64encode(target_url.encode()).decode().rstrip("=")

    class FakeResponse:
        text = f'<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u=a1{encoded_url}">RAG</a></h2></li>'

        def raise_for_status(self):
            pass

    monkeypatch.setattr("datlas.crawler.requests.get", lambda *args, **kwargs: FakeResponse())

    assert search_bing("retrieval augmented generation") == [target_url]


def test_tavily_search_uses_api_key_and_clean_results(monkeypatch):
    import datlas.crawler as crawler

    captured = {}
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{
                "title": "RAG guide",
                "url": "https://docs.example.org/rag",
                "content": "A sufficiently long passage explains how retrieval augmented generation finds documents and grounds generated answers in those documents.",
            }]}

    def fake_post(url, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(crawler.requests, "post", fake_post)

    results = search_tavily("latest retrieval augmented generation news", max_results=5)

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["payload"]["api_key"] == "test-key"
    assert captured["payload"]["topic"] == "news"
    assert captured["payload"]["time_range"] == "week"
    assert results[0]["url"] == "https://docs.example.org/rag"


def test_google_cse_uses_configured_credentials(monkeypatch):
    import datlas.crawler as crawler

    captured = {}
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cse-test-id")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": [{
                "title": "Current result",
                "link": "https://news.example.org/current",
                "snippet": "A current news source with enough searchable text for the model to use.",
            }]}

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(crawler.requests, "get", fake_get)
    results = search_google("latest current news", max_results=8)

    assert captured["url"] == "https://www.googleapis.com/customsearch/v1"
    assert captured["params"]["key"] == "google-test-key"
    assert captured["params"]["cx"] == "cse-test-id"
    assert captured["params"]["sort"] == "date"
    assert results[0]["url"] == "https://news.example.org/current"


def test_internet_archive_search_returns_historical_source(monkeypatch):
    import datlas.crawler as crawler

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": {"docs": [{
                "identifier": "historical-record-1",
                "title": "Historical record",
                "description": "A sufficiently detailed historical text record from the Internet Archive that contains enough content to ground a historical answer.",
            }]}}

    monkeypatch.setattr(crawler.requests, "get", lambda *args, **kwargs: FakeResponse())
    results = search_internet_archive("historical record", max_results=8)

    assert results[0]["title"] == "Historical record"
    assert results[0]["url"] == "https://archive.org/details/historical-record-1"


def test_current_queries_skip_historical_archive(monkeypatch):
    import datlas.crawler as crawler

    current_source = {
        "title": "Current report",
        "url": "https://news.example.org/current",
        "text": "Current report text with enough detail to provide a meaningful, up-to-date source for the response. It includes verified information and context for a recent question.",
    }
    monkeypatch.setattr(crawler, "search_tavily", lambda *args, **kwargs: [])
    monkeypatch.setattr(crawler, "search_google", lambda *args, **kwargs: [current_source])
    monkeypatch.setattr(crawler, "search_duckduckgo", lambda *args, **kwargs: [])
    monkeypatch.setattr(crawler, "search_bing", lambda *args, **kwargs: [])
    monkeypatch.setattr(crawler, "search_wikipedia", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("skip Wikipedia for live news")))
    monkeypatch.setattr(crawler, "search_internet_archive", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("skip historical sources")))

    results = asyncio.run(fetch_web_context("latest current climate news"))

    assert is_time_sensitive_query("What is the latest climate news today?")
    assert results == [current_source]


def test_fetch_web_context_respects_requested_source_count(monkeypatch):
    import datlas.crawler as crawler

    def source_results(prefix, count):
        return [
            {"title": f"{prefix} {index}", "url": f"https://{prefix.lower()}.example.org/{index}", "text": f"Source {prefix} {index} provides enough useful details for grounded answers. " * 3}
            for index in range(count)
        ]

    monkeypatch.setattr(crawler, "search_tavily", lambda *args, **kwargs: source_results("Tavily", 20))
    monkeypatch.setattr(crawler, "search_google", lambda *args, **kwargs: source_results("Google", 20))
    monkeypatch.setattr(crawler, "search_duckduckgo", lambda *args, **kwargs: [])
    monkeypatch.setattr(crawler, "search_bing", lambda *args, **kwargs: [])
    monkeypatch.setattr(crawler, "search_wikipedia", lambda *args, **kwargs: source_results("Wiki", 20))
    monkeypatch.setattr(crawler, "search_internet_archive", lambda *args, **kwargs: source_results("Archive", 20))

    results = asyncio.run(fetch_web_context("ordinary factual query", max_results=12))

    assert len(results) == 12
    assert {item["title"].split()[0] for item in results} >= {"Tavily", "Google"}


def test_country_count_query_uses_wikipedia_before_bing(monkeypatch):
    import datlas.crawler as crawler

    captured = {}
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(crawler, "search_duckduckgo", lambda *args, **kwargs: [])

    def fake_wikipedia(query, max_results=3):
        captured["query"] = query
        return [{
            "title": "List of sovereign states and dependent territories in Africa",
            "url": "https://en.wikipedia.org/wiki/List_of_sovereign_states_and_dependent_territories_in_Africa",
            "text": "This is a list of sovereign states and dependent territories in Africa. The African Union has 55 member states and the United Nations recognizes 54 countries.",
        }]

    def unexpected_bing(*args, **kwargs):
        raise AssertionError("Bing should not be used when Wikipedia returned sources")

    monkeypatch.setattr(crawler, "search_wikipedia", fake_wikipedia)
    monkeypatch.setattr(crawler, "search_bing", unexpected_bing)

    results = asyncio.run(fetch_web_context("How many countries are in Africa?"))

    assert "sovereign states" in captured["query"].lower()
    assert results[0]["title"].startswith("List of sovereign states")


def test_wikipedia_search_filters_unrelated_regions(monkeypatch):
    import datlas.crawler as crawler

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"query": {"pages": {
                "1": {"index": 1, "title": "List of sovereign states in the Caribbean", "extract": "A list of sovereign states and dependent territories in the Caribbean region. " * 3},
                "2": {"index": 2, "title": "List of sovereign states in Africa", "extract": "A list of sovereign states and dependent territories in Africa. The list includes 54 United Nations member countries and other entities. " * 3},
            }}}

    monkeypatch.setattr(crawler.requests, "get", lambda *args, **kwargs: FakeResponse())

    results = crawler.search_wikipedia("List of sovereign states and dependent territories in Africa", max_results=3)

    assert len(results) == 1
    assert results[0]["title"] == "List of sovereign states in Africa"


def test_web_search_normalizes_question_and_extracts_page_content(monkeypatch):
    import datlas.crawler as crawler

    search_queries = []

    def fake_search(query, max_results=3):
        search_queries.append(query)
        return ["https://docs.example.org/rag"]

    class FakeResponse:
        text = """<html><head><title>RAG guide</title></head><body>
        <nav><p>Menu noise that must not appear in the extracted article.</p></nav>
        <main><article><p>Retrieval augmented generation finds relevant documents and uses them to answer a question with citations. This article describes how relevant context is added before the final response is produced for the user.</p></article></main>
        </body></html>"""

        def raise_for_status(self):
            pass

    monkeypatch.setattr(crawler, "search_duckduckgo", fake_search)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(crawler, "search_bing", lambda *args, **kwargs: [])
    monkeypatch.setattr(crawler, "AsyncWebCrawler", None)
    monkeypatch.setattr(crawler.requests, "get", lambda *args, **kwargs: FakeResponse())

    results = asyncio.run(crawler.fetch_web_context("What is retrieval augmented generation?", max_results=1))

    assert search_queries == ["retrieval augmented generation"]
    assert len(results) == 1
    assert "Retrieval augmented generation finds relevant documents" in results[0]["text"]
    assert "Menu noise" not in results[0]["text"]


def test_vector_store_and_cache():
    store = LocalVectorStore(dim=8)
    store.add_document("alpha beta gamma", source_url="https://example.com/a")
    store.add_document("gamma delta omega", source_url="https://example.com/b")

    hits = store.search("alpha beta", top_k=2)
    assert len(hits) >= 1
    assert hits[0].text.lower().startswith("alpha")

    cache = ResultCache(ttl_seconds=60)
    cache.set("alpha", {"answer": "cached"})
    assert cache.get("alpha") == {"answer": "cached"}


def test_persistent_store_and_citations(tmp_path):
    store = PersistentLocalVectorStore(dim=8, persist_dir=str(tmp_path / "vectors"))
    store.add_document("alpha beta gamma", source_url="https://example.com/a", title="Alpha")
    store.save()

    reloaded = PersistentLocalVectorStore(dim=8, persist_dir=str(tmp_path / "vectors"))
    hits = reloaded.search("alpha", top_k=1)
    assert hits and hits[0].source_url == "https://example.com/a"

    citations = build_citation_list([
        {"title": "Alpha", "url": "https://example.com/a", "snippet": "alpha beta"},
        {"title": "Beta", "url": "https://example.com/b", "snippet": "beta gamma"},
    ])
    assert "Alpha" in citations[0]
    assert "https://example.com/a" in citations[0]
