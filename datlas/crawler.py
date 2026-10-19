import asyncio
import base64
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse

import requests
from bs4 import BeautifulSoup

try:  # pragma: no cover - optional dependency
    from crawl4ai import AsyncWebCrawler, BrowserConfig
except Exception:  # pragma: no cover
    AsyncWebCrawler = None
    BrowserConfig = None


def sanitize_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = re.sub(r"\s+", " ", raw_text)
    return cleaned.strip()


def _search_terms(query: str) -> str:
    country_count = re.search(r"\bhow many countries\b.*?\bin\s+([^?.!]+)", query, re.IGNORECASE)
    if country_count:
        return f"List of sovereign states and dependent territories in {country_count.group(1).strip()}"
    stop_words = {
        "a", "about", "an", "and", "are", "can", "could", "describe", "does", "do",
        "explain", "give", "how", "i", "in", "is", "it", "me", "of", "overview",
        "please", "show", "summarize", "tell", "the", "to", "was", "what", "when",
        "where", "which", "who", "why", "with", "briefly", "concisely",
    }
    terms = [term for term in re.findall(r"[\w-]+", query) if term.lower() not in stop_words]
    return " ".join(terms) or query.strip()


def is_time_sensitive_query(query: str) -> bool:
    return bool(re.search(r"\b(latest|current|currently|today|now|recent|recently|this week|this month|breaking|live|news)\b", query, re.IGNORECASE))


def is_historical_query(query: str) -> bool:
    return bool(re.search(r"\b(history|historical|historic|archive|archived|past|formerly|old records)\b", query, re.IGNORECASE))


def search_duckduckgo(query: str, max_results: int = 5) -> List[str]:
    try:
        response = requests.get(
            f"https://html.duckduckgo.com/html/?q={quote(query)}",
                timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        urls: List[str] = []
        seen = set()
        for link in soup.select("a.result-link, a.result__a")[:max_results]:
            href = link.get("href")
            if not href:
                continue
            parsed = urlparse(href)
            if parsed.scheme and parsed.netloc:
                candidate = href
            else:
                params = parse_qs(parsed.query)
                candidate = params.get("uddg", [href])[0]
            if candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
        return urls
    except Exception:
        return []


def search_bing(query: str, max_results: int = 5) -> List[str]:
    """Fallback search for environments where DuckDuckGo blocks automated requests."""
    try:
        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query},
                timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        urls: List[str] = []
        seen = set()
        for link in soup.select("li.b_algo h2 a"):
            href = link.get("href")
            if not href:
                continue
            parsed = urlparse(href)
            candidate = href
            if parsed.hostname in {"bing.com", "www.bing.com"} and parsed.path == "/ck/a":
                encoded = parse_qs(parsed.query).get("u", [""])[0]
                if encoded.startswith("a1"):
                    try:
                        candidate = base64.urlsafe_b64decode(
                            encoded[2:] + "=" * (-len(encoded[2:]) % 4)
                        ).decode("utf-8")
                    except (ValueError, UnicodeDecodeError):
                        continue
            target = urlparse(candidate)
            if target.scheme not in {"http", "https"} or not target.netloc:
                continue
            if candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
            if len(urls) >= max_results:
                break
        return urls
    except Exception:
        return []


def search_google(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Use Google's official Custom Search API when API credentials are configured."""
    api_key = os.getenv("GOOGLE_API_KEY")
    engine_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not engine_id:
        return []
    try:
        params = {
            "key": api_key,
            "cx": engine_id,
            "q": query,
            "num": max(1, min(max_results, 10)),
        }
        if is_time_sensitive_query(query):
            params["sort"] = "date"
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
                timeout=8,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
    except (requests.RequestException, ValueError):
        return []
    return [
        {
            "title": str(item.get("title") or "Google result"),
            "url": str(item.get("link") or ""),
            "text": sanitize_text(str(item.get("snippet") or "")),
        }
        for item in items
        if urlparse(str(item.get("link") or "")).scheme in {"http", "https"}
    ]


def search_tavily(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Return Tavily's cleaned search context when an API key is configured."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            "include_answer": False,
            "max_results": max(1, min(max_results, 20)),
        }
        if is_time_sensitive_query(query):
            payload.update(topic="news", time_range=os.getenv("TAVILY_NEWS_TIME_RANGE", "week"))
        response = requests.post(
            "https://api.tavily.com/search",
            json=payload,
                timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    results = []
    for item in payload.get("results", []):
        text = sanitize_text(str(item.get("content") or ""))
        url = str(item.get("url") or "")
        if len(text) >= 120 and urlparse(url).scheme in {"http", "https"}:
            results.append({
                "title": str(item.get("title") or "Web source"),
                "url": url,
                "text": text[:5000],
            })
    return results


def search_wikipedia(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Use Wikipedia's public search API as a no-key factual fallback."""
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": max(1, min(max_results, 20)),
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 1,
                "format": "json",
            },
            headers={"User-Agent": "Datlas/2.5 (local Web RAG)"},
            timeout=8,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {}).values()
    except (requests.RequestException, ValueError):
        return []

    results = []
    query_terms = {term.lower() for term in re.findall(r"[\w-]+", query)}
    region_match = re.search(r"\bin\s+(.+)$", query, re.IGNORECASE)
    region_terms = set(re.findall(r"[\w-]+", region_match.group(1).lower())) if region_match else set()
    for page in pages:
        title = str(page.get("title") or "Wikipedia source")
        text = sanitize_text(str(page.get("extract") or ""))
        if len(text) < 120:
            continue
        title_terms = {term.lower() for term in re.findall(r"[\w-]+", title)}
        content_terms = {term.lower() for term in re.findall(r"[\w-]+", text[:2000])}
        if region_terms and not region_terms.intersection(title_terms | content_terms):
            continue
        score = 3 * len(query_terms & title_terms) + len(query_terms & content_terms)
        if query_terms and score < 2:
            continue
        results.append((score, page.get("index", 0), {
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            "text": text[:5000],
        }))
    results.sort(key=lambda item: (-item[0], item[1]))
    return [item for _, _, item in results[:max_results]]


def search_internet_archive(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search Internet Archive's public catalog for historical text records."""
    try:
        response = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f"({query}) AND mediatype:texts",
                "fl[]": ["identifier", "title", "description"],
                "rows": max(1, min(max_results, 20)),
                "output": "json",
            },
            headers={"User-Agent": "Datlas/2.5 (local Web RAG)"},
                timeout=8,
        )
        response.raise_for_status()
        documents = response.json().get("response", {}).get("docs", [])
    except (requests.RequestException, ValueError):
        return []

    results = []
    for item in documents:
        identifier = str(item.get("identifier") or "")
        description = item.get("description") or ""
        if isinstance(description, list):
            description = " ".join(str(part) for part in description)
        text = sanitize_text(str(description))
        if not identifier or len(text) < 120:
            continue
        results.append({
            "title": str(item.get("title") or identifier),
            "url": f"https://archive.org/details/{quote(identifier)}",
            "text": text[:3000],
        })
    return results


async def crawl_url(url: str) -> Dict[str, str]:
    title = "Web source"
    text = ""

    try:
        page = await asyncio.to_thread(
            requests.get,
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        page.raise_for_status()
        soup = BeautifulSoup(page.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "button"]):
            tag.decompose()
        content = soup.select_one("article, main, [role='main']") or soup.body or soup
        paragraphs = [sanitize_text(paragraph.get_text(" ", strip=True)) for paragraph in content.select("p")]
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        text = " ".join(paragraphs) or sanitize_text(content.get_text(" ", strip=True))
        title = (soup.title.get_text(" ", strip=True) if soup.title else title) or title
    except Exception:
        text = ""

    if len(text) < 120 and AsyncWebCrawler is not None and BrowserConfig is not None:
        try:
            browser_config = BrowserConfig(headless=True)
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url)
                if result is not None:
                    title = getattr(result, "title", title) or title
                    markdown = getattr(result, "markdown", None)
                    text = getattr(markdown, "fit_markdown", "") if markdown is not None else ""
                    if not text:
                        text = getattr(result, "html", "") or ""
                    text = sanitize_text(text)
        except Exception:
            text = ""

    return {"title": title, "url": url, "text": text[:3000]}


async def fetch_web_context(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    max_results = max(1, min(int(max_results), 20))
    provider_result_count = min(20, max_results * 2)
    search_query = _search_terms(query)
    time_sensitive = is_time_sensitive_query(query)
    search_jobs = [
        asyncio.to_thread(search_tavily, search_query, provider_result_count),
        asyncio.to_thread(search_google, search_query, provider_result_count),
        asyncio.to_thread(search_duckduckgo, search_query, provider_result_count),
    ]
    if not time_sensitive:
        search_jobs.append(asyncio.to_thread(search_wikipedia, search_query, provider_result_count))
    if is_historical_query(query):
        search_jobs.append(asyncio.to_thread(search_internet_archive, search_query, provider_result_count))
    search_jobs.extend([
        asyncio.to_thread(search_bing, search_query, provider_result_count),
        asyncio.to_thread(search_bing, f"site:britannica.com {search_query}", min(max_results, 5)),
    ])
    provider_results = await asyncio.gather(*search_jobs, return_exceptions=True)

    provider_lists = [
        result if isinstance(result, list) else []
        for result in provider_results
    ]
    direct_results: List[Dict[str, str]] = []
    urls: List[str] = []
    seen_urls = set()
    for result_index in range(max_results):
        for provider_result in provider_lists:
            if result_index >= len(provider_result):
                continue
            item = provider_result[result_index]
            if isinstance(item, dict):
                url = str(item.get("url") or "")
                text = sanitize_text(str(item.get("text") or ""))
                title = str(item.get("title") or "Web source")
                if len(text) >= 120:
                    if url and url not in seen_urls:
                        direct_results.append({"title": title, "url": url, "text": text[:3000]})
                        seen_urls.add(url)
                elif url and url not in seen_urls:
                    urls.append(url)
                    seen_urls.add(url)
            elif isinstance(item, str) and item not in seen_urls:
                urls.append(item)
                seen_urls.add(item)

    direct_results = direct_results[:max_results]
    crawl_count = min(len(urls), max_results * 3)
    crawl_jobs = [crawl_url(url) for url in urls[:crawl_count]]
    crawled_results = await asyncio.gather(*crawl_jobs, return_exceptions=True)
    for item in crawled_results:
        if isinstance(item, dict) and len(item.get("text", "").strip()) >= 120:
            direct_results.append(item)
    return direct_results[:max_results]
