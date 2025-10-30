#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Web Crawler & Article Extractor.

Features
- Fetch (retry, timeout, desktop UA)
- Metadata extraction (title, description, og:*)
- Main content extraction (readability-lxml 있으면 우선 사용, 없으면 soup 기반 휴리스틱)
- Markdown export (markdownify 있으면 사용)
- Simple CLI: python crawler.py https://example.com --markdown

Dependencies
- Required: requests, beautifulsoup4
- Optional: readability-lxml, markdownify

Design
- Pure functions + dataclass Article => 모듈화된 파이프라인
- 외부 모듈들과 쉽게 연결 (pdf_reader, summarizer 등)
"""

from __future__ import annotations

import re
import sys
import time
import json
import html
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

try:
    from readability import Document  # readability-lxml (optional)
    HAS_READABILITY = True
except Exception:
    HAS_READABILITY = False

try:
    import markdownify  # optional
    HAS_MARKDOWNIFY = True
except Exception:
    HAS_MARKDOWNIFY = False


# ------------------------------
# Config
# ------------------------------
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = (10, 20)  # connect, read
MAX_RETRIES = 2


# ------------------------------
# Data Model
# ------------------------------
@dataclass
class Article:
    url: str
    final_url: str
    status_code: int
    title: str
    description: str
    site_name: str
    author: str
    published_time: str
    content_html: str
    content_text: str
    content_markdown: str
    word_count: int
    lang: str


# ------------------------------
# Utils
# ------------------------------
_whitespace_re = re.compile(r"[ \t\f\v]+")
_newlines_re = re.compile(r"\n{3,}")

def _collapse_ws(s: str) -> str:
    s = html.unescape(s or "")
    s = s.replace("\r", "")
    s = _whitespace_re.sub(" ", s)
    s = _newlines_re.sub("\n\n", s)
    return s.strip()

def _guess_lang(soup: BeautifulSoup) -> str:
    # 우선순위: <html lang=".."> → meta[name=language] → og:locale → 빈문자열
    html_tag = soup.find("html")
    if html_tag and html_tag.has_attr("lang"):
        return html_tag["lang"].strip()
    meta_lang = soup.find("meta", attrs={"name": "language"})
    if meta_lang and meta_lang.get("content"):
        return meta_lang["content"].strip()
    og_locale = soup.find("meta", property="og:locale")
    if og_locale and og_locale.get("content"):
        return og_locale["content"].strip()
    return ""


# ------------------------------
# Fetch
# ------------------------------
def fetch_url(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: Tuple[int, int] = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    sleep_retry: float = 0.8,
) -> requests.Response:
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8,ko;q=0.7",
        **(headers or {}),
    }
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(sleep_retry * (attempt + 1))
            else:
                raise
    # theoretically unreachable
    raise last_exc  # type: ignore


# ------------------------------
# Metadata
# ------------------------------
def extract_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    def _meta(*, name: Optional[str] = None, prop: Optional[str] = None, attr="content") -> str:
        if name:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get(attr):
                return tag.get(attr, "").strip()
        if prop:
            tag = soup.find("meta", property=prop)
            if tag and tag.get(attr):
                return tag.get(attr, "").strip()
        return ""

    title = ""
    if soup.title and soup.title.text:
        title = soup.title.text.strip()

    og_title = _meta(prop="og:title")
    if og_title:
        title = og_title

    description = _meta(name="description") or _meta(prop="og:description")
    site_name = _meta(prop="og:site_name")
    author = _meta(name="author")
    published_time = _meta(prop="article:published_time") or _meta(name="pubdate")

    return {
        "title": title,
        "description": description,
        "site_name": site_name,
        "author": author,
        "published_time": published_time,
    }


# ------------------------------
# Content Extraction
# ------------------------------
def _remove_noise(soup: BeautifulSoup) -> None:
    # 스크립트/스타일/노이즈 블록 제거
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas", "form"]):
        tag.decompose()
    # 흔한 잡영역
    noise_ids = ["sidebar", "aside", "cookie", "footer", "nav", "comments", "share", "promo", "ads", "advert"]
    for nid in noise_ids:
        for t in soup.select(f'#{nid}, .{nid}'):
            t.decompose()

def _heuristic_main_node(soup: BeautifulSoup) -> Tag:
    # 기본: article, main, [role=main], .post, .entry, .content, .article
    candidates = soup.select("article, main, [role=main], .post, .entry, .content, .article, .post-content")
    if candidates:
        # 가장 긴 텍스트 길이를 가진 노드 선택
        candidates = sorted(candidates, key=lambda t: len(t.get_text(separator="\n", strip=True)), reverse=True)
        return candidates[0]
    # 폴백: body
    return soup.body or soup

def extract_content_html(html_text: str) -> Tuple[str, str, str, int]:
    """
    Returns: (content_html, content_text, content_markdown, word_count)
    """
    soup = BeautifulSoup(html_text, "html.parser")
    lang = _guess_lang(soup)
    _remove_noise(soup)

    # 1) readability 우선
    if HAS_READABILITY:
        try:
            doc = Document(str(soup))
            content_html = doc.summary(html_partial=True)
            # 간혹 summary가 짧으면 휴리스틱 사용
            if content_html and len(content_html) > 200:
                content_soup = BeautifulSoup(content_html, "html.parser")
            else:
                content_soup = _heuristic_main_node(soup)
        except Exception:
            content_soup = _heuristic_main_node(soup)
    else:
        content_soup = _heuristic_main_node(soup)

    # 텍스트
    if isinstance(content_soup, Tag):
        content_html = str(content_soup)
        content_text = content_soup.get_text(separator="\n", strip=True)
    else:
        # unlikely
        content_html = str(content_soup)
        content_text = getattr(content_soup, "text", "")

    content_text = _collapse_ws(content_text)

    # Markdown (optional)
    if HAS_MARKDOWNIFY:
        try:
            md = markdownify.markdownify(content_html or "", heading_style="ATX")
            content_markdown = _collapse_ws(md)
        except Exception:
            content_markdown = content_text
    else:
        content_markdown = content_text

    words = [w for w in re.split(r"\s+", content_text) if w]
    word_count = len(words)

    return content_html, content_text, content_markdown, word_count


# ------------------------------
# High-level API
# ------------------------------
def crawl(url: str) -> Article:
    resp = fetch_url(url)
    status = getattr(resp, "status_code", 0)
    text = resp.text or ""

    soup = BeautifulSoup(text, "html.parser")
    meta = extract_metadata(soup)
    lang = _guess_lang(soup)

    content_html, content_text, content_md, wc = extract_content_html(text)

    return Article(
        url=url,
        final_url=str(resp.url),
        status_code=status,
        title=meta["title"],
        description=meta["description"],
        site_name=meta["site_name"],
        author=meta["author"],
        published_time=meta["published_time"],
        content_html=content_html,
        content_text=content_text,
        content_markdown=content_md,
        word_count=wc,
        lang=lang,
    )


# ------------------------------
# CLI
# ------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lightweight web crawler & article extractor")
    p.add_argument("url", help="Target URL")
    p.add_argument("--markdown", action="store_true", help="Print markdown content")
    p.add_argument("--json", action="store_true", help="Print JSON (metadata + content)")
    p.add_argument("--text", action="store_true", help="Print plain text content")
    p.add_argument("--html", action="store_true", help="Print extracted HTML content")
    p.add_argument("--debug", action="store_true", help="Enable debug logs")
    return p

def main():
    args = _build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    art = crawl(args.url)

    if args.json:
        print(json.dumps(asdict(art), ensure_ascii=False, indent=2))
        return

    if args.markdown:
        print(art.content_markdown)
        return

    if args.text:
        print(art.content_text)
        return

    if args.html:
        print(art.content_html)
        return

    # 기본 출력: 간단 요약 + 텍스트 앞부분
    print(f"# {art.title or '(no title)'}")
    if art.site_name:
        print(f"- site: {art.site_name}")
    if art.published_time:
        print(f"- published: {art.published_time}")
    if art.author:
        print(f"- author: {art.author}")
    print(f"- url: {art.final_url}")
    print(f"- status: {art.status_code}")
    print(f"- lang: {art.lang or '(unknown)'}")
    print(f"- words: {art.word_count}")
    if art.description:
        print(f"\n> {art.description}\n")
    print(art.content_text[:1200] + ("..." if len(art.content_text) > 1200 else ""))

if __name__ == "__main__":
    main()
