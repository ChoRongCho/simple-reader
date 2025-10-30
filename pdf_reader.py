#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight PDF reader for blog/paper summarization pipelines.

Features
- Load from local path or URL
- Extract metadata (title, author, created, keywords, n_pages)
- Extract text per page (pdfminer.six preferred, fallback to pypdf)
- Optional OCR fallback via pytesseract + pdf2image
- Optional image extraction (PyMuPDF if available)
- Markdown rendering (page headings + separators)
- Page range selection ("1,3-5")

Dependencies
- Required: none
- Preferred: pdfminer.six OR pypdf (install at least one)
- Optional: requests (for URL), pytesseract+pdf2image (OCR), fitz/PyMuPDF (images)

Design
- Dataclass PdfDoc for clean interfaces
- Pure functions; easy to plug into summarizer or crawler pipeline
"""

from __future__ import annotations

import io
import re
import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Iterable, Any

# -------- Optional deps detection --------
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# Prefer pdfminer if available
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    HAS_PDFMINER = True
except Exception:
    HAS_PDFMINER = False

try:
    import pypdf
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    HAS_OCR = True
except Exception:
    HAS_OCR = False


# -------- Data model --------
@dataclass
class PdfDoc:
    source: str                 # 입력 경로 또는 URL
    final_source: str           # 리디렉션/정규화된 URL 또는 절대 경로
    n_pages: int
    title: str
    author: str
    created: str
    keywords: str
    producer: str
    subject: str
    pages: List[str]            # 페이지별 텍스트
    text: str                   # 전체 텍스트
    markdown: str               # 간단 변환 마크다운
    page_range: str             # 사용자가 요청한 페이지 범위 (원문 기준 1-index)
    used_engine: str            # "pdfminer" | "pypdf" | "ocr" | "mix"
    images_extracted: int       # (선택) 추출된 이미지 수
    lang_hint: str              # 언어 힌트(메타/추정, 비어있을 수 있음)


# -------- Utilities --------
_PAGE_RANGE_RE = re.compile(r"^\s*\d+(\s*-\s*\d+)?\s*$")

def parse_page_ranges(rng: Optional[str], n_pages: int) -> List[int]:
    """
    Parse "1,3-5" style string into 0-index page list bounded by n_pages.
    """
    if not rng:
        return list(range(n_pages))
    idxs: List[int] = []
    parts = [p.strip() for p in rng.split(",") if p.strip()]
    for p in parts:
        if not _PAGE_RANGE_RE.match(p):
            continue
        if "-" in p:
            a, b = [int(x.strip()) for x in p.split("-")]
            a = max(1, min(a, n_pages))
            b = max(1, min(b, n_pages))
            if a <= b:
                idxs.extend(list(range(a-1, b)))
            else:
                idxs.extend(list(range(b-1, a)))
        else:
            v = int(p)
            if 1 <= v <= n_pages:
                idxs.append(v-1)
    # unique, stable
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out

def bytes_from_source(src: str) -> Tuple[bytes, str]:
    """
    Load PDF bytes from local path or URL. Returns (bytes, final_source).
    """
    if src.lower().startswith(("http://", "https://")):
        if not HAS_REQUESTS:
            raise RuntimeError("URL 입력은 requests 가 필요합니다. `pip install requests`")
        r = requests.get(src, timeout=(10, 60), allow_redirects=True)
        r.raise_for_status()
        return r.content, str(r.url)
    else:
        abspath = os.path.abspath(src)
        with open(abspath, "rb") as f:
            return f.read(), abspath

def _meta_from_pypdf(reader: "pypdf.PdfReader") -> Dict[str, str]:
    info = reader.metadata or {}
    def get(k: str) -> str:
        try:
            v = info.get(k)
            return str(v) if v is not None else ""
        except Exception:
            return ""
    # Normalize common keys
    return {
        "title": get("/Title") or "",
        "author": get("/Author") or "",
        "created": get("/CreationDate") or "",
        "keywords": get("/Keywords") or "",
        "producer": get("/Producer") or "",
        "subject": get("/Subject") or "",
    }

def _extract_text_pdfminer(pdf_bytes: bytes, pages_0idx: Optional[List[int]]) -> List[str]:
    # pdfminer.high_level.extract_text 는 페이지 범위 직접 지정이 까다로워
    # 여러 번 호출한다 (대규모 파일엔 비용 증가 가능).
    results: List[str] = []
    if pages_0idx is None:
        txt = pdfminer_extract_text(io.BytesIO(pdf_bytes)) or ""
        # 큰 문자열을 페이지 단위로 정확히 나누기는 어려우니 그대로 반환
        return [txt]
    else:
        # 페이지별 호출 (성능 고려 필요)
        for i in pages_0idx:
            txt = pdfminer_extract_text(io.BytesIO(pdf_bytes), page_numbers=[i]) or ""
            results.append(txt)
        return results

def _extract_text_pypdf(pdf_bytes: bytes, pages_0idx: Optional[List[int]]) -> Tuple[List[str], int]:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    n_pages = len(reader.pages)
    idxs = pages_0idx or list(range(n_pages))
    out: List[str] = []
    for i in idxs:
        try:
            p = reader.pages[i]
            t = p.extract_text() or ""
        except Exception:
            t = ""
        out.append(t)
    return out, n_pages

def _ocr_pages(pdf_bytes: bytes, pages_0idx: Optional[List[int]]) -> List[str]:
    if not HAS_OCR:
        raise RuntimeError("OCR 사용에는 pytesseract, pdf2image 가 필요합니다.")
    # pdf2image -> PIL Images -> pytesseract
    pil_pages = convert_from_bytes(pdf_bytes, dpi=300)  # 메모리 사용 주의
    idxs = pages_0idx or list(range(len(pil_pages)))
    texts: List[str] = []
    for i in idxs:
        txt = pytesseract.image_to_string(pil_pages[i], lang="eng+kor")
        texts.append(txt or "")
    return texts

def _markdown_from_pages(pages: List[str], title: str = "") -> str:
    parts = []
    if title:
        parts.append(f"# {title}\n")
    for i, t in enumerate(pages, start=1):
        parts.append(f"\n\n---\n\n## Page {i}\n\n{t.strip()}\n")
    return "\n".join(parts).strip() + "\n"

def _guess_lang_from_meta(keywords: str, title: str) -> str:
    hay = f"{keywords} {title}".lower()
    if re.search(r"[가-힣]", hay):
        return "ko"
    if re.search(r"[a-z]", hay):
        return "en"
    return ""


# -------- Image extraction (optional) --------
def extract_images(pdf_bytes: bytes, pages_0idx: Optional[List[int]] = None, limit: int = 20) -> int:
    """
    Save page images to a temporary folder (./_pdf_images) as proof-of-concept.
    Returns number of saved images. Requires PyMuPDF.
    """
    if not HAS_FITZ:
        return 0
    os.makedirs("_pdf_images", exist_ok=True)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    idxs = pages_0idx or list(range(len(doc)))
    saved = 0
    for pi in idxs:
        page = doc[pi]
        imgs = page.get_images(full=True)
        for j, img in enumerate(imgs):
            if saved >= limit:
                break
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            out = os.path.join("_pdf_images", f"p{pi+1}_img{j+1}.png")
            if pix.n >= 5:  # CMYK -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(out)
            saved += 1
        if saved >= limit:
            break
    return saved


# -------- High-level API --------
def read_pdf(
    src: str,
    *,
    pages: Optional[str] = None,
    prefer: str = "pdfminer",       # "pdfminer" | "pypdf" | "auto"
    ocr: bool = False,
    extract_img: bool = False,
) -> PdfDoc:
    """
    Read PDF and return PdfDoc. For scanned PDFs, set ocr=True (optional deps needed).
    """
    pdf_bytes, final_src = bytes_from_source(src)

    # Determine page count early (pypdf preferred for cheap metadata)
    n_pages = 0
    meta = {
        "title": "", "author": "", "created": "", "keywords": "",
        "producer": "", "subject": ""
    }

    # Try pypdf for metadata
    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            n_pages = len(reader.pages)
            meta = _meta_from_pypdf(reader)
        except Exception:
            pass

    # If page count still unknown, try fitz
    if n_pages == 0 and HAS_FITZ:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            n_pages = len(doc)
        except Exception:
            pass

    if n_pages == 0:
        # As a last resort, assume 1
        n_pages = 1

    idxs = parse_page_ranges(pages, n_pages)

    used_engine = "mix"
    pages_text: List[str] = []

    # Extraction order
    if ocr and HAS_OCR:
        pages_text = _ocr_pages(pdf_bytes, idxs)
        used_engine = "ocr"
    else:
        if prefer == "pdfminer" and HAS_PDFMINER:
            # pdfminer per-page; if no pages set, it returns one big text
            per_page = _extract_text_pdfminer(pdf_bytes, idxs if idxs else None)
            if idxs and len(per_page) == len(idxs):
                pages_text = per_page
            else:
                # No reliable page split → distribute text (rough)
                pages_text = per_page if per_page else [""]
                if not idxs:
                    # best effort: keep as single "all pages"
                    pass
                used_engine = "pdfminer"
        elif HAS_PYPDF:
            pages_text, _ = _extract_text_pypdf(pdf_bytes, idxs)
            used_engine = "pypdf"
        elif HAS_PDFMINER:
            per_page = _extract_text_pdfminer(pdf_bytes, idxs if idxs else None)
            pages_text = per_page if per_page else [""]
            used_engine = "pdfminer"
        else:
            raise RuntimeError(
                "pdfminer.six 또는 pypdf 중 하나가 필요합니다. `pip install pdfminer.six` 또는 `pip install pypdf`"
            )

    # Normalize whitespace
    pages_text = [re.sub(r"[ \t\f\v]+", " ", (t or "")).replace("\r", "").strip() for t in pages_text]
    full_text = ("\n\n" + ("-"*80) + "\n\n").join(pages_text)
    md = _markdown_from_pages(pages_text, title=meta.get("title", "") or "")

    # Optional: images
    img_count = extract_images(pdf_bytes, idxs) if extract_img else 0

    lang_hint = _guess_lang_from_meta(meta.get("keywords", ""), meta.get("title", ""))

    return PdfDoc(
        source=src,
        final_source=final_src,
        n_pages=n_pages,
        title=meta.get("title", ""),
        author=meta.get("author", ""),
        created=meta.get("created", ""),
        keywords=meta.get("keywords", ""),
        producer=meta.get("producer", ""),
        subject=meta.get("subject", ""),
        pages=pages_text,
        text=full_text,
        markdown=md,
        page_range=pages or "",
        used_engine=used_engine,
        images_extracted=img_count,
        lang_hint=lang_hint,
    )


# -------- CLI --------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDF reader & text extractor")
    p.add_argument("src", help="PDF path or URL")
    p.add_argument("--pages", help='Page ranges like "1,3-5" (1-indexed)')
    p.add_argument("--engine", default="pdfminer", choices=["pdfminer", "pypdf", "auto"], help="Text extractor preference")
    p.add_argument("--ocr", action="store_true", help="Use OCR fallback (requires pytesseract + pdf2image)")
    p.add_argument("--images", action="store_true", help="Try extracting images (PyMuPDF)")
    p.add_argument("--json", action="store_true", help="Print JSON")
    p.add_argument("--markdown", action="store_true", help="Print Markdown")
    p.add_argument("--text", action="store_true", help="Print plain text")
    p.add_argument("--debug", action="store_true", help="Verbose errors")
    return p

def main():
    args = _build_argparser().parse_args()
    try:
        doc = read_pdf(
            args.src,
            pages=args.pages,
            prefer=args.engine,
            ocr=args.ocr,
            extract_img=args.images,
        )
    except Exception as e:
        if args.debug:
            raise
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(asdict(doc), ensure_ascii=False, indent=2))
        return
    if args.markdown:
        print(doc.markdown)
        return
    if args.text:
        print(doc.text)
        return

    # Default pretty summary
    print(f"# {doc.title or '(no title)'}")
    print(f"- pages: {len(doc.pages)}/{doc.n_pages} (range: {doc.page_range or 'all'})")
    if doc.author:
        print(f"- author: {doc.author}")
    if doc.created:
        print(f"- created: {doc.created}")
    if doc.subject:
        print(f"- subject: {doc.subject}")
    if doc.keywords:
        print(f"- keywords: {doc.keywords}")
    print(f"- engine: {doc.used_engine}")
    if doc.images_extracted:
        print(f"- images extracted: {doc.images_extracted}")
    print(f"- source: {doc.final_source}")
    print("\n---\n")
    print((doc.text[:2000] + ("..." if len(doc.text) > 2000 else "")))

if __name__ == "__main__":
    main()
