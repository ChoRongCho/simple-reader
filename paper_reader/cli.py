# paper_reader/cli.py
from __future__ import annotations
import os
import sys
import argparse
import mimetypes
import re
from urllib.parse import urlparse

from config_loader import load_env
from llm_client import LLMConfig, create_llm_client, summarize_text
from crawler import crawl
from pdf_reader import read_pdf

# paper_reader/cli.py 중 일부 교체/추가
import re
from datetime import datetime, timezone, timedelta


def extract_h1_title(md: str) -> str:
    """
    마크다운에서 첫 번째 H1(# ...)의 텍스트 추출.
    없으면 빈 문자열 반환.
    """
    m = re.search(r'^\s*#\s+(.+)$', md, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "").strip()

def kst_timestamp() -> str:
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST).strftime("%y%m%d-%H%M%S")


# ---------------------------
# 유틸 함수
# ---------------------------
def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def looks_like_pdf(target: str) -> bool:
    if target.lower().endswith(".pdf"):
        return True
    mt, _ = mimetypes.guess_type(target)
    return (mt == "application/pdf")

def slugify(text: str) -> str:
    """제목을 안전한 파일 이름으로 변환"""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "summary"

def ensure_results_dir() -> str:
    """root_folder/results 디렉토리 보장"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result_dir = os.path.join(root, "results")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def get_default_template_path() -> str:
    """기본 템플릿 경로 (root/templates/SUMMARY.md)"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    template_path = os.path.join(root, "templates", "SUMMARY.md")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"기본 템플릿을 찾을 수 없습니다: {template_path}")
    return template_path


# ---------------------------
# 실행 로직
# ---------------------------
def run(target: str, *, pages: str | None, engine: str, ocr: bool, save: bool, template: str | None):
    env = load_env()
    cfg = LLMConfig(
        provider=env.provider,
        model=env.openai_model if env.provider == "openai" else env.ollama_model,
        temperature=env.temperature,
        max_tokens=env.max_tokens,
        openai_api_key=env.openai_api_key,
        openai_base_url=env.openai_base_url,
        ollama_base_url=env.ollama_base_url,
    )
    client = create_llm_client(cfg)
    
    # ✅ 여기 추가
    if not target.lower().startswith(("http://", "https://")):
        target = os.path.abspath(os.path.expanduser(target))
        
    # --- 원문 추출 ---
    if is_url(target) and not looks_like_pdf(target):
        art = crawl(target)
        src_text = art.content_text
        title = art.title or "Untitled"
        meta = {"source": art.final_url, "length": len(src_text)}
    else:
        doc = read_pdf(target, pages=pages, prefer=engine, ocr=ocr)
        src_text = doc.text
        title = doc.title or "PDF Summary"
        meta = {"source": doc.final_source, "length": len(src_text)}

    # --- 템플릿 경로 지정 ---
    template_path = template or get_default_template_path()

    # --- 요약 ---
    print(f"[INFO] Using template: {template_path}")
    print(f"[INFO] Summarizing: {title}")
    md = summarize_text(
        client,
        src_text,
        # title 힌트는 넘기지 않음 (LLM이 결정)
        meta=meta,
        template_path=template_path,
        auto_title=True,             # LLM이 제목 결정
        title_language="en",         # 영어 제목
    )


    # --- 파일명 구성 ---
    derived_title = extract_h1_title(md) or "summary"
    slug = slugify(derived_title)
    ts = kst_timestamp()
    filename = f"{slug}-{ts}.md"

    # --- 출력 및 저장 ---
    print("\n" + "=" * 80)
    print(md)
    print("=" * 80 + "\n")

    if save:
        result_dir = ensure_results_dir()
        out_path = os.path.join(result_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[OK] Summary saved -> {out_path}")


# ---------------------------
# CLI 파서
# ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_reader",
        description="Summarize papers/blogs from URL or PDF (LLM-powered)"
    )
    p.add_argument("target", help="요약 대상: URL 또는 PDF 경로")
    p.add_argument("--template", help="요약 형식 템플릿 파일 경로 (기본: templates/SUMMARY.md)")
    p.add_argument("--pages", help='PDF 페이지 범위 (예: "1,3-5")')
    p.add_argument("--engine", default="pdfminer", choices=["pdfminer", "pypdf", "auto"], help="PDF 텍스트 추출 엔진 선호")
    p.add_argument("--ocr", action="store_true", help="스캔 PDF OCR 사용 (pytesseract+pdf2image 필요)")
    p.add_argument("--no-save", action="store_true", help="결과 파일 저장하지 않음")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(
            args.target,
            pages=args.pages,
            engine=args.engine,
            ocr=args.ocr,
            save=not args.no_save,
            template=args.template,  # 기본값은 내부에서 처리
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
