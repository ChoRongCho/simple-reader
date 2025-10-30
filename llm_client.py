from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass
from typing import Dict, Iterator, AsyncIterator, Optional, List, Literal, Any

from config_loader import load_env
from crawler import crawl
from pdf_reader import read_pdf

Provider = Literal["openai", "ollama"]


# -------- 템플릿 유틸 --------
def apply_template(template: str, *, title: str = "", content: str = "", meta: Dict[str, Any] | None = None) -> str:
    """
    아주 단순한 placeholder 치환:
      - {{title}}, {{content}} 기본 제공
      - {{meta.key}} 형태로 메타도 치환 가능
    """
    meta = meta or {}
    out = template.replace("{{title}}", title).replace("{{content}}", content)
    # {{meta.xxx}} 치환
    for k, v in meta.items():
        out = out.replace(f"{{{{meta.{k}}}}}", str(v))
    return out

# llm_client.py 상단 유틸 근처에 추가
def apply_template_partial(template: str, *, content: str = "", meta: Dict[str, Any] | None = None) -> str:
    """
    title은 건드리지 않고, {{content}}, {{meta.*}}만 치환한다.
    """
    meta = meta or {}
    out = template.replace("{{content}}", content)
    for k, v in meta.items():
        out = out.replace(f"{{{{meta.{k}}}}}", str(v))
    return out

# -------- 설정 --------
@dataclass
class LLMConfig:
    provider: Provider = "openai"
    model: str = "gpt-4o-mini"  # 자유 변경
    temperature: float = 0.2
    max_tokens: int = 8192

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # Ollama
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# -------- 인터페이스 --------
class BaseLLMClient:
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> Iterator[str]:
        raise NotImplementedError

    async def generate_async(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    async def stream_async(self, prompt: str, system: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError


# -------- OpenAI 구현 --------
class OpenAIClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI  # lazy import
        api_key = cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 가 필요합니다.")
        self._client = OpenAI(api_key=api_key, base_url=cfg.openai_base_url or os.environ.get("OPENAI_BASE_URL"))
        self.cfg = cfg

    def _messages(self, prompt: str, system: Optional[str]) -> List[Dict[str, str]]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        params = {
            "model": kwargs.get("model", self.cfg.model),
            "messages": self._messages(prompt, system),
            "temperature": kwargs.get("temperature", self.cfg.temperature),
            "max_tokens": kwargs.get("max_tokens", self.cfg.max_tokens),
        }
        # Chat Completions API (안정적)
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> Iterator[str]:
        params = {
            "model": kwargs.get("model", self.cfg.model),
            "messages": self._messages(prompt, system),
            "temperature": kwargs.get("temperature", self.cfg.temperature),
            "max_tokens": kwargs.get("max_tokens", self.cfg.max_tokens),
            "stream": True,
        }
        stream = self._client.chat.completions.create(**params)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def generate_async(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        # openai 공식 비동기 클라이언트가 없다면 스레드로 감싸기
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, system, **kwargs)

    async def stream_async(self, prompt: str, system: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        # 간단 비동기 래핑
        def _sync_stream():
            for t in self.stream(prompt, system, **kwargs):
                yield t

        queue: asyncio.Queue[str] = asyncio.Queue()

        def _run():
            for token in _sync_stream():
                asyncio.run_coroutine_threadsafe(queue.put(token), asyncio.get_event_loop())
            asyncio.run_coroutine_threadsafe(queue.put("__STREAM_END__"), asyncio.get_event_loop())

        import threading
        th = threading.Thread(target=_run, daemon=True)
        th.start()

        while True:
            item = await queue.get()
            if item == "__STREAM_END__":
                break
            yield item


# -------- Ollama 구현 (로컬 모델) --------
class OllamaClient(BaseLLMClient):
    def __init__(self, cfg: LLMConfig):
        import requests  # lazy import
        self.requests = requests
        self.cfg = cfg

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        url = f"{self.cfg.ollama_base_url}/api/generate"
        model = kwargs.get("model", self.cfg.model)
        payload = {
            "model": model,
            "prompt": prompt if not system else f"<<SYS>>\n{system}\n<</SYS>>\n{prompt}",
            "options": {"temperature": kwargs.get("temperature", self.cfg.temperature)},
            "stream": False,
        }
        r = self.requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    def stream(self, prompt: str, system: Optional[str] = None, **kwargs) -> Iterator[str]:
        url = f"{self.cfg.ollama_base_url}/api/generate"
        model = kwargs.get("model", self.cfg.model)
        payload = {
            "model": model,
            "prompt": prompt if not system else f"<<SYS>>\n{system}\n<</SYS>>\n{prompt}",
            "options": {"temperature": kwargs.get("temperature", self.cfg.temperature)},
            "stream": True,
        }
        with self.requests.post(url, json=payload, timeout=0, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line.decode("utf-8"))
                token = obj.get("response")
                if token:
                    yield token

    async def generate_async(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, system, **kwargs)

    async def stream_async(self, prompt: str, system: Optional[str] = None, **kwargs) -> AsyncIterator[str]:
        def _sync_stream():
            for t in self.stream(prompt, system, **kwargs):
                yield t

        queue: asyncio.Queue[str] = asyncio.Queue()

        def _run():
            for token in _sync_stream():
                asyncio.run_coroutine_threadsafe(queue.put(token), asyncio.get_event_loop())
            asyncio.run_coroutine_threadsafe(queue.put("__STREAM_END__"), asyncio.get_event_loop())

        import threading
        th = threading.Thread(target=_run, daemon=True)
        th.start()

        while True:
            item = await queue.get()
            if item == "__STREAM_END__":
                break
            yield item


# -------- 팩토리 --------
def create_llm_client(cfg: LLMConfig) -> BaseLLMClient:
    if cfg.provider == "openai":
        return OpenAIClient(cfg)
    elif cfg.provider == "ollama":
        return OllamaClient(cfg)
    else:
        raise ValueError(f"지원하지 않는 provider: {cfg.provider}")


# -------- 고수준 요약 함수 --------
DEFAULT_SUMMARY_TMPL = """# {{title}}

## TL;DR
- 핵심 한 줄 요약

## 핵심 포인트
- 항목 1
- 항목 2
- 항목 3

## 상세 요약
{{content}}

## 메타
- 길이: {{meta.length}}
- 소스: {{meta.source}}
"""



# 기존 summarize_text() 교체
def summarize_text(
    client: BaseLLMClient,
    raw_text: str,
    *,
    title: str | None = None,                 # 힌트로만 사용 (auto_title=True면 무시)
    template_path: str | None = None,         # 템플릿 파일 경로
    meta: Optional[Dict[str, Any]] = None,
    system_prompt: str = (
        "You are a concise research assistant. "
        "Follow the given template strictly. Output in Korean, except the title in English."
    ),
    auto_title: bool = True,                   # LLM이 제목 결정
    title_language: str = "en",                # en/ko 등
    **gen_kwargs,
) -> str:
    import os
    meta = meta or {}
    meta.setdefault("length", len(raw_text))
    meta.setdefault("source", "unknown")

    # 1) 템플릿 불러오기
    if template_path and os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template_md = f.read()
    else:
        template_md = DEFAULT_SUMMARY_TMPL  # 내부 기본값

    # 2) 제목 처리: auto_title이면 {{title}}를 남겨두고, 나머지만 치환
    if auto_title:
        template_instr = apply_template_partial(
            template_md,
            content="<여기에 상세 요약 본문을 한국어로 채워 넣어라>",
            meta=meta,
        )
        title_directive = (
            f"Replace {{title}} in the template with your own concise {title_language.upper()} title "
            f"(6–12 words, no emojis)."
        )
    else:
        # 사용자가 준 제목 힌트로 채움
        template_instr = apply_template(
            template_md,
            title=title or "Summary",
            content="<여기에 상세 요약 본문을 한국어로 채워 넣어라>",
            meta=meta,
        )
        title_directive = "Use the provided {{title}} as-is."

    # 3) 프롬프트
    prompt = (
        "Read the ORIGINAL TEXT and write a summary strictly following the TEMPLATE structure.\n"
        "Do not add or remove sections. Keep headings exactly as in the template.\n"
        f"{title_directive}\n"
        f"- Body language: Korean.\n"
        f"- Title language: {title_language.upper()}.\n\n"
        f"--- TEMPLATE ---\n{template_instr}\n"
        f"\n--- ORIGINAL TEXT ---\n{raw_text}\n"
    )

    return client.generate(prompt, system=system_prompt, **gen_kwargs)



# -------- 사용 예시 --------
if __name__ == "__main__":
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

    doc = read_pdf("paper.pdf", pages="1-5", prefer="pdfminer")
    md = summarize_text(
        client,
        doc.text,
        title=doc.title or "PDF Summary",
        meta={"source": doc.final_source, "length": len(doc.text)},
    )
    print(md)