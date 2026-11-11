"""Model-agnostic structured-output client: swap provider/model by config or env, nothing else changes.

    LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini python cli.py ...
    LLM_PROVIDER=mock                      -> offline heuristic stand-in, for smoke tests only

Debug features:
  * every API attempt is logged (llm.log) with schema / status / seconds / error
  * the first request+answer per schema is kept (llm.samples) so you can SEE what the model returned
  * errors are classified with a plain-language hint (classify_error)
  * fatal errors (bad key, wrong model name, no quota) fail fast instead of retrying 100 times
  * an EMPTY structured answer is treated as an error (and never cached)
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Type, TypeVar

from pydantic import BaseModel

from .config import Settings, resolve_dir

T = TypeVar("T", bound=BaseModel)

# env var each provider reads its key from ("" = no key needed)
PROVIDER_KEY_ENV = {
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mock": "",
}
# providers that are OpenAI-compatible: (langchain provider, default base_url)
_OPENAI_COMPAT = {
    "together": ("openai", "https://api.together.xyz/v1"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1"),
}
_PIP = {
    "google_genai": "langchain-google-genai", "openai": "langchain-openai", "anthropic": "langchain-anthropic",
    "groq": "langchain-groq", "mistralai": "langchain-mistralai", "together": "langchain-openai",
    "openrouter": "langchain-openai",
}


class FatalLLMError(RuntimeError):
    """Retrying cannot help (bad key, unknown model, no quota, missing package)."""


# ------------------------------------------------------------------ error classification
_RULES = [  # (code, fatal, substrings (lower-case), hint)
    ("no_key", True, ["no api key", "api key not set", "api_key not set"],
     "API key is missing. Put it in .env (GOOGLE_API_KEY=...) or paste it in the sidebar."),
    ("bad_key", True, ["api_key_invalid", "api key not valid", "invalid api key", "incorrect api key", "401",
                       "unauthenticated", "authentication", "permission_denied", "403", "forbidden"],
     "The key was rejected. Check it is for THIS provider, has no extra spaces/quotes, and the API is enabled."),
    ("bad_model", True, ["404", "not found", "is not supported for", "unknown model", "model_not_found",
                         "does not exist", "no such model"],
     "Model name not accepted by this provider. Check the exact model id (e.g. gemini-2.5-flash)."),
    ("no_quota", True, ["insufficient_quota", "billing", "exceeded your current quota", "quota exceeded",
                        "limit: 0"],
     "Quota / billing problem on this key. Use another key / provider, or enable billing."),
    ("missing_pkg", True, ["no module named", "could not import", "unable to import", "install it with"],
     "A provider package is missing: pip install langchain-google-genai (or the one for your provider)."),
    ("rate_limit", False, ["429", "resource_exhausted", "rate limit", "rate_limit", "too many requests"],
     "Rate limit. Lower 'Parallel calls' to 1-2 and set 'Max requests / minute' to ~10 (free tiers)."),
    ("empty", False, ["empty structured", "returned no structured", "returned none", "empty answer"],
     "Model answered but with nothing usable. Try a stronger model, or check llm.samples for the raw answer."),
    ("bad_output", False, ["validation error", "outputparserexception", "json", "parse", "field required"],
     "Model output did not match the schema. A stronger model, or temperature 0, usually fixes this."),
    ("timeout", False, ["timeout", "timed out", "deadline", "503", "unavailable", "connection", "network"],
     "Network / server problem (temporary). Re-run: results are cached, finished calls are not repeated."),
]


def classify_error(msg: str) -> tuple[str, str]:
    """-> (code, plain-language hint)."""
    m = (msg or "").lower()
    for code, _fatal, needles, hint in _RULES:
        if any(n in m for n in needles):
            return code, hint
    return "unknown", "Unknown error. Run `python debug_check.py --input <file>` and read the first failing step."


def _is_fatal(msg: str) -> bool:
    m = (msg or "").lower()
    return any(fatal and any(n in m for n in needles) for _c, fatal, needles, _h in _RULES)


# ------------------------------------------------------------------ model construction
def _key_env(provider: str) -> str:
    return PROVIDER_KEY_ENV.get(provider, "")


def get_llm(schema: Type[T], cfg: Settings, max_retries: int | None = None):
    """LangChain chat model that returns `schema` instances."""
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as e:  # noqa: BLE001
        raise FatalLLMError(f"could not import langchain: {e}. pip install -r requirements.txt") from e
    provider = cfg.provider
    kw: dict = {"temperature": cfg.temperature,
                "max_retries": cfg.max_retries if max_retries is None else max_retries}
    lc_provider = provider
    if provider in _OPENAI_COMPAT:
        lc_provider, default_url = _OPENAI_COMPAT[provider]
        kw["base_url"] = cfg.base_url or default_url
        kw["api_key"] = os.environ.get(_key_env(provider), "")
    elif provider == "openai" and cfg.base_url:
        kw["base_url"] = cfg.base_url
    env = _key_env(provider)
    if env and not os.environ.get(env):
        raise FatalLLMError(f"no API key: environment variable {env} is empty (provider={provider})")
    try:
        llm = init_chat_model(cfg.model, model_provider=lc_provider, **kw)
    except ImportError as e:  # noqa: BLE001
        raise FatalLLMError(f"could not import provider package: {e}. pip install {_PIP.get(provider, '')}") from e
    except Exception as e:  # noqa: BLE001
        raise FatalLLMError(f"could not build model {provider}:{cfg.model}: {e!r}") from e
    return llm.with_structured_output(schema)


class _RateLimiter:
    def __init__(self, rpm: int):
        self.gap = 60.0 / rpm if rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self) -> None:
        if not self.gap:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


ALLOW_EMPTY = {"ComboOut"}   # "no good combos" is a legitimate answer


def _is_empty(out: BaseModel) -> bool:
    """True when the model 'answered' but every field is empty: that is a failure, not a result."""
    if type(out).__name__ in ALLOW_EMPTY:
        return False
    vals = list(out.model_dump().values())
    return bool(vals) and all(v in ("", None, [], {}) for v in vals)


class LLMClient:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.provider, self.model = cfg.provider, cfg.model
        self.cache_dir = resolve_dir(cfg.cache_dir)
        if cfg.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = _RateLimiter(cfg.rpm)
        self.lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0}
        self.log: list[dict] = []          # one row per API attempt
        self.samples: dict = {}            # first request/answer per schema, for eyeballing
        self._chains: dict[str, object] = {}
        self._fatal: str | None = None     # once set, no more API calls are attempted

    # ---- cache: key = hash(provider + model + prompt), one small JSON file per key (thread-safe)
    def _key(self, schema: Type[T], system: str, user: str) -> str:
        raw = "\x00".join([self.provider, self.model, str(self.cfg.temperature), schema.__name__, system, user])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _chain(self, schema: Type[T]):
        with self.lock:
            if schema.__name__ not in self._chains:
                self._chains[schema.__name__] = get_llm(schema, self.cfg)
            return self._chains[schema.__name__]

    def warmup(self) -> None:
        """Build every model on the main thread so a misconfiguration fails HERE, with a clear message."""
        if self.provider == "mock":
            return
        from .schemas import ComboOut, EpisodeOut, FactCheck, Recap, Score
        for s in (EpisodeOut, ComboOut, Score, Recap, FactCheck):
            self._chain(s)

    def _record(self, schema: str, status: str, seconds: float, error: str = "") -> None:
        with self.lock:
            self.log.append({"schema": schema, "status": status, "seconds": round(seconds, 2), "error": error[:400]})

    def _sample(self, schema: Type[T], system: str, user: str, out: BaseModel) -> None:
        with self.lock:
            if schema.__name__ not in self.samples:
                self.samples[schema.__name__] = {"system": system[:1500], "user": user[:2500],
                                                 "answer": out.model_dump()}

    def invoke(self, schema: Type[T], system: str, user: str) -> T:
        name = schema.__name__
        if self._fatal:
            raise FatalLLMError(self._fatal)
        key = self._key(schema, system, user)
        path = self.cache_dir / f"{key}.json"
        if self.cfg.use_cache and path.exists():
            try:
                out = schema.model_validate_json(path.read_text(encoding="utf-8"))
                with self.lock:
                    self.stats["cache_hits"] += 1
                self._record(name, "cache", 0.0)
                return out
            except Exception:  # noqa: BLE001 - corrupt cache entry -> recompute
                pass

        last = ""
        for attempt in range(self.cfg.call_retries + 1):
            t0 = time.time()
            try:
                self.limiter.wait()
                if self.provider == "mock":
                    from .mock_llm import mock_respond
                    out = mock_respond(schema, system, user)
                else:
                    res = self._chain(schema).invoke([("system", system), ("human", user)])
                    if res is None:
                        raise ValueError("model returned None (no structured output)")
                    out = res if isinstance(res, schema) else schema.model_validate(res)
                if _is_empty(out):
                    raise ValueError("empty structured answer: every field is empty")
            except FatalLLMError as e:
                self._fatal = str(e)
                self._record(name, "error: fatal", time.time() - t0, str(e))
                with self.lock:
                    self.stats["failures"] += 1
                raise
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                self._record(name, f"error (attempt {attempt + 1})", time.time() - t0, last)
                if _is_fatal(last):
                    self._fatal = f"{last}\nHINT: {classify_error(last)[1]}"
                    with self.lock:
                        self.stats["failures"] += 1
                    raise FatalLLMError(self._fatal) from e
                time.sleep(min(2 ** attempt, 8))
                continue
            with self.lock:
                self.stats["calls"] += 1
            self._record(name, "ok", time.time() - t0)
            self._sample(schema, system, user, out)
            if self.cfg.use_cache:
                path.write_text(out.model_dump_json(), encoding="utf-8")
            return out
        with self.lock:
            self.stats["failures"] += 1
        raise RuntimeError(f"LLM call failed ({name}): {last}")

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


# ------------------------------------------------------------------ connection test
class _Ping(BaseModel):
    answer: str
    number: int


def preflight(cfg: Settings) -> dict:
    """One tiny structured call, no retries. -> {"ok": bool, "steps": [{"name", "ok", "detail"}]}"""
    steps: list[dict] = []

    def step(name: str, ok: bool, detail: str) -> bool:
        steps.append({"name": name, "ok": ok, "detail": detail})
        return ok

    env = _key_env(cfg.provider)
    if cfg.provider == "mock":
        step("provider", True, "mock: offline stand-in, no API involved")
        return {"ok": True, "steps": steps}
    if cfg.provider not in PROVIDER_KEY_ENV:
        step("provider", True, f"'{cfg.provider}' is not in the known list; trying it via langchain anyway")
    if env:
        k = os.environ.get(env, "")
        if not step(f"API key ({env})", bool(k),
                    f"found, {len(k)} chars, starts with '{k[:4]}...'" if k else
                    f"{env} is empty. Put it in .env or the sidebar, then restart/rerun."):
            return {"ok": False, "steps": steps}
        if k != k.strip() or k[:1] in "\"'":
            step("API key format", False, "key has spaces or quote characters around it; remove them in .env")
            return {"ok": False, "steps": steps}
    try:
        chain = get_llm(_Ping, cfg, max_retries=0)
        step("build model", True, f"{cfg.provider}:{cfg.model}")
    except Exception as e:  # noqa: BLE001
        step("build model", False, f"{e}\nHINT: {classify_error(str(e))[1]}")
        return {"ok": False, "steps": steps}
    t0 = time.time()
    try:
        res = chain.invoke([("system", "Reply with the requested fields."),
                            ("human", "answer = the word hello, number = 7")])
        if res is None:
            step("API call", False, "call went through but the model returned NOTHING (None). "
                                    "Usually: wrong/too small model for structured output, or a safety block.")
            return {"ok": False, "steps": steps}
        step("API call", True, f"answered in {time.time() - t0:.1f}s: {res!r}")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        step("API call", False, f"{msg[:600]}\nHINT: {classify_error(msg)[1]}")
        return {"ok": False, "steps": steps}
    return {"ok": True, "steps": steps}
