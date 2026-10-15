"""Native local Ollama runner with explicit, reproducible inference settings."""

import ipaddress
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .runner import _NoRedirects, _MAX_RESPONSE_BYTES, _execute, _prepare_run


def _local_url(base_url):
    try:
        if not isinstance(base_url, str) or any(ord(c) <= 32 or ord(c) >= 127 for c in base_url):
            raise ValueError
        url = urlsplit(base_url)
        host = url.hostname
        if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
            raise ValueError
        if (url.scheme not in ("http", "https") or url.username is not None or url.password is not None
                or url.query or url.fragment or url.path not in ("", "/") or url.port == 0):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Ollama URL must be a loopback origin without credentials or a path") from None
    return base_url.rstrip("/")


def _request(opener, url, body, timeout):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError
        result = json.loads(raw.decode("utf-8"))
        if not isinstance(result, dict) or "error" in result:
            raise ValueError
        return result
    except HTTPError as error:
        status = error.code
        error.close()
        raise RuntimeError(f"Ollama request failed with HTTP status {status}") from None
    except Exception:
        raise RuntimeError("Ollama request failed or returned an invalid response") from None


def run_ollama(prompts_path, output_path, *, model, base_url="http://127.0.0.1:11434",
               strategy="direct", trace_path=None, limit=None, timeout=180.0,
               seed=20261015, num_ctx=4096, num_predict=512):
    """Use an already installed local model. Never download or use cloud models.

    Each call disables thinking and uses temperature zero. Two-stage mode makes
    at most two calls per prompt, each with its own num_predict token allowance.
    """
    base_url = _local_url(base_url)
    for name, value in (("seed", seed), ("num_ctx", num_ctx), ("num_predict", num_predict)):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
    rows = _prepare_run(prompts_path, output_path, model, limit, timeout, strategy, trace_path)
    opener = build_opener(ProxyHandler({}), _NoRedirects())
    options = {"temperature": 0, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict}

    def request(path, body=None):
        return _request(opener, base_url + path, body, timeout)

    # An explicit inventory check prevents implicit pulls; show rejects cloud aliases.
    models = request("/api/tags").get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama returned an invalid model inventory")
    installed = next((entry for entry in models if isinstance(entry, dict)
                      and entry.get("name") in (model, model + ":latest")), None)
    if installed is None:
        raise ValueError("Model is not installed locally; this runner never downloads models")
    details = request("/api/show", {"model": model})
    if any(item.get("remote_host") or item.get("remote_model") for item in (installed, details)):
        raise ValueError("Cloud models are not allowed by the local Ollama runner")
    environment = {"provider": "ollama", "model": model, "digest": installed.get("digest"),
                   "ollama_version": request("/api/version").get("version"),
                   "think": False, "options": options}

    def complete(messages):
        result = request("/api/chat", {"model": model, "messages": messages, "stream": False,
                                       "think": False, "keep_alive": "5m", "options": options})
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or result.get("done") is not True:
            raise RuntimeError("Ollama returned an incomplete or invalid response")
        try:
            content.encode("utf-8")
        except UnicodeError:
            raise RuntimeError("Ollama returned invalid response text") from None
        # Store timing/token metadata, never hidden reasoning content.
        stats = {key: result.get(key) for key in (
            "model", "created_at", "done", "done_reason", "total_duration", "load_duration",
            "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")}
        stats["thinking_returned"] = bool(message.get("thinking"))
        return content, stats

    return _execute(rows, output_path, complete, strategy, trace_path, environment)
