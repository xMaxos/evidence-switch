"""Optional, sequential runner for OpenAI-compatible chat endpoints."""

import ipaddress
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _read_prompts(path):
    prompts = []
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                prompt = json.loads(line, object_pairs_hook=_unique_object)
            except ValueError:
                raise ValueError(f"Invalid prompt JSON on line {line_number}") from None
            valid = (
                isinstance(prompt, dict)
                and set(prompt) == {"id", "messages"}
                and isinstance(prompt["id"], str)
                and bool(prompt["id"].strip())
                and isinstance(prompt["messages"], list)
                and bool(prompt["messages"])
            )
            if not valid:
                raise ValueError(f"Invalid prompt structure on line {line_number}")
            if prompt["id"] in ids:
                raise ValueError(f"Duplicate prompt ID on line {line_number}")
            for message in prompt["messages"]:
                if not (
                    isinstance(message, dict)
                    and set(message) == {"role", "content"}
                    and isinstance(message["role"], str)
                    and message["role"] in {"system", "user"}
                    and isinstance(message["content"], str)
                ):
                    raise ValueError(f"Invalid prompt message on line {line_number}")
            # Validate encoding for every prompt before making any network request.
            try:
                json.dumps(prompt, ensure_ascii=False).encode("utf-8")
            except UnicodeError:
                raise ValueError(f"Invalid prompt text on line {line_number}") from None
            ids.add(prompt["id"])
            prompts.append(prompt)
    if not prompts:
        raise ValueError("Prompt file contains no prompts")
    return prompts


def _endpoint(base_url):
    if not isinstance(base_url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in base_url):
        raise ValueError("Invalid base URL")
    try:
        url = urlsplit(base_url)
        host = url.hostname
        port = url.port
        if not host or url.username is not None or url.password is not None:
            raise ValueError
        if url.query or url.fragment or (port is not None and port == 0):
            raise ValueError
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if url.scheme != "https" and not (url.scheme == "http" and loopback):
            raise ValueError
        base_url.encode("ascii")
    except (ValueError, UnicodeError):
        raise ValueError("Base URL must use HTTPS, or HTTP on a loopback host") from None
    return base_url.rstrip("/") + "/chat/completions"


def _api_key(api_key_env):
    if api_key_env is None:
        return None
    if not isinstance(api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError("Invalid API key environment variable name")
    key = os.environ.get(api_key_env)
    if not key or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise ValueError("API key environment variable is missing or contains invalid characters")
    return key


def _complete(opener, endpoint, model, messages, key, timeout):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "stream": False}).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    request = Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ValueError
        data = json.loads(raw.decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError
        content.encode("utf-8")
        return content
    except HTTPError as error:
        status = error.code
        error.close()
        raise RuntimeError(f"Model request failed with HTTP status {status}") from None
    except Exception:
        # Endpoint errors can reflect Authorization headers or secret-bearing bodies.
        raise RuntimeError("Model request failed or returned an invalid response") from None


def _prepare_run(prompts_path, output_path, model, limit, timeout, strategy, trace_path):
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model must be a nonempty string")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Limit must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be a positive finite number")
    if strategy not in ("direct", "two-stage", "two-stage-ids", "two-stage-retrieval"):
        raise ValueError("Strategy must be direct, two-stage, two-stage-ids or two-stage-retrieval")
    paths = [Path(prompts_path), Path(output_path)]
    if trace_path is not None:
        paths.append(Path(trace_path))
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input, prediction and trace paths must differ")
    if any(path.exists() for path in paths[1:]):
        raise FileExistsError("Prediction or trace output already exists")
    prompts = _read_prompts(paths[0])
    if strategy in ("two-stage", "two-stage-ids", "two-stage-retrieval"):
        from .workflow import prepare
        if strategy == "two-stage-ids":
            from .selection import prepare_ids as prepare
        elif strategy == "two-stage-retrieval":
            from .retrieval import prepare_retrieval as prepare
        # Validate every template before requests, including entries beyond limit.
        prepared = [prepare(prompt["messages"]) for prompt in prompts]
    else:
        prepared = [None] * len(prompts)
    return list(zip(prompts[:limit], prepared[:limit]))


def _messages_digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()


def _execute(rows, output_path, complete, strategy, trace_path=None, environment=None):
    """Shared sequential workflow; complete returns (raw content, call metadata)."""
    count = 0
    with ExitStack() as stack:
        output = stack.enter_context(Path(output_path).open("x", encoding="utf-8", newline="\n"))
        trace_file = (stack.enter_context(Path(trace_path).open("x", encoding="utf-8", newline="\n"))
                      if trace_path is not None else None)
        for prompt, prepared in rows:
            start = time.perf_counter()
            trace = {"id": prompt["id"], "strategy": strategy, "calls": [],
                     "original_messages_sha256": _messages_digest(prompt["messages"])}

            def call(messages):
                called = time.perf_counter()
                content, metadata = complete(messages)
                trace["calls"].append(dict(metadata, messages_sha256=_messages_digest(messages),
                                           wall_seconds=time.perf_counter() - called))
                return content

            if prepared is None:
                content = call(prompt["messages"])
            else:
                extraction = call(prepared.extraction_messages)
                trace.update(extraction_response=extraction, extraction_valid=False)
                try:
                    final_messages = prepared.finish(extraction)
                except (ValueError, RecursionError):
                    # An invalid extraction is a scored failure, never repaired or retried.
                    content = {"error": "invalid_extraction"}
                else:
                    trace["extraction_valid"] = True
                    content = call(final_messages)
            output.write(json.dumps({"id": prompt["id"], "response": content}, ensure_ascii=False) + "\n")
            output.flush()
            if trace_file is not None:
                trace["wall_seconds"] = time.perf_counter() - start
                if environment is not None:
                    trace["environment"] = environment
                trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_file.flush()
            count += 1
    return count


def run_prompts(prompts_path, output_path, *, base_url, model, api_key_env=None, limit=None,
                timeout=60.0, strategy="direct", trace_path=None):
    """Send prompt-only JSONL. No overwrite, retries, redirects or implicit proxies.

    Completed predictions survive later request failures. Two-stage mode supports
    generated EvidenceSwitch prompts only; its extraction is validated before a
    second call with the same model. Endpoint use may incur provider charges.
    """
    endpoint = _endpoint(base_url)
    rows = _prepare_run(prompts_path, output_path, model, limit, timeout, strategy, trace_path)
    key = _api_key(api_key_env)
    # Ignore environment proxies; never send credentials through an implicit proxy.
    opener = build_opener(ProxyHandler({}), _NoRedirects())
    def complete(messages):
        return _complete(opener, endpoint, model, messages, key, timeout), {"model": model}
    return _execute(rows, output_path, complete, strategy, trace_path)
