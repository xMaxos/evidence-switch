import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from evidenceswitch.runner import run_prompts


@contextlib.contextmanager
def endpoint(responses):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
            })
            status, body, headers = responses[min(len(requests) - 1, len(responses) - 1)]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def do_GET(self):
            requests.append({"path": self.path, "method": "GET"})
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def completion(text='{"answer":"blue","citations":["r1"]}'):
    return 200, json.dumps({"choices": [{"message": {"content": text}}]}), {}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.prompts = Path(self.directory.name) / "prompts.jsonl"
        self.output = Path(self.directory.name) / "predictions.jsonl"
        self.prompt = {"id": "opaque-1", "messages": [
            {"role": "system", "content": "Use only supplied records."},
            {"role": "user", "content": "What is the color? [r1] color: blue"},
        ]}
        self.write_prompts([self.prompt])

    def write_prompts(self, prompts):
        self.prompts.write_text("".join(json.dumps(p) + "\n" for p in prompts), encoding="utf-8")

    def run_runner(self, url, **kwargs):
        return run_prompts(self.prompts, self.output, base_url=url, model="test-model", **kwargs)

    def test_request_contains_messages_only_and_saves_raw_content(self):
        text = '{"answer":"blau","citations":["r1"]}'
        with endpoint([completion(text)]) as (url, requests):
            self.assertEqual(self.run_runner(url), 1)
        self.assertEqual(requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(requests[0]["body"], {
            "model": "test-model", "messages": self.prompt["messages"],
            "temperature": 0, "stream": False,
        })
        self.assertNotIn("Authorization", requests[0]["headers"])
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), {"id": "opaque-1", "response": text})

    def test_entire_file_validated_before_request_even_with_limit(self):
        invalid = {**self.prompt, "id": "opaque-2", "expected": {"answer": "gold"}}
        self.write_prompts([self.prompt, invalid])
        with endpoint([completion()]) as (url, requests):
            with self.assertRaises(ValueError):
                self.run_runner(url, limit=1)
            self.assertEqual(requests, [])
        self.assertFalse(self.output.exists())

    def test_rejects_extra_message_fields_invalid_roles_and_duplicate_ids(self):
        for message in [
            {"role": "user", "content": "test", "expected": "gold"},
            {"role": "assistant", "content": "gold"},
            {"role": [], "content": "test"},
            {"role": "user", "content": 12},
        ]:
            with self.subTest(message=message):
                self.write_prompts([{**self.prompt, "messages": [message]}])
                with self.assertRaises(ValueError):
                    self.run_runner("http://127.0.0.1:1")
        self.write_prompts([self.prompt, self.prompt])
        with self.assertRaises(ValueError):
            self.run_runner("http://127.0.0.1:1")
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_json_keys(self):
        self.prompts.write_text('{"id":"x","id":"y","messages":[]}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_runner("http://127.0.0.1:1")

    def test_refuses_overwrite_and_same_path(self):
        self.output.write_text("preserve me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_runner("http://127.0.0.1:1")
        self.assertEqual(self.output.read_text(encoding="utf-8"), "preserve me")
        with self.assertRaises(ValueError):
            run_prompts(self.prompts, self.prompts, base_url="http://127.0.0.1:1", model="test")

    def test_https_required_except_loopback_and_url_credentials_rejected(self):
        for url in ["http://example.com/v1", "http://127.0.0.1.evil.test", "ftp://127.0.0.1", "https://user:secret@example.com", "https://example.com?q=secret", "https://example.com#fragment", "https://example.com\n", "https://example.com:bad"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.run_runner(url)
        self.assertFalse(self.output.exists())

    def test_partial_results_survive_failure_without_exposing_key_or_body(self):
        secret = "local-test-secret"
        self.write_prompts([self.prompt, {**self.prompt, "id": "opaque-2"}])
        with patch.dict(os.environ, {"EVIDENCESWITCH_TEST_KEY": secret}):
            with endpoint([completion(), (500, secret + " private body", {})]) as (url, requests):
                with self.assertRaises(RuntimeError) as caught:
                    self.run_runner(url, api_key_env="EVIDENCESWITCH_TEST_KEY")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["headers"]["Authorization"], "Bearer " + secret)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn("private body", str(caught.exception))
        saved = self.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0])["id"], "opaque-1")

    def test_redirects_are_not_followed(self):
        with endpoint([(302, "", {"Location": "/stolen"})]) as (url, requests):
            with self.assertRaisesRegex(RuntimeError, "302"):
                self.run_runner(url)
            self.assertEqual(len(requests), 1)

    def test_malformed_or_missing_model_content_is_sanitized(self):
        for body in ["private-body-not-json", "{}", '{"choices":[{"message":{"content":null}}]}']:
            with self.subTest(body=body):
                self.output.unlink(missing_ok=True)
                with endpoint([(200, body, {})]) as (url, requests):
                    with self.assertRaises(RuntimeError) as caught:
                        self.run_runner(url)
                self.assertNotIn(body, str(caught.exception))

    def test_limit_and_configuration_validation(self):
        self.write_prompts([self.prompt, {**self.prompt, "id": "opaque-2"}])
        with endpoint([completion()]) as (url, requests):
            self.assertEqual(self.run_runner(url, limit=1), 1)
        self.assertEqual(len(requests), 1)
        for kwargs in [{"limit": 0}, {"limit": True}, {"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")}, {"api_key_env": "KEY\nBAD"}]:
            with self.subTest(kwargs=kwargs):
                self.output.unlink(missing_ok=True)
                with self.assertRaises(ValueError):
                    self.run_runner("http://127.0.0.1:1", **kwargs)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            self.run_runner("http://127.0.0.1:1", api_key_env="MISSING_KEY")


if __name__ == "__main__":
    unittest.main()
