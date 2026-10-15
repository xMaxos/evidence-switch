import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from evidenceswitch.generate import generate_cases
from evidenceswitch.io import read_jsonl
from evidenceswitch.ollama import run_ollama
from evidenceswitch.runner import run_prompts
from evidenceswitch.score import score
from evidenceswitch.workflow import prepare
from tests.test_runner import completion, endpoint


def native_completion(content, **changes):
    return dict({"model": "test-model", "message": {"content": content},
                 "done": True, "done_reason": "stop", "eval_count": 12}, **changes)


@contextlib.contextmanager
def native_endpoint(chat_responses):
    requests = []
    chats = iter(chat_responses)

    class Handler(BaseHTTPRequestHandler):
        def respond(self):
            body = (json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    if self.command == "POST" else None)
            requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
            responses = {"/api/tags": {"models": [{"name": "test-model", "digest": "local-digest"}]},
                         "/api/show": {"details": {}}, "/api/version": {"version": "test-version"}}
            response = next(chats) if self.path == "/api/chat" else responses[self.path]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            raw = response if isinstance(response, str) else json.dumps(response)
            self.wfile.write(raw.encode("utf-8"))

        do_GET = do_POST = respond

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class OllamaTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.prompts = self.directory / "prompts.jsonl"
        self.output = self.directory / "predictions.jsonl"
        self.trace = self.directory / "trace.jsonl"
        self.cases = generate_cases(seed=7, groups=1, languages=("en",))
        self.prompt = {key: self.cases[0][key] for key in ("id", "messages")}
        self.records = json.loads(self.prompt["messages"][1]["content"].split("\n\nRecords:\n")[1])
        self.write_prompts([self.prompt])

    def write_prompts(self, rows):
        self.prompts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def run_native(self, **kwargs):
        return run_ollama(self.prompts, self.output, model="test-model", **kwargs)

    def test_native_two_stage_payload_preflight_and_trace(self):
        extraction = json.dumps({"records": self.records[:2]})
        answer = json.dumps(self.cases[0]["expected"])
        hidden = "private-reasoning-must-not-be-saved"
        first = native_completion(extraction, message={"content": extraction, "thinking": hidden})
        with native_endpoint([first, native_completion(answer)]) as (url, requests):
            self.assertEqual(self.run_native(base_url=url, strategy="two-stage", trace_path=self.trace), 1)
        self.assertEqual([request["path"] for request in requests],
                         ["/api/tags", "/api/show", "/api/version", "/api/chat", "/api/chat"])
        self.assertEqual(requests[1]["body"], {"model": "test-model"})
        prepared = prepare(self.prompt["messages"])
        for request, messages in zip(requests[3:], (prepared.extraction_messages, prepared.finish(extraction))):
            self.assertEqual(request["body"], {"model": "test-model", "messages": messages,
                "stream": False, "think": False, "keep_alive": "5m",
                "options": {"temperature": 0, "seed": 20261015, "num_ctx": 4096, "num_predict": 512}})
            self.assertNotIn("Authorization", request["headers"])
        self.assertEqual(read_jsonl(self.output), [{"id": self.prompt["id"], "response": answer}])
        trace = read_jsonl(self.trace)[0]
        self.assertTrue(trace["extraction_valid"])
        self.assertEqual(len(trace["calls"]), 2)
        self.assertTrue(trace["calls"][0]["thinking_returned"])
        self.assertEqual(trace["calls"][0]["eval_count"], 12)
        self.assertEqual(trace["environment"]["digest"], "local-digest")
        self.assertEqual(trace["environment"]["ollama_version"], "test-version")
        self.assertNotIn(hidden, self.trace.read_text(encoding="utf-8"))

    def test_invalid_extraction_is_scored_failure_without_second_call(self):
        responses = [{"models": [{"name": "test-model"}]}, {}, {"version": "test"},
                     native_completion('{"records":[{"invented":"record"}]}')]
        with patch("evidenceswitch.ollama._request", side_effect=responses) as request:
            self.assertEqual(self.run_native(strategy="two-stage", trace_path=self.trace), 1)
        self.assertEqual(request.call_count, 4)
        predictions = read_jsonl(self.output)
        self.assertEqual(predictions[0]["response"], {"error": "invalid_extraction"})
        result = next(row for row in score(self.cases, predictions)["results"] if row["id"] == self.prompt["id"])
        self.assertFalse(result["format_valid"])
        self.assertFalse(result["correct"])
        trace = read_jsonl(self.trace)[0]
        self.assertFalse(trace["extraction_valid"])
        self.assertEqual(len(trace["calls"]), 1)

    def test_missing_and_cloud_models_fail_before_chat_or_output(self):
        for responses in ([{"models": []}],
                          [{"models": [{"name": "test-model"}]}, {"remote_host": "https://cloud.example"}],
                          [{"models": [{"name": "test-model", "remote_model": "cloud-model"}]}, {}]):
            with self.subTest(responses=responses), patch("evidenceswitch.ollama._request", side_effect=responses) as request:
                with self.assertRaises(ValueError):
                    self.run_native(trace_path=self.trace)
                self.assertTrue(all(not call.args[1].endswith("/api/chat") for call in request.call_args_list))
                self.assertFalse(self.output.exists())
                self.assertFalse(self.trace.exists())

    def test_nonloopback_and_ambiguous_urls_never_make_requests(self):
        urls = ["https://example.com", "http://192.168.1.1", "http://127.0.0.1.evil.test",
                "http://user:password@127.0.0.1", "http://localhost/api", "http://localhost?key=secret",
                "http://localhost:bad", "http://localhost:0", "http://localhost\n", None]
        with patch("evidenceswitch.ollama._request") as request:
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    self.run_native(base_url=url)
            request.assert_not_called()

    def test_entire_template_file_validated_before_preflight_despite_limit(self):
        invalid = copy.deepcopy(self.prompt)
        invalid["id"] = "another-id"
        invalid["messages"][0]["content"] += " unsupported instruction"
        self.write_prompts([self.prompt, invalid])
        with patch("evidenceswitch.ollama._request") as request:
            with self.assertRaises(ValueError):
                self.run_native(strategy="two-stage", limit=1)
            request.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_trace_paths_and_numeric_options_fail_before_preflight(self):
        with patch("evidenceswitch.ollama._request") as request:
            for kwargs in ({"trace_path": self.output}, {"trace_path": self.prompts},
                           {"seed": -1}, {"seed": True}, {"num_ctx": 0}, {"num_predict": 1.5}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.run_native(**kwargs)
            self.trace.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.run_native(trace_path=self.trace)
            request.assert_not_called()
        self.assertEqual(self.trace.read_text(encoding="utf-8"), "preserve")
        self.assertFalse(self.output.exists())

    def test_malformed_http_response_is_sanitized(self):
        secret = "private-invalid-json-response"
        with native_endpoint([secret]) as (url, requests):
            with self.assertRaises(RuntimeError) as caught:
                self.run_native(base_url=url)
        self.assertEqual(len(requests), 4)
        self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(self.output.read_text(encoding="utf-8"), "")

    def test_completed_predictions_and_traces_survive_later_incomplete_response(self):
        self.write_prompts([self.prompt, dict(self.prompt, id="another-id")])
        answer = json.dumps(self.cases[0]["expected"])
        responses = [{"models": [{"name": "test-model"}]}, {}, {"version": "test"},
                     native_completion(answer), native_completion("unfinished", done=False)]
        with patch("evidenceswitch.ollama._request", side_effect=responses) as request:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                self.run_native(trace_path=self.trace)
        self.assertEqual(request.call_count, 5)
        self.assertEqual(read_jsonl(self.output), [{"id": self.prompt["id"], "response": answer}])
        self.assertEqual([row["id"] for row in read_jsonl(self.trace)], [self.prompt["id"]])

    def test_openai_compatible_two_stage_uses_shared_workflow(self):
        extraction = json.dumps({"records": []})
        answer = '{"answer":null,"citations":[]}'
        with endpoint([completion(extraction), completion(answer)]) as (url, requests):
            self.assertEqual(run_prompts(self.prompts, self.output, base_url=url, model="test-model",
                                         strategy="two-stage", trace_path=self.trace), 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual([request["body"]["model"] for request in requests], ["test-model"] * 2)
        self.assertEqual(requests[0]["body"]["messages"], prepare(self.prompt["messages"]).extraction_messages)
        self.assertEqual(requests[1]["body"]["messages"], prepare(self.prompt["messages"]).finish(extraction))
        self.assertEqual(read_jsonl(self.output)[0]["response"], answer)
        self.assertTrue(read_jsonl(self.trace)[0]["extraction_valid"])


if __name__ == "__main__":
    unittest.main()
