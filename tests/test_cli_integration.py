"""Полный запуск solution.py через настоящий HTTP к локальной заглушке API."""
import csv
import json
import os
import subprocess
import sys
import threading
import zipfile
import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from solution import validate_submission
from tools.package_submission import package


@pytest.mark.parametrize("mode", [None, True, False])
def test_cli_with_real_http_csv_and_archive(tmp_path, mode):
    transferred = tmp_path / "submission_root"
    transferred.mkdir()
    output = package(tmp_path / "submission.zip", with_report=mode is not None, verify_answer=mode)
    with zipfile.ZipFile(output) as archive:
        archive.extractall(transferred)
    data = tmp_path / "input"
    data.mkdir()
    (data / "questions.csv").write_text('id,question\n001,"Срок оплаты по договору Т-1/2026?"\n', encoding="utf-8")
    with zipfile.ZipFile(data / "knowledge-base.zip", "w") as archive:
        archive.writestr("contract.txt", 'ДОГОВОР ПОСТАВКИ № Т-1/2026\nООО «Тестовый поставщик»\n4.2. Оплата в течение 30 календарных дней после подписания УПД.\n')
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = request["messages"][0]["content"]
            data = json.loads(request["messages"][1]["content"])
            requests.append((self.path, request))
            if "Разбери последнюю" in system:
                reply = {"action": "search", "query": "Срок оплаты Т-1/2026", "parts": ["Срок оплаты"], "filters": {"contract_number": "Т-1/2026"}}
            elif "Независимо проверь" in system:
                reply = {"supported": [0]}
            else:
                evidence = data["evidence"][0]
                quote = next(line for line in evidence["text"].split("\n") if line.startswith("4.2."))
                reply = {"status": "answer", "claims": [{"text": "Оплата в течение 30 календарных дней после подписания УПД.", "supports": [{"document_id": evidence["document_id"], "chunk_id": evidence["chunk_id"], "quote": quote}], "addresses": [0]}]}
            body = json.dumps({"choices": [{"message": {"content": json.dumps(reply, ensure_ascii=False)}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {k: v for k, v in os.environ.items() if k not in {"SEARCH_FACTORY", "RERANKER_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "GEN_MODEL", "EMB_MODEL", "OTOROP_EMBEDDING_CACHE_DIR", "OTOROP_KNOWLEDGE_CACHE_DIR", "KNOWLEDGE_LLM_EXTRACTION"}}
    blocked_cache = tmp_path / "readonly-home-cache"
    blocked_cache.mkdir(mode=0o555)
    env["XDG_CACHE_HOME"] = str(blocked_cache)
    env.update(DATA_DIR=str(data), LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1", LLM_API_KEY="test-only-key", LLM_MODEL="test-model", MAX_WORKERS="1", VERIFY_ANSWER="1")
    try:
        result = subprocess.run([sys.executable, str(transferred / "solution.py")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        blocked_cache.chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert not list(blocked_cache.iterdir())
    assert "answer=1" in result.stdout
    validate_submission(tmp_path / "submission.csv", ["001"])
    with (tmp_path / "submission.csv").open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert "30 календарных" in row["answer"] and "п. 4.2" in row["answer"]
    expected = 2 if mode is False else 3
    assert len(requests) == expected
    assert all(path == "/v1/chat/completions" for path, _ in requests)
    assert "test-only-key" not in result.stdout + result.stderr
    if mode is not None:
        report = (tmp_path / "run-report.jsonl").read_text(encoding="utf-8")
        record = json.loads(report)
        assert record["id"] == "001" and record["verify"] is mode
        assert record["calls"] == {"/chat/completions": expected}
        assert record["answer"]["status"] == "answer" and record["answer"]["citations"]
        assert "test-only-key" not in report and "127.0.0.1:" not in report
