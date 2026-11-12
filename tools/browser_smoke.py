"""Реальный браузер + веб + общий движок + HTTP-заглушка модели.

Только проверка: вымышленные документы создаются во временной папке. Рабочий
веб не имеет demo-режима и не читает этот файл или заданные здесь ответы.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
import zipfile

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "browser-check")
    parser.add_argument("--channel", default="chrome", help="chrome либо chromium (после playwright install chromium)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    intents = []
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            instruction = body["messages"][0]["content"]
            data = json.loads(body["messages"][1]["content"])
            if "Разбери последнюю" in instruction:
                intents.append(data)
                question = data["history"][-1]["content"]
                if "долгий" in question.casefold():
                    time.sleep(1.5)
                if "ошибка" in question.casefold():
                    self.send_response(401); self.end_headers(); return
                query = "согласование типовой формы" if "соглас" in question.casefold() else ("гарантийный срок" if "гарант" in question.casefold() else "срок оплаты")
                reply = {"action": "search", "query": query + " Т-1/2026", "parts": [question], "filters": {"contract_number": "Т-1/2026"}}
            elif "Независимо проверь" in instruction:
                reply = {"checks": [{"index": i, "status": "conditional" if c.get("conditions") else "confirmed"} for i,c in enumerate(data["claims"])]}
            else:
                question = data["history"][-1]["content"]
                if "соглас" in question.casefold():
                    e = next((e for e in data["evidence"] if e["text"].startswith("4.1. Типовая")), None)
                    if e is None:
                        reply={"status":"insufficient","claims":[],"missing_info":["Положение типовой формы"]}
                        raw=json.dumps({"choices":[{"message":{"content":json.dumps(reply)},"finish_reason":"stop"}]}).encode()
                        self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
                        return
                    ref = lambda e: {"document_id": e["document_id"], "chunk_id": e["chunk_id"], "quote": e["text"]}
                    limitations=[ref(x) for x in data["evidence"] if x["text"].startswith(("2.1. Все договоры", "1. Для договора"))]
                    known=data.get("scenario",{}).get("facts",{}).get("standard_form_unchanged") is True
                    reply={"status":"answer" if known else "partial","claims":[{"text":"Если используется типовая форма без изменений, договор согласуется только с главным бухгалтером.","rule":"duty","supports":[ref(e)],"limitations":limitations,"conditions":[{"text":"Типовая форма без изменений","key":"standard_form_unchanged","expected":True}],"addresses":[0]}]}
                else:
                    prefix = "5.2." if "гарант" in question.casefold() else "4.2."
                    e = next(e for e in data["evidence"] if e["text"].startswith(prefix))
                    reply = {"status": "answer", "claims": [{"text": re.sub(r"^\d+(?:\.\d+)*\.\s*", "", e["text"]), "supports": [{"document_id": e["document_id"], "chunk_id": e["chunk_id"], "quote": e["text"]}], "addresses": [0]}]}
            raw = json.dumps({"choices": [{"message": {"content": json.dumps(reply, ensure_ascii=False)}, "finish_reason": "stop"}]}).encode()
            try:
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass  # отменённое браузерное задание
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    errors = []
    with tempfile.TemporaryDirectory(prefix="psb-browser-") as directory:
        data_dir = Path(directory)
        with zipfile.ZipFile(data_dir / "knowledge-base.zip", "w") as archive:
            archive.writestr("документы компании__Тестовый договор Т-1-2026.txt", "ДОГОВОР ПОСТАВКИ № Т-1/2026\nг. Москва — 1 января 2026 г.\nООО «Тестовый поставщик»\n🙂 Проверка координат Unicode.\n4.2. Покупатель оплачивает товар в течение 30 календарных дней после подписания УПД.\n5.2. Гарантийный срок составляет 24 месяца с даты поставки.\n<script>window.compromised=true</script>\n")
            archive.writestr("документы компании__Регламент типовых форм.txt", "РЕГЛАМЕНТ ТИПОВЫХ ФОРМ\n2.1. Все договоры согласуются с юридическим отделом.\n4.1. Типовая форма без изменений согласуется только с главным бухгалтером.\n")
            archive.writestr("документы компании__Доп соглашение 1 к Т-1-2026.txt", "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1\nк договору № Т-1/2026\n1. Для договора применяется правило согласования типовой формы.\n")
        env = {k: v for k, v in os.environ.items() if k not in {"SEARCH_FACTORY", "RERANKER_MODEL", "EMB_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "GEN_MODEL"}}
        env.update(DATA_DIR=str(data_dir), WEB_STATE_DIR=str(data_dir / "state"), LLM_BASE_URL=f"http://127.0.0.1:{proxy.server_port}/v1", LLM_API_KEY="test-only-key", LLM_MODEL="browser-test-model", API_RETRIES="0", VERIFY_ANSWER="1")
        port = free_port()
        with tempfile.TemporaryFile() as logs:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "web_app:app", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"], cwd=ROOT, env=env, stdout=logs, stderr=logs)
            try:
                url = f"http://127.0.0.1:{port}"
                for _ in range(100):
                    if process.poll() is not None: raise RuntimeError("Веб-процесс завершился до проверки")
                    try:
                        with urllib.request.urlopen(url + "/readyz", timeout=1) as response:
                            if response.status == 200: break
                    except OSError: time.sleep(.1)
                else: raise RuntimeError("Веб не запустился")
                with sync_playwright() as p:
                    options = {"headless": True}
                    if args.channel != "chromium": options["channel"] = args.channel
                    browser = p.chromium.launch(**options)
                    context = browser.new_context(viewport={"width": 1440, "height": 980}, locale="ru-RU")
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url, wait_until="networkidle")
                    expect(page.locator("#service-status-text")).to_have_text("Готов к работе")
                    page.screenshot(path=str(args.output / "desktop-start.png"))
                    page.locator("#question").fill("Какой срок оплаты по договору Т-1/2026?")
                    page.get_by_role("button", name="Отправить вопрос").click()
                    expect(page.locator(".message-assistant")).to_have_count(1)
                    expect(page.locator(".message-status")).to_have_text("Ответ с основаниями")
                    expect(page.locator(".message-assistant .message-text")).to_contain_text("30 календарных")
                    page.locator(".citation-card").click()
                    expect(page.locator("#document-text mark")).to_contain_text("30 календарных")
                    assert page.evaluate("window.compromised === undefined")
                    page.screenshot(path=str(args.output / "desktop-citation.png"))
                    page.locator("#close-panel").click()
                    page.locator("#question").fill("А какой гарантийный срок?")
                    page.get_by_role("button", name="Отправить вопрос").click()
                    expect(page.locator(".message-assistant")).to_have_count(2)
                    expect(page.locator(".message-assistant .message-text").last).to_contain_text("24 месяца")
                    assert len(intents[-1]["history"]) == 3
                    page.reload(wait_until="networkidle")
                    expect(page.locator(".message-assistant")).to_have_count(2)
                    page.get_by_role("button", name="Новый диалог").click()
                    expect(page.locator(".message-user")).to_have_count(0)
                    page.locator("#question").fill("Ошибка API")
                    page.get_by_role("button", name="Отправить вопрос").click()
                    expect(page.locator(".message-status")).to_have_text("Техническая ошибка")
                    expect(page.get_by_role("button", name="Повторить запрос")).to_be_visible()
                    assert len(intents[-1]["history"]) == 1
                    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                    page.get_by_role("button", name="Новый диалог").click()
                    page.locator("#question").fill("Кто согласует типовой договор?")
                    page.get_by_role("button", name="Отправить вопрос").click()
                    expect(page.locator(".message-status")).to_have_text("Подтверждена часть")
                    page.locator(".statement-card summary").click()
                    expect(page.locator(".statement-content")).to_contain_text("Нужно установить")
                    page.get_by_role("button", name="Сравнить положения рядом").click()
                    expect(page.locator("dialog .comparison-grid section")).to_have_count(2)
                    page.screenshot(path=str(args.output / "desktop-proof-comparison.png"))
                    page.get_by_role("button", name="Закрыть", exact=True).click()
                    page.locator("#scenario-settings summary").click()
                    page.locator("#fact-standard_form_unchanged").select_option("true")
                    page.locator("#recheck-scenario").click()
                    expect(page.locator(".message-status")).to_have_text("Ответ с основаниями")
                    expect(page.locator(".source-diff")).to_be_visible()
                    page.get_by_role("button",name="Сохранить запуск для сравнения").click()
                    expect(page.locator("#toast")).to_contain_text("Запуск сохранён")
                    runs=page.request.get(url+"/api/runs").json()["runs"]
                    saved=page.request.get(url+"/api/runs/"+runs[0]["id"]).json()
                    record=saved["records"][0]
                    report=data_dir/"comparison.jsonl"
                    report.write_text(json.dumps({"id":record["id"],"question":"<img src=x onerror=evil()> отчёт", "answer":record["answer"],"configuration":{"model":"alternate-test-model"},"stages":[{"stage":"search","evidence":[{"chunk_id":"other","document_id":"other","document_title":"Другой источник","section":"1","text":"Фрагмент другого запуска"}]}],"model_outputs":[{"valid":False}]},ensure_ascii=False)+"\n")
                    page.locator("#diagnostics-button").click()
                    page.locator('dialog input[type="file"]').set_input_files(str(report))
                    expect(page.locator("dialog")).to_contain_text("Отчёты импортированы")
                    all_runs=page.request.get(url+"/api/runs").json()["runs"]
                    page.get_by_label("Первый запуск",exact=True).select_option(runs[0]["id"])
                    page.get_by_label("Второй запуск",exact=True).select_option(all_runs[0]["id"])
                    page.get_by_role("button",name="Показать различия").click()
                    page.locator(".run-case summary").click()
                    expect(page.locator(".run-case")).to_contain_text("Другой источник")
                    assert page.locator(".run-case img").count()==0
                    page.get_by_label("Ручная оценка корректности "+record["id"]).select_option("partial")
                    page.get_by_role("button",name="Сохранить оценку").click()
                    expect(page.locator("dialog")).to_contain_text("Оценка сохранена")
                    page.screenshot(path=str(args.output/"desktop-run-comparison.png"))
                    page.get_by_role("button",name="Закрыть",exact=True).click()
                    page.get_by_role("button",name="Новый диалог").click()
                    before=len(intents)
                    page.locator("#question").fill("Долгий запрос: срок оплаты?")
                    page.get_by_role("button",name="Отправить вопрос").click()
                    expect(page.locator(".loading-message")).to_be_visible()
                    page.reload(wait_until="networkidle")
                    expect(page.locator(".message-status")).to_have_text("Ответ с основаниями")
                    assert len(intents)==before+1, "Reload repeated model execution"
                    page.locator("#question").fill("Долгий запрос отмена: срок оплаты?")
                    page.get_by_role("button",name="Отправить вопрос").click()
                    page.get_by_role("button",name="Отменить",exact=True).click()
                    expect(page.locator(".message-status").last).to_have_text("Отменено")
                    context.close()
                    mobile = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, device_scale_factor=1, locale="ru-RU")
                    page = mobile.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url, wait_until="networkidle")
                    page.screenshot(path=str(args.output / "mobile-start.png"))
                    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                    page.locator("#question").fill('<img src=x onerror="window.compromised=true"> Какой срок оплаты?')
                    page.get_by_role("button", name="Отправить вопрос").click()
                    expect(page.locator(".message-assistant")).to_have_count(1)
                    assert page.locator(".message-user img").count() == 0
                    page.locator(".citation-card").click()
                    expect(page.locator("#document-text mark")).to_contain_text("30 календарных")
                    assert page.locator("#document-panel").get_attribute("aria-modal") == "true"
                    assert page.locator("#main-area").evaluate("e => e.inert")
                    assert page.evaluate("window.compromised === undefined")
                    page.screenshot(path=str(args.output / "mobile-citation.png"))
                    page.locator("#close-panel").click()
                    page.locator("#menu-button").click()
                    expect(page.locator("#sidebar")).to_have_class("sidebar open")
                    page.get_by_role("button", name="Новый диалог").click()
                    expect(page.locator("#welcome")).to_be_visible()
                    assert not page.locator("#main-area").evaluate("e => e.inert")
                    page.locator("#question").fill("Кто согласует типовой договор?")
                    page.get_by_role("button",name="Отправить вопрос").click()
                    expect(page.locator(".message-status")).to_have_text("Подтверждена часть")
                    page.locator(".statement-card summary").click()
                    page.get_by_role("button",name="Сравнить положения рядом").click()
                    expect(page.locator("dialog .comparison-grid section")).to_have_count(2)
                    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                    assert page.locator("dialog").evaluate("e=>e.scrollWidth<=e.clientWidth+1")
                    page.screenshot(path=str(args.output/"mobile-proof-comparison.png"))
                    page.get_by_role("button",name="Закрыть",exact=True).click()
                    mobile.close()
                    browser.close()
                assert not errors, errors
                report = {"result": "BROWSER_SMOKE_OK", "model": "local HTTP stub; no real model calls", "desktop": "1440x980", "mobile": "390x844", "checks": ["chat", "citation and Unicode highlight", "dialog continuation", "tab history", "new dialog isolation", "API error and retry control", "document/user HTML stays text", "statement conditions", "side-by-side proof comparison", "scenario recheck", "run import and comparison", "manual rating", "reload resumes same job", "cancel job", "mobile menu and viewer", "no horizontal overflow", "no JavaScript page errors"], "intent_requests": len(intents)}
                (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps(report, ensure_ascii=False))
            finally:
                process.terminate()
                try: process.wait(timeout=8)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                proxy.shutdown(); proxy.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    main()
