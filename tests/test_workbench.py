import asyncio

import httpx
import pytest

from schemas import Answer
from web_app import create_app


def payload(key="request-0001", question="Вопрос"):
    return {"idempotency_key": key, "messages": [{"role": "user", "content": question}]}


async def wait_terminal(client, job_id):
    for _ in range(100):
        row = (await client.get("/api/jobs/"+job_id)).json()
        if row["state"] in {"ready", "error", "cancelled"}:
            return row
        await asyncio.sleep(.01)
    raise AssertionError("Задание не завершилось")


class ControlledEngine:
    def __init__(self):
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def answer(self, messages, as_of=None, scenario=None, on_progress=None):
        self.calls += 1
        self.started.set()
        if on_progress:
            on_progress({"stage": "search", "sources": [{"document_id": "doc", "chunk_id": "chunk", "document_title": "Источник", "section": "1"}]})
        await self.release.wait()
        return Answer(status="out_of_scope", text="Проверка завершена")


async def test_jobs_are_idempotent_owned_and_independent_of_connection():
    engine = ControlledEngine()
    app = create_app(engine=engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as a, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as b:
            bootstrap = await a.get("/")
            assert bootstrap.status_code == 200 and "httponly" in bootstrap.headers["set-cookie"].lower()
            original_cookie = a.cookies.get("psb_owner")
            first = await a.post("/api/jobs", json=payload())
            assert first.status_code == 202
            assert a.cookies.get("psb_owner") == original_cookie
            job_id = first.json()["job_id"]
            await asyncio.wait_for(engine.started.wait(), 1)
            again = await a.post("/api/jobs", json=payload())
            assert again.json()["job_id"] == job_id and engine.calls == 1
            assert (await a.post("/api/jobs", json=payload(question="Другой вопрос"))).status_code == 409
            assert (await b.get("/api/jobs/"+job_id)).status_code == 404
            assert (await b.post("/api/jobs/"+job_id+"/cancel")).status_code == 404
            active = (await a.get("/api/jobs/"+job_id)).json()
            assert active["answer"] is None and active["events"][0]["sources"][0]["chunk_id"] == "chunk"
            cookies = httpx.Cookies(a.cookies)
            engine.release.set()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test", cookies=cookies) as reconnected:
            final = await wait_terminal(reconnected, job_id)
            assert final["state"] == "ready" and final["answer"]["text"] == "Проверка завершена"
            assert engine.calls == 1


async def test_cancel_does_not_publish_answer_or_destroy_worker():
    engine = ControlledEngine()
    app = create_app(engine=engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            job = (await client.post("/api/jobs", json=payload())).json()["job_id"]
            await asyncio.wait_for(engine.started.wait(), 1)
            assert (await client.post(f"/api/jobs/{job}/cancel")).json()["state"] == "cancelled"
            engine.release.set()
            second = (await client.post("/api/jobs", json=payload("request-0002"))).json()["job_id"]
            assert (await wait_terminal(client, second))["state"] == "ready"
            cancelled = await wait_terminal(client, job)
            assert cancelled["answer"] is None and cancelled["state"] == "cancelled"


async def test_completed_result_survives_server_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    engine = ControlledEngine()
    engine.release.set()
    app = create_app(engine=engine, state_path=path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            job = (await client.post("/api/jobs", json=payload())).json()["job_id"]
            await wait_terminal(client, job)
            cookies = httpx.Cookies(client.cookies)
    next_engine = ControlledEngine()
    new_app = create_app(engine=next_engine, state_path=path)
    async with new_app.router.lifespan_context(new_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(new_app), base_url="http://test", cookies=cookies) as client:
            assert (await client.get("/api/jobs/"+job)).json()["state"] == "ready"
            duplicate = await client.post("/api/jobs", json=payload())
            assert duplicate.json()["job_id"] == job
            assert next_engine.calls == 0


async def test_run_import_rating_and_scores_are_separate_and_owned():
    engine = ControlledEngine()
    app = create_app(engine=engine)
    record = {"id": "001", "question": "<script>evil()</script>", "answer": Answer(status="insufficient", text="Нет достаточных оснований", missing_info=["документ"]).model_dump(), "model_outputs": [{"valid": False}], "stages": [{"stage": "rejected_claim", "reason": "Неверный источник"}], "configuration": {"model": "test-model", "api_key": "DO_NOT_IMPORT"}}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as a, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as b:
            response = await a.post("/api/runs/import", json={"name": "Локальный прогон", "records": [record]})
            assert response.status_code == 200
            run = response.json()["run_id"]
            data = (await a.get("/api/runs/"+run)).json()
            assert data["records"][0]["format_errors"] == 1
            assert "api_key" not in data["configuration"] and "DO_NOT_IMPORT" not in str(data)
            assert (await b.get("/api/runs/"+run)).status_code == 404
            rating = {"correctness": "partial", "unnecessary_refusal": True, "note": "Проверено вручную"}
            assert (await a.patch(f"/api/runs/{run}/cases/001/rating", json=rating)).status_code == 200
            assert (await a.get("/api/runs/"+run)).json()["records"][0]["rating"] == rating
            assert (await b.patch(f"/api/runs/{run}/cases/001/rating", json=rating)).status_code == 404
            await a.post("/api/official-scores", json={"archive": "candidate.zip", "score": .1234, "duration_seconds": 99})
            assert len((await a.get("/api/official-scores")).json()["scores"]) == 1
            assert (await b.get("/api/official-scores")).json()["scores"] == []
            assert "score" not in (await a.get("/api/runs/"+run)).json()
            assert engine.calls == 0


async def test_job_capacity_and_cross_origin_protection():
    engine = ControlledEngine()
    app = create_app(engine=engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            assert (await client.post("/api/jobs", json=payload(), headers={"Origin": "https://elsewhere.invalid"})).status_code == 403
            for i in range(4):
                assert (await client.post("/api/jobs", json=payload("request-"+str(i)))).status_code == 202
            assert (await client.post("/api/jobs", json=payload("request-full"))).status_code == 429
            bad = {**payload("request-invalid"), "messages": [{"role": "assistant", "content": "PRIVATE"}]}
            response = await client.post("/api/jobs", json=bad)
            assert response.status_code == 422 and "PRIVATE" not in response.text
            engine.release.set()
