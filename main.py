"""
CloudNerd Agentic RAG backend.

Runs alongside customnerd-backend (port 8000) on port 8001 by default.
Reuses retrieval / relevance / synthesis from the linear pipeline via bridge/.
"""

from __future__ import annotations

import logging
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from config import DEFAULT_PORT, load_environment, AGENT_RUN_RETRIES
from bridge.legacy import ensure_legacy_backend
from agent.orchestrator import build_result_object, run_agent
from agent.state import AgentState
from agent import llm_meter
from sse_utils import event_generator, send_update_sync, update_queues

load_environment()
ensure_legacy_backend()
llm_meter.install()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cloudnerd-agent")

app = FastAPI(
    title="CloudNerd Agentic RAG",
    description="Agent-driven RAG backend reusing CustomNerd retrieval modules.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)


@app.get("/health")
async def health():
    warnings = []
    import os

    from bridge.domain import get_domain, is_dietnerd, list_domains

    domain = get_domain()
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY not set")
    if is_dietnerd():
        if not os.getenv("ENTREZ_EMAIL"):
            warnings.append("ENTREZ_EMAIL not set (PubMed Entrez may fail)")
        if not os.getenv("NCBI_API_KEY"):
            warnings.append("NCBI_API_KEY not set (PubMed rate limits may be tight)")
    else:
        if not os.getenv("STACK_API_KEY"):
            warnings.append("STACK_API_KEY not set (Stack Overflow search may fail)")

    status = "degraded" if warnings else "ok"
    return {
        "status": status,
        "backend": "cloudnerd-agent-backend",
        "domain": domain,
        "available_domains": list_domains(),
        "retrieval_mode": os.getenv("RETRIEVAL_MODE"),
        "legacy_root": str(ensure_legacy_backend()),
        "warnings": warnings,
    }


@app.post("/agent/load_domain")
async def agent_load_domain(domain: str = Form(...)):
    """Hot-switch domain pack (DietNerd | CloudNerd | ...) without restarting the process.

    Copies saved_states/<domain>/ into customnerd-backend and reloads prompts/search.
    """
    from bridge.domain import activate_domain, list_domains

    if domain not in list_domains():
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found. Available: {list_domains()}",
        )
    try:
        status = activate_domain(domain, force=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", **status}


@app.get("/sse")
async def sse(session_id: str = Query(default=None)):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return EventSourceResponse(event_generator(session_id))


def _run_agent_logic(
    user_query: str,
    session_id: str,
    max_date: Optional[str],
    query_title: Optional[str],
) -> None:
    def progress(msg: str) -> None:
        send_update_sync(session_id, msg)

    last_exc: Optional[Exception] = None
    for attempt in range(1, AGENT_RUN_RETRIES + 1):
        try:
            if attempt > 1:
                progress(f"Agent retry {attempt}/{AGENT_RUN_RETRIES}...")
            send_update_sync(session_id, "CloudNerd agent starting...")
            llm_meter.reset()
            state = AgentState(
                session_id=session_id,
                user_query=user_query,
                query_title=query_title,
                max_date=max_date,
            )
            run_agent(state, on_progress=progress)
            result = build_result_object(state)
            send_update_sync(session_id, result)
            return
        except Exception as exc:
            last_exc = exc
            logger.exception("Agent run failed (attempt %s/%s)", attempt, AGENT_RUN_RETRIES)
            if attempt >= AGENT_RUN_RETRIES:
                break

    send_update_sync(
        session_id,
        {
            "end_output": f"Agent error: {last_exc}",
            "final_output": f"Agent error: {last_exc}",
            "citations_obj": [],
            "agent_backend": "cloudnerd-agent-backend",
        },
    )


@app.post("/process_detailed_combined_query")
async def process_detailed_combined_query(
    background_tasks: BackgroundTasks,
    user_query: str = Form(...),
    search_pubmed: bool = Form(True),
    search_pmid: bool = Form(False),
    pmids: Optional[str] = Form(None),
    search_pdf: bool = Form(False),
    max_date: Optional[str] = Form(None),
    query_title: Optional[str] = Form(None),
    files: List[UploadFile] = File(None),
):
    """Compatible entry point with the linear backend (agent ignores PDF/PMID for now)."""
    del background_tasks, search_pubmed, search_pmid, pmids, search_pdf, files

    session_id = str(uuid.uuid4())
    update_queues[session_id] = queue.Queue()
    logger.info("Session ID: %s", session_id)

    executor.submit(
        _run_agent_logic,
        user_query,
        session_id,
        max_date,
        query_title,
    )
    return {"session_id": session_id}


@app.post("/agent/query")
async def agent_query(
    background_tasks: BackgroundTasks,
    user_query: str = Form(...),
    max_date: Optional[str] = Form(None),
    query_title: Optional[str] = Form(None),
):
    """Explicit agent endpoint (same behavior as process_detailed_combined_query)."""
    return await process_detailed_combined_query(
        background_tasks=background_tasks,
        user_query=user_query,
        search_pubmed=True,
        search_pmid=False,
        search_pdf=False,
        max_date=max_date,
        query_title=query_title,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)
