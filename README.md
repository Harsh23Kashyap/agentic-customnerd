# Agentic CustomNerd

Agentic RAG backend for **CloudNerd** (Stack Overflow) and **DietNerd** (PubMed).  
Same pipeline for both — switch with `AGENT_DOMAIN`.

This repo is the **agent layer only**. It reuses retrieval / relevance / synthesis from a sibling **CustomNerd linear backend** (`customnerd-backend/`).

## Layout (expected)

```
your-workspace/
├── agentic-customnerd/          # this repo (or rename to cloudnerd-agent-backend)
└── customnerd-backend/          # linear CustomNerd backend (clone separately)
    └── saved_states/
        ├── CloudNerd/
        └── DietNerd/
```

`config.py` resolves the legacy root as `../customnerd-backend` next to this folder.

## Pair with CustomNerd

1. Clone / place the linear backend as a **sibling** folder named `customnerd-backend`.
2. Ensure domain packs exist under `customnerd-backend/saved_states/CloudNerd` and `DietNerd` (prompts + search APIs).
3. Put API keys in `customnerd-backend/variables.env` (or copy `variables.env.template` → `variables.env` here for agent overrides).
4. Prefer reusing the linear backend venv (it already has BioPython, OpenAI, etc.).

## Domains (`AGENT_DOMAIN`)

| Domain | Retrieval | Notes |
|--------|-----------|-------|
| `CloudNerd` | SO cascade (`RETRIEVAL_MODE=cascade`) | Needs `STACK_API_KEY` |
| `DietNerd` | PubMed Entrez (`RETRIEVAL_MODE=legacy`) | Needs `ENTREZ_EMAIL` (+ `NCBI_API_KEY`) |

On boot (and via `POST /agent/load_domain`), the agent copies  
`customnerd-backend/saved_states/<DOMAIN>/` into the live backend and reloads prompts / search.

```bash
# DietNerd
AGENT_DOMAIN=DietNerd
RETRIEVAL_MODE=legacy

# CloudNerd
AGENT_DOMAIN=CloudNerd
RETRIEVAL_MODE=cascade
```

Hot-switch: `POST /agent/load_domain` with form field `domain=DietNerd`.

## Modes (`AGENT_MODE`)

| Mode | Behavior |
|------|----------|
| `hybrid` (default) | Enhanced pipeline + optional rescue pass |
| `parity` | Enhanced pipeline, no rescue |
| `adaptive` | ReAct or LangGraph (`USE_LANGGRAPH=true`) |

## Pipeline (high level)

```
search → organize → classify → assess
  → (optional re-search)
  → pre-verify faith → synthesize → claim filter → post-verify
  → (optional hybrid rescue) → answer + citations
```

## Quick start

```powershell
# From parent workspace that also contains customnerd-backend/
cd agentic-customnerd
copy variables.env.template variables.env
# edit AGENT_DOMAIN / keys as needed

..\customnerd-backend\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Health: `GET http://127.0.0.1:8001/health`

## Eval (from linear backend)

```powershell
cd ..\customnerd-backend
.\venv\Scripts\python.exe eval_rag.py --csv path\to\input.csv `
  --api-url http://127.0.0.1:8001 --output ..\result\agent_eval --agent-trace
```

## Key env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_DOMAIN` | `CloudNerd` / `DietNerd` | Domain pack to activate |
| `AGENT_MODE` | `hybrid` | `hybrid`, `parity`, or `adaptive` |
| `AGENT_FAST_ORGANIZE` | `true` | Fast article map (0 LLM per paper for Diet/SO) |
| `AGENT_PREFER_COMBINED_SYNTHESIS` | `true` | Grounded combined_lite answers |
| `AGENT_FAITH_FIRST` | `true` | Faithfulness-oriented verify / regen |
| `HYBRID_RESCUE_ENABLED` | `true` | Rescue pass in hybrid mode |
| `RETRIEVAL_MODE` | `legacy` / `cascade` | Diet vs Cloud retrieval |

Do **not** commit `variables.env` (secrets). Use `variables.env.template` only.
