# Disaster Response Multi-Modal AI Agent

[![Project Status](https://img.shields.io/badge/status-beta-orange)](#) [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)

> **Turn incoming reports and images into fast, actionable emergency plans.**
> A real-time, multi-modal decision-support system that combines vision, LLM planning, and an easy dashboard to help responders act faster and save lives.

---

## TL;DR — Why this matters

When disaster strikes, seconds matter. This project automates situational awareness and action planning by combining:

* **Fast visual damage analysis** (Cerebras Cloud),
* **Smart action planning** (Google Gemini),
* **A lightweight dashboard** for reporting and monitoring (Streamlit),
* **A resilient gateway** to orchestrate these AI services (FastAPI + MCP pattern).

It’s built to be modular and production-ready so emergency teams can scale during real crises.

---

# Table of contents

1. [Demo & Screenshots](#demo--screenshots)
2. [Key features](#key-features)
3. [Architecture overview](#architecture-overview)
4. [Quick start (run locally)](#quick-start-run-locally)
5. [API usage examples](#api-usage-examples)
6. [Troubleshooting & common errors](#troubleshooting--common-errors)
7. [For judges: evaluation & impact](#for-judges-evaluation--impact)
8. [For learners: how it works (high level)](#for-learners-how-it-works-high-level)
9. [Contributing & roadmap](#contributing--roadmap)
10. [License & acknowledgements](#license--acknowledgements)

---

# Demo & Screenshots

> Open the dashboard after you start the app:

```
http://localhost:8501
```

Include (or replace) these placeholders in your repo for judges:

* `docs/demo-video.mp4` — short 2–3 minute walkthrough
* `docs/screenshot-dashboard.png` — annotated dashboard screenshot
* `docs/architecture-diagram.png` — visual architecture

---

# Key features

* **Multi-modal ingestion** — accepts textual reports, GPS location, and images.
* **Vision analysis** — automatic damage classification and severity scoring using Cerebras.
* **Action planning** — LLM generates prioritized, step-by-step response plans.
* **Real-time MCP gateway** — orchestrates analysis + planning in background tasks.
* **Dashboard** — report incidents, view live map, and track active incidents.
* **Fallback & resilient design** — keyword/fallback logic when embeddings or external services are unavailable.
* **Extensible knowledge base** — add/update procedures and resources (ChromaDB + SentenceTransformers supported).

---

# Architecture overview

1. **UI (Streamlit)** — user-friendly form for reporters and a live map of incidents.
2. **MCP Gateway (FastAPI)** — orchestrates ingestion, background processing, and endpoints for status.
3. **Vision Service (Cerebras)** — fast image analysis → severity, disaster type, confidence.
4. **LLM Service (Google Gemini)** — generate prioritized action plans and resource estimates.
5. **Knowledge RAG (ChromaDB + SentenceTransformers)** — optional retrieval of local emergency procedures for context.
6. **Storage** — SQLite (default) or any SQL database for production; ChromaDB for vector store.

(Include an architecture PNG in `docs/` for judges.)

---

# Quick start (run locally)

> Tested on Windows/macOS/Linux. Use the repository root when running commands.

```bash
# 1. Clone
git clone <your-repository-link>
cd disaster-response-agent

# 2. Create conda environment
conda create -n disaster-env python=3.9 -y
conda activate disaster-env

# 3. Install Python deps
pip install -r requirements.txt

# 4. Set environment variables (example)
# - CEREBRAS_API_KEY is required for image analysis
# - OPTIONAL: GOOGLE_GEMINI_KEY (or configure your LLM access)
# Windows (PowerShell)
setx CEREBRAS_API_KEY "your_key_here"
# macOS / Linux
export CEREBRAS_API_KEY="your_key_here"

# 5. Start backend gateway
python -m uvicorn src.orchestrator.mcp_server:app --host 0.0.0.0 --port 8080 --reload

# 6. Start dashboard (in a new terminal)
streamlit run src/ui/dashboard.py --server.address localhost --server.port 8501
```

Open: `http://localhost:8501`

---

# API usage examples

## JSON-only report (no image)

```bash
curl -X POST "http://localhost:8080/api/v1/incident/report" \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "wildfire near river bank",
    "location": {"lat": 40.7128, "lng": -74.0060},
    "source": "manual",
    "priority_override": "HIGH"
  }'
```

## Multipart/form-data (image + metadata)

If your endpoint expects `Form` + `File` fields:

```bash
curl -X POST "http://localhost:8080/api/v1/incident/report" \
  -F "text_content=building on fire" \
  -F "source=manual" \
  -F "lat=40.7128" \
  -F "lng=-74.0060" \
  -F "priority_override=HIGH" \
  -F "image=@/path/to/photo.jpg"
```

## Check incident status

```bash
curl "http://localhost:8080/api/v1/incident/status/INC_202501011230_abcd1234"
```

---

# Troubleshooting & common errors

### 422 Unprocessable Entity on `/incident/report`

Cause: your frontend sent JSON while the endpoint expects `Form` (multipart) or vice versa.

* Solution A (JSON endpoint): Use a Pydantic model param (`incident: EmergencyIncidentRequest`) and send JSON.
* Solution B (Form + File): Use `Form(...)` for fields and `UploadFile = File(...)` for file; send `multipart/form-data`.

### PyTorch / transformers errors (e.g. `register_pytree_node`)

* Fix by installing compatible versions:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 transformers==4.40.2
```

or prefer conda-forge:

```bash
conda install -c conda-forge pytorch torchvision torchaudio transformers sentence-transformers
```

### ChromaDB telemetry warning / errors

* Telemetry tracebacks are harmless. To silence:

```bash
export CHROMA_TRACKING=false   # linux/mac
set CHROMA_TRACKING=false      # windows (cmd)
```

* If collection not found, ensure `data/chroma_db/` is writable and that indexing runs at startup.

### ChromaDB indexing / sentence-transformers

* If using Chroma + SentenceTransformers, ensure `sentence-transformers` is installed and `all-MiniLM-L6-v2` can be downloaded. Indexing computes embeddings and persists them into Chroma.

---

# For judges — evaluation & impact

**Evaluation criteria we focus on:**

* **Latency & throughput** — background processing with MCP Gateway, scalable with Docker.
* **Accuracy** — measured via confusion matrix on labeled images (vision) and human review of generated action plans.
* **Operational usefulness** — speed of converting a report → actionable plan (measured in seconds).
* **Resilience** — fallback mechanisms when network or external services are unavailable.

**Impact highlights:**

* Reduces time-to-action by automating triage and recommended procedures.
* Enables responders to coordinate resources and evacuation efficiently.
* Designed for real deployment: containerized, modular, and instrumented for monitoring.

---

# For learners — how it works (high level)

* **UI** sends a report (text + optional image) to the FastAPI MCP gateway.
* **MCP gateway** stores the report, saves image to a temp folder, and enqueues a background task.
* **Vision service** analyzes the image and returns disaster type, severity, and confidence.
* **RAG/LLM** uses retrieval-augmented context (ChromaDB or fallback) to craft a prioritized action plan.
* **Result** is saved and surfaced on the dashboard and accessible via API for responders.

---

# Contributing & roadmap

We welcome contributions! Suggested next improvements:

* ✅ Improve evaluation suite with labeled image set and plan quality metrics.
* ✅ Add authentication & role-based access for responders.
* ✅ Add more granular resource allocation & scheduling.
* 🔜 Deploy scalable worker pool (Kubernetes + autoscaling).
* 🔜 Integrate SMS / emergency broadcast connectors (Twilio, local alerts).
* 🔜 Add offline mode: caching & edge inference (on-device model).

How to contribute:

1. Fork the repo, create a feature branch.
2. Run tests (we recommend adding pytest coverage).
3. Open a PR with a clear description and testing steps.

---

# Security & privacy notes

* Be careful with personally identifying information (PII) in reports. In production, store sensitive info encrypted and follow local data protection laws.
* Limit access to API keys and use environment variables or secret stores (e.g., HashiCorp Vault or cloud secret manager).
* Use HTTPS, authentication, and RBAC in production deployments.

---

# License & acknowledgements

Licensed under the **MIT License** — see `LICENSE` file.

**Acknowledgements**

* Cerebras Cloud for fast vision inference.
* Google Gemini for LLM action planning.
* Streamlit for dashboard UI.
* FastAPI for lightweight, production-ready APIs.
* ChromaDB & SentenceTransformers for RAG capabilities (optional).

---

# Contact & demo

* Repo: `<your-repository-link>`
* Demo video: `docs/demo-video.mp4`
* Contact / Maintainer: `your-name <your-email@example.com>`

---

## Final note (for judges & newcomers)

This project is built to be both **practical** and **educational**: judges can evaluate real-world impact, performance, and robustness; learners can explore how vision models, LLMs, and retrieval systems combine to support life-critical decisions. Try the demo, inspect the code, and run local tests — then imagine how this could augment your next emergency response system.

---

Would you like a formatted `README.md` file saved into the repository (I can paste the exact file content you can copy/paste), or do you want a shorter 1-page executive summary targeted for judges?
