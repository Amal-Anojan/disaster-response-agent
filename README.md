
# Disaster Response Multi-Modal AI Agent

[![Project Status](https://img.shields.io/badge/status-beta-orange)](#) [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)

> **Turn incoming disaster reports and images into fast, effective emergency plans.**
> A real-time, multi-modal decision-support system combining computer vision, language models, and an intuitive dashboard to help emergency responders act faster and save lives.

---

## TL;DR — Why this matters

When disaster strikes—like floods or fires—every second counts. Our system uses AI to automate key tasks:

- **Fast visual damage analysis** powered by Cerebras Cloud API,
- **Smart step-by-step emergency action planning** using Google Gemini language models,
- **User-friendly dashboard** for reporting and monitoring incidents,
- **A robust backend gateway** that orchestrates all AI services seamlessly in the background.

This modular system is built to be ready for real-world deployment, helping emergency teams scale rapidly during crises.

---

# Table of contents

1. [Demo & Screenshots](#demo--screenshots)  
2. [Key features](#key-features)  
3. [Architecture overview](#architecture-overview)  
4. [Quick start (run locally)](#quick-start-run-locally)  
5. [API usage examples](#api-usage-examples)  
6. [Troubleshooting & common errors](#troubleshooting--common-errors)  
8. [For learners: how it works (high level)](#for-learners-how-it-works-high-level)  
9. [Contributing & roadmap](#contributing--roadmap)  
10. [License & acknowledgements](#license--acknowledgements)  

---

# Demo & Screenshots

> After starting the project, open the dashboard in your browser:

```

[http://localhost:8501](http://localhost:8501)

```

(Include these files in your repo for judges:

- `docs/demo-video.mp4` — short 2–3 minute project walkthrough  
- `docs/screenshot-dashboard.png` — annotated screenshot of the dashboard  
- `docs/architecture-diagram.png` — system architecture visualization)

---

# Key features

- **Multi-modal input:** Accepts textual reports, GPS coordinates, and images.  
- **Rapid vision analysis:** Uses Cerebras Cloud AI to classify disaster damage and score severity instantly.  
- **Automated action planning:** Employs Google Gemini LLM to generate prioritized, actionable emergency response plans.  
- **Resilient MCP gateway:** Coordinates multi-AI workflows reliably with FastAPI and MCP Gateway patterns.  
- **Live interactive dashboard:** Enables users to submit reports and track active incidents on a realtime map.  
- **Fallback & resilience:** Supports graceful fallback when knowledge base or external AI services are unavailable.  
- **Extensible knowledge base:** Integrates ChromaDB and SentenceTransformers for advanced retrieval of emergency protocols (optional).

---

# Architecture overview

1. **Streamlit UI:** Simple web interface for field agents and responders to report incidents and monitor status.  
2. **FastAPI MCP Gateway:** Central orchestrator for ingestion, processing, and coordination of AI services.  
3. **Vision Service (Cerebras):** High-speed damage analysis extracting disaster types, severity, and confidence metrics.  
4. **LLM Service (Google Gemini):** Generates intelligent and context-aware emergency action plans.  
5. **Optional Knowledge Base (ChromaDB + SentenceTransformers):** Retrieval-augmented generation for emergency procedures.  
6. **Data Storage:** Uses SQLite or any preferred database for storing incident data safely and efficiently.

*(Refer to `docs/architecture-diagram.png` for a visualization of the system components and flow.)*

---

# Quick start (run locally)

Tested on Windows/macOS/Linux.

```

# 1. Clone repository

git clone <your-repository-link>
cd disaster-response-agent

# 2. Create Python environment

conda create -n disaster-env python=3.9 -y
conda activate disaster-env

# 3. Install all dependencies

pip install -r requirements.txt

# 4. Set required environment variables (example)

# Cerebras API key is required to analyze images

# Replace your_key_here with your actual key

# PowerShell (Windows)

setx CEREBRAS_API_KEY "your_key_here"

# bash (macOS/Linux)

export CEREBRAS_API_KEY="your_key_here"

# 5. Start the MCP backend gateway

python -m uvicorn src.orchestrator.mcp_server:app --host 0.0.0.0 --port 8080 --reload

# 6. In a new terminal, start the dashboard UI

streamlit run src/ui/dashboard.py --server.address localhost --server.port 8501

```

Open browser at: `http://localhost:8501`

---

# API usage examples

## JSON-only incident report (no image):

```

curl -X POST "[http://localhost:8080/api/v1/incident/report](http://localhost:8080/api/v1/incident/report)" 
-H "Content-Type: application/json" 
-d '{
"text_content": "Fire near highway exit",
"location": {"lat": 40.7128, "lng": -74.0060},
"source": "manual",
"priority_override": "HIGH"
}'

```

## Multipart form data with image:

```

curl -X POST "[http://localhost:8080/api/v1/incident/report](http://localhost:8080/api/v1/incident/report)" 
-F "text_content=Building collapse" 
-F "source=manual" 
-F "lat=40.7128" 
-F "lng=-74.0060" 
-F "priority_override=CRITICAL" 
-F "image=@/path/to/disaster_photo.jpg"

```

## Query incident status:

```

curl "[http://localhost:8080/api/v1/incident/status/INC_202510051230_abcd1234](http://localhost:8080/api/v1/incident/status/INC_202510051230_abcd1234)"

```

---

# Troubleshooting & common errors

### 422 Unprocessable Entity on `/incident/report`

This error usually means the data format or required fields sent from the UI or API client do not match the expected API model.

**Fix:**

- Verify the request body matches the schema (including `source` and proper `location` dict).
- Ensure image files are sent with `multipart/form-data` when uploading photos.
- Check that all required fields are present and correctly formatted.

---

# for-learners-how-it-works-high-level

- Users submit emergency reports (text + location + optional photo) through the dashboard.
- The MCP Gateway API receives and stores reports, handling uploads safely.
- Celery-like background tasks analyze images using Cerebras AI to estimate damage severity.
- Google Gemini LLM generates recommended emergency action plans based on analysis.
- Results update live on the dashboard and API endpoints for responder use.

---

# Contributing & roadmap

Contributions welcome! Here’s what’s next:

- ✅ Improve evaluation with labeled disaster image sets and planning metrics.
- ✅ Add authentication and role management.
- ✅ Enhance resource scheduling and prioritization.
- 🔜 Implement scalable worker pools (Kubernetes autoscaling).
- 🔜 Integrate SMS and emergency broadcast services.
- 🔜 Build offline capabilities with edge AI models.

---

# License & acknowledgements

Licensed under **MIT License** — see `LICENSE` file for details.

**Acknowledgements:**

- Cerebras Cloud API for rapid vision inference.
- Google Gemini for natural language emergency action planning.
- Streamlit for easy dashboard creation.
- FastAPI and MCP Gateway toolkit for resilient microservices orchestration.
- ChromaDB and SentenceTransformers for optional knowledge retrieval.

---

## Final note

This project is designed to be **practical, scalable, and lifesaving**. Judges can validate real impact through performance, accuracy, and user experience. Learners get a deep dive into cutting-edge AI technologies combined for critical disaster response. Try the demo, browse the code, and imagine the future of emergency management.
```
