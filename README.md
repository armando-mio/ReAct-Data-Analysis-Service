# ReAct Data Analysis Service

A production-ready, testable, and robust ReAct-style Data Analysis Service following **Hexagonal Architecture (Ports and Adapters)** powered by **LangGraph**, **SQLite**, **FastAPI**, and **Google Gemini API**.

---

## 1. System Architecture

The codebase enforces strict inward dependency flow:
$$\text{Adapters} \longrightarrow \text{Use Cases} \longrightarrow \text{Ports} \longrightarrow \text{Core Entities}$$

Neither the core domain logic nor the LangGraph agent state machine depends on FastAPI, SQLite, or external cloud SDKs.

```mermaid
graph TD
    subgraph Driving Adapters
        API[FastAPI Routers: /analyze, /sessions, /artifacts]
    end

    subgraph Use Cases
        AUC[AnalyzeDataUseCase]
        MUC[ManageSessionUseCase]
        RG[LangGraph ReAct StateGraph]
    end

    subgraph Ports
        LLMPort[ILLMClient]
        SandPort[ISandboxRunner]
        RepoPort[ISessionRepository]
        StorPort[IArtifactStorage]
    end

    subgraph Core Domain
        Entities[Session, Message, TraceStep, Artifact, Dataset]
        Exceptions[Domain Exceptions]
    end

    subgraph Driven Adapters
        Gemini[GeminiLLMAdapter & MockLLMAdapter]
        Sandbox[ProcessSandboxRunner + NetworkGuard]
        SQLite[SQLiteSessionRepository + SQLAlchemy ORM]
        Storage[LocalArtifactStorage]
    end

    API --> AUC
    API --> MUC
    AUC --> RG
    RG --> LLMPort
    RG --> SandPort
    AUC --> RepoPort
    AUC --> StorPort
    MUC --> RepoPort
    MUC --> StorPort

    LLMPort -.-> Entities
    SandPort -.-> Entities
    RepoPort -.-> Entities
    StorPort -.-> Entities

    Gemini -.-> LLMPort
    Sandbox -.-> SandPort
    SQLite -.-> RepoPort
    Storage -.-> StorPort
```

---

## 2. ReAct Agent Loop & Self-Healing State Machine

The analysis engine executes an iterative **Plan $\rightarrow$ Act $\rightarrow$ Observe $\rightarrow$ Recover** loop compiled with **LangGraph**:

```mermaid
stateDiagram-v2
    [*] --> Planner: User Query + Dataset Preview
    Planner --> CodeGenerator: Formulate Step-by-Step Plan
    CodeGenerator --> SandboxRunner: Execute Python Code
    SandboxRunner --> Reflection: Capture stdout, stderr, & HTML plots
    Reflection --> CodeGenerator: Needs Code Fix (Error in stderr)
    Reflection --> Finalizer: Resolved or Max Iterations (4) Reached
    Finalizer --> [*]: Natural Language Summary + Artifact Links
```

### Self-Healing Guarantee
If generated code raises an error (e.g. `KeyError`, `ValueError`, `TypeError`), the **Reflection Node** catches the failure and routes execution back to **Code Generator** with the full error feedback and previous code, enabling automated code correction without failing the request.

---

## 3. Key Design Decisions

| Decision | Rationale |
| :--- | :--- |
| **Hexagonal Architecture** | Isolates business analysis logic from framework and database details. Enables 100% deterministic unit testing via `MockLLMAdapter` without network or API costs. |
| **SQLite with SQLAlchemy 2.0** | Zero memory overhead, in-process, ACID-compliant, requires no extra container daemon, and supports multi-turn session resumption across service restarts. |
| **Subprocess Sandbox with Network Neutralization** | Untrusted code executes in an isolated `tempfile.TemporaryDirectory` with strict timeout kills (15s default). Sockets (`connect`, `create_connection`, `urllib`, `http.client`) are patched at startup to raise `PermissionError`, preventing data exfiltration. |
| **LangGraph StateGraph** | Explicit state persistence with declarative conditional edge routing for the self-healing recovery loop and bounded iteration limits. |
| **Plotly HTML Visualizations** | Visualizations are generated as self-contained standalone HTML documents (`include_plotlyjs="cdn"`) streamed directly via `GET /artifacts/{id}`. |

---

## 4. API Specification

### Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `POST` | `/analyze` | Run ReAct data analysis over an uploaded CSV or existing session dataset. |
| `GET` | `/sessions/{id}` | Retrieve session history, full reasoning traces, and artifact links. |
| `GET` | `/sessions` | List recent conversation sessions. |
| `GET` | `/artifacts/{id}` | Stream the standalone Plotly HTML visualization (`media_type="text/html"`). |
| `GET` | `/health` | Service health status check. |

### Sample `POST /analyze` Request (cURL)

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "question=What is the total revenue by product category? Generate an interactive chart." \
  -F "file=@data/sample_sales.csv"
```

### Sample Response Payload

```json
{
  "session_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "answer": "The highest revenue category is Electronics with $3,700 across 32 units sold. An interactive bar chart has been generated.",
  "artifacts": [
    {
      "id": "e4eaaaf2-d142-11e1-b3e4-080027620cdd",
      "file_name": "output_plot.html",
      "url": "/artifacts/e4eaaaf2-d142-11e1-b3e4-080027620cdd"
    }
  ],
  "trace": [
    {
      "step_index": 1,
      "thought": "1. Inspect data. 2. Group by category and sum revenue. 3. Output Plotly bar chart.",
      "code": "import pandas as pd\nimport plotly.express as px\n...",
      "stdout": "Category Revenue:\nElectronics: 3700.0\n",
      "stderr": null,
      "duration_seconds": 0.42
    }
  ]
}
```

---

## 5. Getting Started

### Prerequisites
- Python 3.11+
- Google Gemini API Key (get one from [Google AI Studio](https://aistudio.google.com/))

### Native Setup (Virtualenv)

1. **Clone and setup environment**:
   ```bash
   git clone https://github.com/armando-mio/AI-ML-Engineer-Technical-Assignment.git
   cd AI-ML-Engineer-Technical-Assignment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and supply your GEMINI_API_KEY
   ```

3. **Run the server**:
   ```bash
   uvicorn data_agent.adapters.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   Interactive Swagger docs are available at: `http://localhost:8000/docs`.

---

## 6. Running with Docker Compose

Run the entire service inside a resource-constrained container (1024MB memory limit):

```bash
# 1. Set your Gemini API key
export GEMINI_API_KEY="your-api-key"

# 2. Build and start service
docker compose up --build -d

# 3. View logs
docker compose logs -f
```

The service will be live on `http://localhost:8000`.

---

## 7. Automated Testing Suite

Execute the full suite of unit, integration, and E2E tests:

```bash
python -m pytest -v
```

### Coverage Highlights:
- **Unit (`tests/unit/`)**: Domain aggregate invariants, exceptions, LangGraph state machine transitions, and Step 1 `KeyError` $\rightarrow$ Step 2 self-healing recovery.
- **Integration (`tests/integration/`)**: Subprocess timeout kill (1.5s enforcement), network guard neutralization (blocking `socket` and `urllib`), filesystem isolation, Plotly HTML extraction, and SQLite session persistence.
- **E2E (`tests/e2e/`)**: FastAPI `TestClient` verifying multipart CSV upload, `/analyze` response schema, `/sessions/{id}`, and `/artifacts/{id}` HTML streaming.

---

## 8. Known Limitations & Future Improvements

1. **Multi-File Datasets**: Current implementation mounts a single primary tabular dataset per session. Future work could support zip archives with multi-table relational schema joins.
2. **Containerized Worker Pools**: For extreme enterprise untrusted code execution, process-level isolation can be upgraded to microVM-based sandboxes (e.g. Firecracker or gVisor) alongside the current network neutralization.
3. **Streaming Trace Updates**: Add Server-Sent Events (SSE) or WebSockets to stream reasoning steps and stdout in real-time as the agent iterates through the ReAct loop.
