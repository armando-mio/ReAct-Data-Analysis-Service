# ReAct Data Analysis Service

A production-ready, testable, and robust ReAct-style Data Analysis Service following **Hexagonal Architecture (Ports and Adapters)** powered by **LangGraph**, **SQLite**, **FastAPI**, and **Google Gemini API**.

---

## 1. System Architecture

The service enforces a strict inward dependency direction:
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

### ReAct Agent Loop & Self-Healing State Machine
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

- **Self-Healing Recovery**: If the generated Python code raises runtime or syntax errors (e.g. `KeyError`, `ValueError`), the **Reflection Node** catches the failure and routes execution back to **Code Generator** with error feedback and the previous code, correcting the execution without crashing the service.

---

## 2. Key Design Decisions

### A. Why this Datastore (SQLite with SQLAlchemy 2.0)
- **Zero Daemon Overhead**: Unlike PostgreSQL or MySQL, SQLite runs in-process with zero extra container bloat and minimal memory footprint, making it ideal for the 1024MB RAM budget.
- **ACID-Compliant State & Resumption**: Supports multi-turn conversations, historical trace retrieval, and generated artifact references across service restarts via persistent volume mounts.
- **Portability & Simplicity**: The entire database state resides in a single file (`storage/data_agent.db`), trivial to mount, back up, or test in-memory (`sqlite:///:memory:`).

### B. Why this API Contract (FastAPI + Multipart `/analyze`)
- **Single-Flight Atomic Submissions**: The `POST /analyze` endpoint accepts `multipart/form-data`, allowing the user to submit an analytical prompt and a new CSV dataset file in one atomic request without separate upload ceremonies.
- **Autonomous Plotly Visualization Mandate**: The client does not need to explicitly instruct the agent to "generate a chart" or mention Plotly in the query. The ReAct agent autonomously determines and executes the optimal interactive Plotly visualization for every analytical query as a core feature of the service.
- **Conversational Resumption**: Accepts an optional `session_id` to continue multi-turn analysis over an existing dataset without re-uploading the file.
- **Comprehensive Response Payload**: Returns a unified JSON schema containing the natural language answer, direct URLs to generated visualizations (`/artifacts/{id}`), and the step-by-step reasoning trace (thought, code, stdout, stderr, duration).
- **Direct HTML Streaming**: `GET /artifacts/{id}` streams standalone Plotly HTML files (`media_type="text/html"`), allowing immediate in-browser rendering.

### C. Why this Sandbox Strategy (Subprocess Isolation + Network Neutralization)
- **Pragmatic Security without Over-Engineering**: Spawning a separate Python subprocess inside an isolated `tempfile.TemporaryDirectory()` avoids the complexity, latency, and host privileges needed for Docker-in-Docker or VM-level sandboxes.
- **Strict Timeout Enforcement**: A hard execution timeout (15s default) terminates the entire subprocess tree upon expiry, raising `SandboxTimeoutError` to prevent infinite loops or CPU starvation.
- **Filesystem Isolation**: Code executes inside an isolated temporary directory; input datasets are copied locally so host files cannot be overwritten.
- **Network Neutralization**: Sockets (`socket.socket.connect`, `socket.create_connection`, `urllib.request`, `http.client`) are patched at startup by `network_guard.py` to immediately raise `PermissionError`, preventing data exfiltration or SSRF.

### D. Strategy for Testing Non-Deterministic LLM Components
- **Hexagonal Port Decoupling**: Core domain logic and use cases depend exclusively on the abstract `ILLMClient` port, never importing vendor SDKs directly.
- **Deterministic Mocking (`MockLLMAdapter`)**: Replaces external LLM calls with deterministic, scripted responses for testing state transitions, self-healing error recovery, and schema output consistency with zero network flakiness, latency, or API costs.
- **Contract & Schema Invariance**: Assertions validate structural invariants (state machine node transitions, required dictionary keys, non-empty outputs, valid Plotly HTML tags) rather than brittle verbatim string equality.
- **Safety Boundary Invariance**: Sandbox security tests (timeout kill, socket blocking, filesystem confinement) execute actual Python subprocesses with injected guards, verifying security independently of LLM outputs.

---

## 3. Known Limitations

1. **Single-Table Scope**: The current pipeline mounts a single primary tabular dataset (`.csv`) per analysis session. It does not natively ingest multi-table archives (e.g. zip of multiple CSVs with relational joins) in a single prompt.
2. **Process-Level Sandbox vs MicroVM**: While the subprocess sandbox enforces timeouts, temporary filesystem boundaries, and socket neutralization, highly hostile untrusted environments in multi-tenant cloud platforms would benefit from microVM isolation (e.g. Firecracker or gVisor).
3. **Rate Limits on Free-Tier LLM APIs**: Google AI Studio free tier limits requests to 5–15 requests per minute, which can be reached quickly during multi-turn ReAct loops. The system includes an automatic resilient fallback to local execution to prevent service downtime.

---

## 4. What Would Be Improved with More Time

1. **Streaming Reasoning Traces (SSE / WebSockets)**: Implement Server-Sent Events (SSE) so users can watch the agent's thoughts, code generation, and stdout stream in real time.
2. **DSPy Prompt Optimization (GEPA)**: Integrate DSPy modules and the GEPA optimizer on a curated dev set of data analysis questions to optimize prompt instructions automatically.
3. **Multi-File & SQL Ingestion**: Support relational multi-file datasets and direct read-only SQL connection strings as dataset inputs.
4. **Automated Plot Quality Evaluation (LLM-as-a-Judge)**: Add a secondary evaluation node assessing chart visual quality, label legibility, and metric alignment before finalizing.

---

## 5. Deliverables & Operational References

| # | Deliverable | Location in Repository |
| :-: | :--- | :--- |
| **1** | **Source Code** | Entire codebase structured under [`data_agent/`](data_agent/) |
| **2** | **README (Mandatory)** | This document ([`README.md`](README.md)) |
| **3** | **Example Plotly HTML Output** | Standalone visualization at [`examples/example_plot.html`](examples/example_plot.html) |
| **4** | **Instructions to start system (`docker compose up`)** | Documented in detail in [`INSTRUCTIONS.md`](INSTRUCTIONS.md#1-start-the-system-with-docker-compose-docker-compose-up) |
| **5** | **Command to run the test suite** | Documented in detail in [`INSTRUCTIONS.md`](INSTRUCTIONS.md#2-command-to-run-the-test-suite-pytest) |

> [!TIP]
> For instructions on running the service via **Docker Compose**, running the **test suite**, or testing the API endpoints, see **[`INSTRUCTIONS.md`](INSTRUCTIONS.md)**.
