# Operational Runbook & Execution Instructions

This operational guide provides step-by-step instructions to configure, run, and evaluate the **ReAct Data Analysis Service**.

---

## 1. Prerequisites

Before evaluating or running the application, ensure the following tooling is available on your host environment:
- **Python 3.11+** (tested and verified on Python 3.11 – 3.14)
- **Docker** & **Docker Compose** (v2.0+)
- **cURL** or an HTTP client (e.g. Postman, HTTPie)
- A valid **Google Gemini API Key** (obtainable from Google AI Studio)

---

## 2. Environment Configuration

The application strictly requires a valid Gemini API key to operate in live execution mode.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/armando-mio/AI-ML-Engineer-Technical-Assignment.git
   cd AI-ML-Engineer-Technical-Assignment
   ```

2. **Initialize the environment file**:
   ```bash
   cp .env.example .env
   ```

3. **Configure the API key in `.env`**:
   Open `.env` in an editor and set your key:
   ```env
   # Google Gemini API Configuration (Mandatory for live LLM operations)
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL=gemini-3.1-flash-lite

   # SQLite Database Connection
   DATABASE_URL=sqlite:///storage/data_agent.db

   # Storage Directories
   ARTIFACTS_DIR=storage/artifacts
   UPLOADS_DIR=storage/uploads

   # Sandbox Safety Threshold (seconds)
   SANDBOX_TIMEOUT=15.0
   ```

> **Note**: If `GEMINI_API_KEY` is omitted or empty on startup, the application deliberately raises a fatal configuration error (`ValueError: GEMINI_API_KEY environment variable is required`), preventing silent degradation or unexpected failures.

---

## 3. Quickstart with Docker (Primary Evaluation Path)

Docker Compose provides the simplest, zero-configuration evaluation path. It encapsulates all system dependencies, volume persistence, and networking.

### Build and Start Containers
```bash
docker compose up --build
```
*(To run in detached/background mode, append `-d`)*

### Verify Service Health
In a separate terminal, test the health check endpoint:
```bash
curl -s http://localhost:8000/health
```
**Expected response**:
```json
{"status":"healthy","service":"react-data-agent","version":"1.0.0"}
```

### Access API Documentation
Open your web browser and navigate to:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Stop the Service
```bash
docker compose down
```

---

## 4. Local Development Setup

If you prefer to run the service natively outside of Docker:

1. **Create and activate a virtual environment**:
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install project dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Launch the development server with live reload**:
   ```bash
   uvicorn data_agent.adapters.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

The service will start listening on `http://localhost:8000`.

---

## 5. Running the Automated Test Suite

The test suite validates the entire architecture without making external network calls or requiring active API keys. Non-deterministic LLM behavior is isolated using deterministic test doubles strictly confined to `tests/`.

### Run All Tests
```bash
pytest -v tests/
```

### Test Hierarchy
The test suite consists of 60 automated tests divided into three cohesive layers:

1. **Unit Tests (`tests/unit/`)**:
   - `test_domain.py`: Validates pure domain models (`Session`, `TraceStep`, `Artifact`, `MessageRole`), exception hierarchies, code sanitizer/extractor, and structured JSON log formatting.
   - `test_dspy_modules.py`: Verifies DSPy signature declarations, predictor wiring, and container configuration guards.
   - `test_react_loop.py`: Validates the LangGraph ReAct state machine, Plan -> Act -> Observe -> Iterate cycles, and self-healing recovery upon code execution errors.

2. **Integration Tests (`tests/integration/`)**:
   - `test_sandbox_safety.py`: Validates process-level execution timeout enforcement, deterministic network neutralization (blocking raw sockets and `urllib`), and neutralization of interactive GUI calls (`fig.show()`, `plt.show()`).
   - `test_repository.py`: Verifies SQLite transactional persistence, WAL mode concurrency, multi-turn message appending, and trace serialization.
   - `test_dspy_optimization.py`: Verifies the dev set construction, multi-aspect scoring metric, and offline prompt optimization runner.

3. **End-to-End Tests (`tests/e2e/`)**:
   - `test_api.py`: Validates the complete HTTP contract (`POST /analyze`, `GET /sessions/{id}`, `GET /artifacts/{id}`), file upload handling, error mapping, and missing entity 404 responses.
   - `test_progressive_queries.py`: Executes 20 progressive analytical queries against `data/sample_sales.csv` through the full sandbox pipeline, followed by multi-turn conversation continuity tests.

---

## 6. API Usage & Example Workflows

The following concrete `curl` commands illustrate key operational workflows.

### Workflow 1: Start a New Analytical Session
Submit a tabular question along with the included sample sales CSV dataset:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "question=What is the total revenue and units sold by category? Generate an interactive chart." \
  -F "file=@data/sample_sales.csv;type=text/csv"
```

**Sample Response**:
```json
{
  "session_id": "8fa19e83-74b2-4d9f-a2e6-c148bb0174ad",
  "status": "success",
  "answer": "Analysis complete. Total revenue and units sold were computed across all categories. Electronics generated the highest total revenue ($14,508.62), followed by Clothing ($7,767.26).",
  "artifact_id": "c71e24fb-81df-4b95-a50d-d42199f7dca1",
  "artifact_url": "/artifacts/c71e24fb-81df-4b95-a50d-d42199f7dca1",
  "artifacts": [
    {
      "id": "c71e24fb-81df-4b95-a50d-d42199f7dca1",
      "file_name": "output_plot.html",
      "url": "/artifacts/c71e24fb-81df-4b95-a50d-d42199f7dca1"
    }
  ],
  "trace": [
    {
      "step_index": 0,
      "thought": "1. Load dataset.csv. 2. Group by Category and compute sum of Revenue and Units_Sold. 3. Output interactive bar chart to output_plot.html.",
      "code": "import pandas as pd\nimport plotly.express as px\n...",
      "stdout": "Category Revenue Summary:\nElectronics: 14508.62\n...",
      "stderr": null,
      "duration_seconds": 1.42
    }
  ]
}
```

### Workflow 2: Progressive Drill-Down (Resume Existing Session)
Ask a follow-up question in the same conversation without re-uploading the dataset:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "session_id=8fa19e83-74b2-4d9f-a2e6-c148bb0174ad" \
  -F "question=Now isolate the Electronics category and identify which individual transaction had the highest unit price."
```

### Workflow 3: Retrieve Full Session Audit Trail
Retrieve the complete conversational record, full reasoning trace, and generated artifacts:

```bash
curl -s http://localhost:8000/sessions/8fa19e83-74b2-4d9f-a2e6-c148bb0174ad
```

### Workflow 4: Inspect Generated Plotly Visualization
Fetch and view the generated interactive Plotly HTML chart in your browser:
```bash
# Terminal download:
curl -s http://localhost:8000/artifacts/c71e24fb-81df-4b95-a50d-d42199f7dca1 -o plot.html

# Or open directly in browser:
# Navigate to http://localhost:8000/artifacts/c71e24fb-81df-4b95-a50d-d42199f7dca1
```

---

## 7. Offline Prompt Optimization (DSPy GEPA)

To reproduce the prompt evolution process or re-train the DSPy signatures on a customized dataset:

```bash
# Ensure GEMINI_API_KEY is exported
export GEMINI_API_KEY="your_api_key_here"

# Execute the evolutionary prompt optimizer CLI
python -m data_agent.optimization.optimizer --output examples/gepa_optimization_report.json
```

The script will:
1. Load the development set questions from `data_agent/optimization/dev_set.py`.
2. Measure zero-shot baseline performance across the sandbox execution and Plotly generation metric.
3. Apply evolutionary mutations to refine the planning, code generation, and reflection prompts.
4. Export the quantitative comparison report to the specified JSON path.
