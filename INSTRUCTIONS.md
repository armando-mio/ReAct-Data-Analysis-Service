# System Execution & Testing Instructions

This document provides operational instructions to run, test, and verify the **ReAct Data Analysis Service**, fulfilling deliverables **#4 (Instructions to start the system with `docker compose up`)** and **#5 (Command to run the test suite)**.

---

## 1. Start the System with Docker Compose (`docker compose up`)

The entire production service is containerized with a strict memory limit of **1024MB** and persistent host volume mounts for datasets and artifacts.

### Prerequisites
- Docker & Docker Compose installed.

### Steps to Run

1. **Configure Environment Variables**:
   Ensure `.env` exists in the project root with your Gemini API key and model:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-3.6-flash
   ```

2. **Start the Service**:
   Run the following command in the repository root:
   ```bash
   docker compose up --build -d
   ```

3. **Verify Service Health**:
   Check container logs:
   ```bash
   docker compose logs -f
   ```
   Check the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```
   *Expected response*: `{"status":"healthy","service":"react-data-agent","version":"1.0.0"}`

4. **Access the Interactive API Documentation**:
   Open your browser at:
   👉 **[http://localhost:8000/docs](http://localhost:8000/docs)** (Swagger UI)

5. **Stop the Service**:
   ```bash
   docker compose down
   ```

---

## 2. Command to Run the Test Suite (`pytest`)

The automated test suite covers the agent loop (with mocked LLM), sandbox security guarantees (timeout and network blocking), SQLite persistence, and full FastAPI API contracts.

### Command to Run All Tests:

```bash
python -m pytest -v
```

Or with fail-fast mode:
```bash
python -m pytest --maxfail=1 -v
```

### Targeted Test Commands:

- **Run ReAct Loop & Self-Healing Unit Tests**:
  ```bash
  python -m pytest tests/unit/test_react_loop.py -v
  ```
- **Run Sandbox Safety & Isolation Tests (Timeout + Network Neutralization)**:
  ```bash
  python -m pytest tests/integration/test_sandbox_safety.py -v
  ```
- **Run SQLite Repository Persistence Tests**:
  ```bash
  python -m pytest tests/integration/test_repository.py -v
  ```
- **Run End-to-End API Contract Tests**:
  ```bash
  python -m pytest tests/e2e/test_api.py -v
  ```

---

## 3. Native Virtualenv Setup (Alternative to Docker)

If you prefer to run the service natively without Docker:

1. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI Server**:
   ```bash
   python -m uvicorn data_agent.adapters.api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 4. Testing the API Endpoints

A sample transactional dataset with 50 rows and 4 columns is provided at [`data/sample_sales.csv`](data/sample_sales.csv).

### A. Run Analysis via cURL (`POST /analyze`)
> **Autonomous Plotly Generation**: You do **not** need to request a chart or mention Plotly in your query. The ReAct LLM agent autonomously interprets the data, plans the best visual representation, and generates an interactive Plotly chart (`output_plot.html`) as an intrinsic feature of the service.

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "question=Quale categoria ha generato il maggior fatturato totale?" \
  -F "file=@data/sample_sales.csv"
```

### B. Retrieve Historical Session (`GET /sessions/{id}`)
```bash
curl "http://localhost:8000/sessions/<SESSION_ID>"
```

### C. Retrieve Plotly HTML Visualization (`GET /artifacts/{id}`)
```bash
curl "http://localhost:8000/artifacts/<ARTIFACT_ID>"
```
Or open the URL directly in any web browser to interact with the chart.

---

## 5. Example HTML Plot Output Deliverable

As required by deliverable **#3**, a pre-generated standalone Plotly HTML visualization is available in the repository at:
👉 [`examples/example_plot.html`](examples/example_plot.html)

Open it directly in any browser to inspect the interactive features (tooltips, zoom, pan, hover states).
