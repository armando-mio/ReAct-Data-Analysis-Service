"""End-to-End API contract tests using FastAPI TestClient."""

import io
from fastapi.testclient import TestClient


def test_post_analyze_with_csv(test_app: TestClient, sample_csv_content: bytes):
    """Test POST /analyze with multipart CSV upload and full ReAct loop response."""
    files = {
        "file": ("sales_data.csv", io.BytesIO(sample_csv_content), "text/csv"),
    }
    data = {
        "question": "What is the total sales amount by region?",
    }

    response = test_app.post("/analyze", data=data, files=files)
    assert response.status_code == 200

    payload = response.json()
    assert "session_id" in payload
    assert "answer" in payload
    assert isinstance(payload["answer"], str)
    assert "trace" in payload
    assert isinstance(payload["trace"], list)
    assert len(payload["trace"]) >= 1

    # Verify trace structure
    first_step = payload["trace"][0]
    assert "step_index" in first_step
    assert "thought" in first_step
    assert "stdout" in first_step
    assert "duration_seconds" in first_step

    # Verify artifacts
    assert "artifacts" in payload
    assert isinstance(payload["artifacts"], list)
    if payload["artifacts"]:
        artifact = payload["artifacts"][0]
        assert "id" in artifact
        assert "file_name" in artifact
        assert "url" in artifact
        assert artifact["url"].startswith("/artifacts/")


def test_session_and_artifact_retrieval_flow(test_app: TestClient, sample_csv_content: bytes):
    """Test full cycle: Analyze -> GET /sessions/{id} -> GET /artifacts/{id}."""
    # 1. Analyze
    files = {
        "file": ("sales_data.csv", io.BytesIO(sample_csv_content), "text/csv"),
    }
    data = {
        "question": "Which product category generated the highest total revenue?",
    }
    analyze_res = test_app.post("/analyze", data=data, files=files)
    assert analyze_res.status_code == 200
    analyze_payload = analyze_res.json()
    session_id = analyze_payload["session_id"]

    # 2. Get session details
    session_res = test_app.get(f"/sessions/{session_id}")
    assert session_res.status_code == 200
    session_payload = session_res.json()
    assert session_payload["session_id"] == session_id
    assert len(session_payload["messages"]) >= 2  # user prompt + assistant answer
    assert len(session_payload["traces"]) >= 1

    # 3. Get artifact if generated
    if analyze_payload["artifacts"]:
        artifact_id = analyze_payload["artifacts"][0]["id"]
        art_res = test_app.get(f"/artifacts/{artifact_id}")
        assert art_res.status_code == 200
        assert "text/html" in art_res.headers["content-type"]
        assert len(art_res.content) > 0


def test_analyze_invalid_csv_validation_error(test_app: TestClient):
    """Test that missing dataset or unparseable upload returns HTTP 400."""
    data = {
        "question": "Analyze with no data provided",
    }
    response = test_app.post("/analyze", data=data)
    assert response.status_code == 400
    assert "No dataset available" in response.json()["detail"]


def test_get_non_existent_session_404(test_app: TestClient):
    """Test that querying an invalid session ID returns HTTP 404."""
    response = test_app.get("/sessions/non-existent-session-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
