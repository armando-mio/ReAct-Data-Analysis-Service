"""E2E suite testing 20 progressive analytical queries of varying complexity."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROGRESSIVE_QUERIES = [
    # Tier 1: Basic Retrieval & Aggregations
    (1, "How many total rows are in the dataset and what is the overall total revenue?"),
    (2, "What is the ranking of categories ordered by total revenue descending?"),
    (3, "What is the total units sold and average revenue for the Electronics category only?"),
    (4, "On which date did the single highest-revenue sale occur, and which category was it?"),
    # Tier 2: Computed Metrics & Date Filtering
    (5, "Calculate the average price per unit sold for each category and identify which category has the highest average unit price."),
    (6, "Compare total revenue between the first two weeks of January and the last two weeks of January 2024."),
    (7, "What percentage of total revenue is represented by the top two categories? Is there a Pareto-style concentration?"),
    (8, "On which days of the week (e.g. Monday, Tuesday, etc.) is the highest revenue concentrated?"),
    # Tier 3: Statistical Distributions & Moving Windows
    (9, "Identify any anomalous transactions or outliers in Revenue using the Interquartile Range (IQR with 1.5 threshold). Which ones are they?"),
    (10, "Calculate the correlation between Units_Sold and Revenue for each category. In which categories is the relationship strongest or weakest?"),
    (11, "Compute a 7-day rolling average of daily revenue and list the dates where actual revenue exceeded the rolling average by at least 40%."),
    (12, "Show how the cumulative percentage share of categories evolved day by day across the month."),
    # Tier 4: Trend Modeling & Complex Segmentation
    (13, "Determine the slope of the daily sales trend (linear regression) and project a revenue estimate for the next 7 days."),
    (14, "Calculate the Gini coefficient of the revenue distribution to measure inequality across individual transactions."),
    (15, "Segment transactions into 4 quadrants based on the median of Units_Sold and Unit Price (e.g. High Volume/High Price, High Volume/Low Price, etc.). How many transactions fall into each quadrant?"),
    (16, "Calculate net profit assuming the Cost column is 60% of Revenue for Electronics and 40% for other categories. What is the estimated total profit?"),
    # Tier 5: Edge Cases, Self-Healing & Unsupervised
    (17, "Show the sales performance for the Automotive category, and if it does not exist, explain which existing categories cover similar volumes."),
    (18, "Calculate the percentage change (Pct Change) in revenue between consecutive days for each category separately, explicitly handling null or infinite values."),
    (19, "Run a 1000-iteration bootstrap simulation to estimate the 95% confidence interval for mean daily revenue and calculate the 95% Value at Risk (VaR)."),
    (20, "Perform K-Means clustering (k=3) on z-score standardized Units_Sold and Revenue. Report the coordinates of the 3 centroids, sample counts per cluster, and what characterizes each cluster."),
]


@pytest.mark.parametrize("query_idx,question", PROGRESSIVE_QUERIES)
def test_progressive_query_execution(test_app: TestClient, query_idx: int, question: str):
    """Test each progressive query through the end-to-end API and sandbox pipeline."""
    csv_path = Path("data/sample_sales.csv")
    csv_bytes = csv_path.read_bytes()

    files = {
        "file": ("sample_sales.csv", io.BytesIO(csv_bytes), "text/csv"),
    }
    data = {
        "question": question,
    }

    response = test_app.post("/analyze", data=data, files=files)
    assert response.status_code == 200, f"Query #{query_idx} failed: {response.text}"

    payload = response.json()
    assert "session_id" in payload
    assert "answer" in payload
    assert len(payload["answer"].strip()) > 0
    assert "trace" in payload
    assert len(payload["trace"]) >= 1

    # Verify autonomous Plotly artifact generation
    assert "artifacts" in payload
    assert len(payload["artifacts"]) >= 1
    artifact = payload["artifacts"][0]
    assert artifact["file_name"] == "output_plot.html"
    assert artifact["url"].startswith("/artifacts/")

    # Verify artifact retrieval endpoint works
    artifact_res = test_app.get(artifact["url"])
    assert artifact_res.status_code == 200
    assert "text/html" in artifact_res.headers.get("content-type", "")


def test_session_resuming_multi_turn_continuity(test_app: TestClient):
    """Verify that submitting an existing session_id resumes conversation context across turns."""
    csv_path = Path("data/sample_sales.csv")
    csv_bytes = csv_path.read_bytes()

    # Turn 1: Initial upload and analytical query
    files = {"file": ("sample_sales.csv", io.BytesIO(csv_bytes), "text/csv")}
    turn1_res = test_app.post(
        "/analyze",
        data={"question": "What is the total revenue for the Electronics category?"},
        files=files,
    )
    assert turn1_res.status_code == 200
    payload1 = turn1_res.json()
    session_id = payload1["session_id"]
    assert session_id is not None

    # Turn 2: Follow-up question using the SAME session_id without re-uploading the file
    turn2_res = test_app.post(
        "/analyze",
        data={
            "question": "Now compare that previous result to the Clothing category.",
            "session_id": session_id,
        },
    )
    assert turn2_res.status_code == 200
    payload2 = turn2_res.json()
    assert payload2["session_id"] == session_id
    assert payload2["status"] == "success"

    # Turn 3: Verify the full session contains both turns (2 user messages + 2 assistant messages)
    session_res = test_app.get(f"/sessions/{session_id}")
    assert session_res.status_code == 200
    session_data = session_res.json()
    assert len(session_data["messages"]) >= 4
    roles = [m["role"] for m in session_data["messages"]]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 2

