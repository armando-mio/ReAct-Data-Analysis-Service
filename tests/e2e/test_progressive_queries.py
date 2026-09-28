"""E2E suite testing 20 progressive analytical queries of varying complexity."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROGRESSIVE_QUERIES = [
    # Tier 1: Basic Retrieval & Aggregations
    (1, "Quante righe totali ci sono nel dataset e qual è il fatturato complessivo?"),
    (2, "Qual è la classifica delle categorie ordinate per fatturato totale decrescente?"),
    (3, "Qual è il numero totale di unità vendute e il ricavo medio per la sola categoria Electronics?"),
    (4, "In quale data si è verificata la singola vendita con il ricavo più alto e di quale categoria si trattava?"),
    # Tier 2: Computed Metrics & Date Filtering
    (5, "Calcola il prezzo medio per unità venduta per ciascuna categoria e dimmi quale categoria ha il prezzo medio unitario più elevato."),
    (6, "Confronta il fatturato totale delle prime due settimane di gennaio con le ultime due settimane di gennaio 2024."),
    (7, "Quale percentuale del fatturato totale è rappresentata dalle prime due categorie? C'è una concentrazione stile Pareto?"),
    (8, "In quali giorni della settimana (es. Lunedì, Martedì, ecc.) si concentra il maggior fatturato?"),
    # Tier 3: Statistical Distributions & Moving Windows
    (9, "Identifica eventuali transazioni anomale o outlier nel Revenue utilizzando l'Interquartile Range (IQR con soglia 1.5). Quali sono?"),
    (10, "Calcola la correlazione tra Units_Sold e Revenue per ciascuna categoria. In quali categorie la relazione è più forte o più debole?"),
    (11, "Calcola la media mobile a 7 giorni del fatturato giornaliero ed elenca le date in cui il fatturato reale ha superato la media mobile di almeno il 40%."),
    (12, "Mostra come è cambiata la quota percentuale cumulata delle categorie giorno per giorno nel corso del mese."),
    # Tier 4: Trend Modeling & Complex Segmentation
    (13, "Determina la pendenza del trend giornaliero delle vendite (regressione lineare) e proietta una stima del fatturato per i successivi 7 giorni."),
    (14, "Calcola l'indice di Gini della distribuzione del fatturato per misurare il livello di disuguaglianza tra le singole transazioni."),
    (15, "Segmenta le transazioni in 4 quadranti basandoti sulla mediana di Units_Sold e del Prezzo Unitario (es. Alto Volume/Alto Prezzo, Alto Volume/Basso Prezzo, ecc.). Quante transazioni ricadono in ciascun quadrante?"),
    (16, "Calcola il margine netto ipotizzando che la colonna Cost sia pari al 60% del Revenue per l'elettronica e al 40% per le altre categorie. Qual è il profitto stimato?"),
    # Tier 5: Edge Cases, Self-Healing & Unsupervised
    (17, "Mostrami l'andamento delle vendite della categoria Automotive, e se non esiste spiegami quali categorie coprono volumi simili."),
    (18, "Calcola la variazione percentuale (Pct Change) del fatturato tra giorni consecutivi per ogni singola categoria separatamente, gestendo esplicitamente i valori nulli o infiniti."),
    (19, "Esegui una simulazione Bootstrap a 1000 iterazioni per stimare l'intervallo di confidenza al 95% del fatturato medio giornaliero e calcola il Value at Risk (VaR 95%)."),
    (20, "Esegui un clustering K-Means (k=3) su Units_Sold e Revenue dopo averli standardizzati con z-score. Riporta le coordinate dei 3 centroidi, il numero di campioni per cluster e spiega cosa caratterizza ciascun cluster."),
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
