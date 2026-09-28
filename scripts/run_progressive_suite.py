"""Benchmark runner executing all 20 progressive queries against the live API."""

import json
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
CSV_FILE = Path("data/sample_sales.csv")

QUERIES = [
    # Tier 1
    (1, "Quante righe totali ci sono nel dataset e qual è il fatturato complessivo?"),
    (2, "Qual è la classifica delle categorie ordinate per fatturato totale decrescente?"),
    (3, "Qual è il numero totale di unità vendute e il ricavo medio per la sola categoria Electronics?"),
    (4, "In quale data si è verificata la singola vendita con il ricavo più alto e di quale categoria si trattava?"),
    # Tier 2
    (5, "Calcola il prezzo medio per unità venduta per ciascuna categoria e dimmi quale categoria ha il prezzo medio unitario più elevato."),
    (6, "Confronta il fatturato totale delle prime due settimane di gennaio con le ultime due settimane di gennaio 2024."),
    (7, "Quale percentuale del fatturato totale è rappresentata dalle prime due categorie? C'è una concentrazione stile Pareto?"),
    (8, "In quali giorni della settimana (es. Lunedì, Martedì, ecc.) si concentra il maggior fatturato?"),
    # Tier 3
    (9, "Identifica eventuali transazioni anomale o outlier nel Revenue utilizzando l'Interquartile Range (IQR con soglia 1.5). Quali sono?"),
    (10, "Calcola la correlazione tra Units_Sold e Revenue per ciascuna categoria. In quali categorie la relazione è più forte o più debole?"),
    (11, "Calcola la media mobile a 7 giorni del fatturato giornaliero ed elenca le date in cui il fatturato reale ha superato la media mobile di almeno il 40%."),
    (12, "Mostra come è cambiata la quota percentuale cumulata delle categorie giorno per giorno nel corso del mese."),
    # Tier 4
    (13, "Determina la pendenza del trend giornaliero delle vendite (regressione lineare) e proietta una stima del fatturato per i successivi 7 giorni."),
    (14, "Calcola l'indice di Gini della distribuzione del fatturato per misurare il livello di disuguaglianza tra le singole transazioni."),
    (15, "Segmenta le transazioni in 4 quadranti basandoti sulla mediana di Units_Sold e del Prezzo Unitario (es. Alto Volume/Alto Prezzo, Alto Volume/Basso Prezzo, ecc.). Quante transazioni ricadono in ciascun quadrante?"),
    (16, "Calcola il margine netto ipotizzando che la colonna Cost sia pari al 60% del Revenue per l'elettronica e al 40% per le altre categorie. Qual è il profitto stimato?"),
    # Tier 5
    (17, "Mostrami l'andamento delle vendite della categoria Automotive, e se non esiste spiegami quali categorie coprono volumi simili."),
    (18, "Calcola la variazione percentuale (Pct Change) del fatturato tra giorni consecutivi per ogni singola categoria separatamente, gestendo esplicitamente i valori nulli o infiniti."),
    (19, "Esegui una simulazione Bootstrap a 1000 iterazioni per stimare l'intervallo di confidenza al 95% del fatturato medio giornaliero e calcola il Value at Risk (VaR 95%)."),
    (20, "Esegui un clustering K-Means (k=3) su Units_Sold e Revenue dopo averli standardizzati con z-score. Riporta le coordinate dei 3 centroidi, il numero di campioni per cluster e spiega cosa caratterizza ciascun cluster."),
]


def run_suite():
    print(f"=== Starting Progressive Execution of {len(QUERIES)} Test Queries ===")
    print(f"Target: {BASE_URL}/analyze")
    print(f"Dataset: {CSV_FILE} ({CSV_FILE.stat().st_size} bytes)\n")

    results = []

    for idx, question in QUERIES:
        print(f"\n--- [{idx:02d}/20] Running Query ---")
        print(f"Prompt: {question}")
        t0 = time.time()

        try:
            with open(CSV_FILE, "rb") as f:
                response = requests.post(
                    f"{BASE_URL}/analyze",
                    data={"question": question},
                    files={"file": ("sample_sales.csv", f, "text/csv")},
                    timeout=120,
                )
            elapsed = time.time() - t0

            if response.status_code == 200:
                data = response.json()
                session_id = data.get("session_id", "N/A")
                answer = data.get("answer", "")
                artifacts = data.get("artifacts", [])
                trace = data.get("trace", [])

                plot_url = artifacts[0].get("url") if artifacts else "None"
                steps = len(trace)

                print(f"Status: HTTP 200 (took {elapsed:.2f}s, {steps} ReAct step(s))")
                print(f"Plot Artifact: {plot_url}")
                print(f"Answer Preview: {answer[:180]}...")

                results.append({
                    "idx": idx,
                    "question": question,
                    "status": 200,
                    "elapsed_sec": round(elapsed, 2),
                    "steps": steps,
                    "plot_url": plot_url,
                    "answer_preview": answer[:250],
                    "success": True,
                })
            else:
                print(f"FAILED with HTTP {response.status_code}: {response.text[:200]}")
                results.append({
                    "idx": idx,
                    "question": question,
                    "status": response.status_code,
                    "elapsed_sec": round(elapsed, 2),
                    "error": response.text[:200],
                    "success": False,
                })

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"EXCEPTION: {exc}")
            results.append({
                "idx": idx,
                "question": question,
                "status": "EXCEPTION",
                "elapsed_sec": round(elapsed, 2),
                "error": str(exc),
                "success": False,
            })

        # Rate limiting delay between queries
        if idx < len(QUERIES):
            print("Pacing: waiting 8s to respect API rate limits...")
            time.sleep(8)

    # Save summary report
    out_file = Path("storage/benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for r in results if r.get("success"))
    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY: {passed}/{len(QUERIES)} queries completed successfully (HTTP 200).")
    print(f"Results saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_suite()
