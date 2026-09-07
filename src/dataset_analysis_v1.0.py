"""
dataset_analysis.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Motore di analisi descrittiva e profilazione dei dati per la pipeline QML.

Questo modulo si occupa dell'analisi statistica descrittiva e del controllo di qualità
preliminare sui dataset reali e sintetici (German Credit, Heart Disease, Student Performance).
Il suo scopo principale è profilare e confrontare le distribuzioni delle feature classiche,
garantendo l'allineamento strutturale necessario prima dell'addestramento dei VQC.

Il sistema genera sistematicamente tre report diagnostici fondamentali:
1. Statistiche demografiche sull'età (age statistics).
2. Frequenze e distribuzioni degli attributi sensibili, isolando rigorosamente l'attributo 
   intersezionale "overall" (unione di età e sesso) per mitigare i bias di filtraggio.
3. Distribuzioni e frequenze della variabile target downstream

Per ciascun dataset (german_credit, heart_disease, student_performance)
lo script produce, in una sottocartella dedicata DatasetsAnalysis/<dataset>/:

  age_statistics.csv     -- statistiche descrittive numeriche di 'age'
                            (Min, First_Qu, Median, Mean, Third_Qu, Max,
                            Std_Dev, Var, Range, IQR), una riga per ogni
                            file CSV (reale + 15 sintetici).
  overall_statistics.csv -- tabella di frequenza dell'attributo sensibile
                            intersectional 'overall' (A_sex & A_age),
                            una riga per valore per ogni file CSV.
  target_statistics.csv  -- tabella di frequenza del target binario
                            (credit_risk_good / heart_disease_present /
                            pass_fail), una riga per valore per ogni file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_COLUMN_CANDIDATES = [
    "credit_risk_good",
    "heart_disease_present",
    "pass_fail",
    "target",
    "Target",
]


OVERALL_COLUMN = "overall"

LEGACY_SENSITIVE_COLUMNS = {"sex", "age_group", "sensitive_proxy"}


def compute_age_stats(df: pd.DataFrame, dataset_label: str) -> dict | None:
    """10 metriche descrittive per 'age', Dataset come prima colonna."""
    if "age" not in df.columns:
        return None
    s = pd.to_numeric(df["age"], errors="coerce").dropna()
    if len(s) == 0:
        return None
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return {
        "Dataset": dataset_label,
        "Min": float(s.min()),
        "First_Qu": q1,
        "Median": float(s.median()),
        "Mean": float(s.mean()),
        "Third_Qu": q3,
        "Max": float(s.max()),
        "Std_Dev": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "Var": float(s.var(ddof=1)) if len(s) > 1 else 0.0,
        "Range": float(s.max() - s.min()),
        "IQR": q3 - q1,
    }


def compute_freq_stats(df: pd.DataFrame, col_name: str, dataset_label: str) -> pd.DataFrame | None:
    """Tabella di frequenza [col_name, Frequency, Dataset] per una colonna binaria/categorica."""
    if col_name not in df.columns:
        return None
    counts = df[col_name].value_counts(dropna=False).reset_index()
    counts.columns = [col_name, "Frequency"]
    counts["Dataset"] = dataset_label
    return counts[[col_name, "Frequency", "Dataset"]]


def _find_target_column(df: pd.DataFrame) -> str | None:
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def analyze_datasets(root: str = "Quantum_Dataset_Generation_Online_Appendix") -> None:
    root_path = Path(root)
    dataset_root = root_path / "Datasets"
    if not dataset_root.exists():
        print(f"[ERRORE] Directory non trovata: {dataset_root}")
        return

    out_base = root_path / "DatasetsAnalysis"
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DATASET DESCRIPTIVE ANALYSIS -- statistiche descrittive dei dataset esportati")
    print("=" * 78)
    print(f"Directory sorgente: {dataset_root}")
    print(f"Directory output:   {out_base}")

    dataset_dirs = [p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not dataset_dirs:
        print(f"[ERRORE] Nessuna sottocartella dataset trovata in {dataset_root}")
        return

    processed = 0
    for d_dir in sorted(dataset_dirs, key=lambda p: p.name):
        csv_files = sorted(d_dir.glob("*.csv"))
        if not csv_files:
            print(f"[SKIP] Nessun CSV in {d_dir.name}")
            continue

        print(f"\nElaborazione dataset: '{d_dir.name}' ({len(csv_files)} file CSV)")
        out_dir = out_base / d_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        age_rows, overall_rows, target_rows = [], [], []
        legacy_rows: dict[str, list[pd.DataFrame]] = {c: [] for c in LEGACY_SENSITIVE_COLUMNS}

        for csv in csv_files:
            dataset_label = csv.stem
            try:
                df = pd.read_csv(csv)
            except Exception as exc:
                print(f"  [WARN] Impossibile leggere {csv.name}: {exc}")
                continue

            age_stat = compute_age_stats(df, dataset_label)
            if age_stat:
                age_rows.append(age_stat)

            overall_stat = compute_freq_stats(df, OVERALL_COLUMN, dataset_label)
            if overall_stat is not None:
                overall_rows.append(overall_stat)

            target_col = _find_target_column(df)
            if target_col:
                target_stat = compute_freq_stats(df, target_col, dataset_label)
                if target_stat is not None:
                    target_stat = target_stat.rename(columns={target_col: "target"})
                    target_rows.append(target_stat)

            for legacy_col in LEGACY_SENSITIVE_COLUMNS:
                legacy_stat = compute_freq_stats(df, legacy_col, dataset_label)
                if legacy_stat is not None:
                    legacy_rows[legacy_col].append(legacy_stat)

        if age_rows:
            pd.DataFrame(age_rows).to_csv(out_dir / "age_statistics.csv", index=False)
            print(f"  [OK] age_statistics.csv ({len(age_rows)} righe)")

        if overall_rows:
            pd.concat(overall_rows, ignore_index=True).to_csv(out_dir / "overall_statistics.csv", index=False)
            print(f"  [OK] overall_statistics.csv ({sum(len(r) for r in overall_rows)} righe)")
        else:
            print(f"  [INFO] Nessuna colonna '{OVERALL_COLUMN}' trovata in {d_dir.name}.")

        if target_rows:
            pd.concat(target_rows, ignore_index=True).to_csv(out_dir / "target_statistics.csv", index=False)
            print(f"  [OK] target_statistics.csv ({sum(len(r) for r in target_rows)} righe)")

        for legacy_col, rows in legacy_rows.items():
            if rows:
                pd.concat(rows, ignore_index=True).to_csv(out_dir / f"{legacy_col}_statistics.csv", index=False)
                print(f"  [OK] {legacy_col}_statistics.csv ({sum(len(r) for r in rows)} righe, export legacy)")

        if not (age_rows or overall_rows or target_rows):
            print(f"  [SKIP] Nessuna colonna tracciabile trovata in {d_dir.name}.")
            continue

        processed += 1

    print("\n" + "=" * 78)
    print(f"DATASET ANALYSIS COMPLETATA: {processed}/{len(dataset_dirs)} dataset elaborati.")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descriptive statistics of exported datasets")
    parser.add_argument("--root", default="Quantum_Dataset_Generation_Online_Appendix")
    args = parser.parse_args()
    analyze_datasets(root=args.root)
