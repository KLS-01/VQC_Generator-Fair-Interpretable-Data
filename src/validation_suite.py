"""
validation_suite.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Suite di validazione e controllo qualità per l'integrità dei dati e dei test QML

Questo modulo funge da gatekeeper di qualità automatizzato (Quality Assurance Gatekeeper)
per verificare l'integrità strutturale e la consistenza numerica dei dati sperimentali.
Esegue audit sistematici sui dataset sintetici e reali (German Credit, Heart Disease,
Student Performance), validando le relazioni inter-fase e la correttezza formale dei test.

Il sistema assicura che le matrici dei test statistici siano perfettamente quadrate e
simmetriche, che i p-value siano matematicamente validi e che l'unione dei dati delle
diverse fasi della pipeline (merge validation) non presenti record nulli o mancanti.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIRS = {
    "german_credit": "German",
    "heart_disease": "Heart",
    "student_performance": "Student",
}
EXPECTED_CONFIGS = {
    "L1_linear", "L1_all_to_all", "L2_linear",
    "L2_all_to_all", "L2_re_upload", "L2_all_to_all_re_upload",
}
EXPECTED_SEEDS = {7, 17, 27}
EXPECTED_MODELS = {"RandomForest", "XGBoost_stub", "MLP"}
EXPECTED_ATTRIBUTES = {"age", "sex", "overall"}


def record_check(condition, ok_message, error_message, report_lines, error_lines):
    if condition:
        report_lines.append(f"[OK] {ok_message}")
    else:
        report_lines.append(f"[ERROR] {error_message}")
        error_lines.append(f"[ERROR] {error_message}")


def check_file_exists(path, report_lines, error_lines):
    if not path.exists():
        msg = f"[MISSING] {path}"
        report_lines.append(msg)
        error_lines.append(msg)
        return False
    report_lines.append(f"[OK] {path}")
    return True


def validate_raw(root, report_lines, error_lines):
    analysis = Path(root) / "Analysis"
    paths = {
        "RQ1": analysis / "RQ1_data" / "rq1_models_results.csv",
        "RQ2": analysis / "RQ2_data" / "rq2_all_fairness_results_multiattribute.csv",
        "RQ3": analysis / "RQ3_data" / "rq3_all_results.csv",
    }
    frames = {}
    for label, path in paths.items():
        if check_file_exists(path, report_lines, error_lines):
            try:
                frames[label] = pd.read_csv(path)
            except Exception as exc:
                msg = f"[ERROR] {label}: impossibile leggere {path}: {exc!r}"
                report_lines.append(msg)
                error_lines.append(msg)

    if not all(k in frames for k in paths):
        return frames

    rq1, rq2, rq3 = frames["RQ1"], frames["RQ2"], frames["RQ3"]
    key_rq1 = ["dataset", "configuration", "seed", "model"]
    key_rq2 = ["dataset", "configuration", "seed", "model", "attribute"]
    key_rq3 = ["dataset", "configuration", "seed", "gate_group"]

    for label, df, key in [
        ("RQ1", rq1, key_rq1),
        ("RQ2", rq2, key_rq2),
        ("RQ3", rq3, key_rq3),
    ]:
        missing = [c for c in key if c not in df.columns]
        record_check(
            not missing,
            f"{label}: colonne chiave presenti",
            f"{label}: colonne chiave mancanti: {missing}",
            report_lines, error_lines,
        )
        if missing:
            continue
        duplicate_count = int(df.duplicated(key).sum())
        record_check(
            duplicate_count == 0,
            f"{label}: chiavi uniche",
            f"{label}: {duplicate_count} righe duplicate rispetto alla chiave {key}",
            report_lines, error_lines,
        )

    expected_rows_rq1 = len(EXPECTED_CONFIGS) * len(EXPECTED_SEEDS) * len(EXPECTED_MODELS) * len(DATASET_DIRS)
    expected_rows_rq2 = expected_rows_rq1 * len(EXPECTED_ATTRIBUTES)
    record_check(
        len(rq1) == expected_rows_rq1,
        f"RQ1: cardinalita' {expected_rows_rq1} righe",
        f"RQ1: cardinalita' {len(rq1)} invece di {expected_rows_rq1}",
        report_lines, error_lines,
    )
    record_check(
        len(rq2) == expected_rows_rq2,
        f"RQ2: cardinalita' {expected_rows_rq2} righe",
        f"RQ2: cardinalita' {len(rq2)} invece di {expected_rows_rq2}",
        report_lines, error_lines,
    )

    for dataset in DATASET_DIRS:
        for label, df in [("RQ1", rq1), ("RQ2", rq2), ("RQ3", rq3)]:
            sub = df[df["dataset"] == dataset]
            if label == "RQ1":
                expected = len(EXPECTED_CONFIGS) * len(EXPECTED_SEEDS) * len(EXPECTED_MODELS)
            elif label == "RQ2":
                expected = len(EXPECTED_CONFIGS) * len(EXPECTED_SEEDS) * len(EXPECTED_MODELS) * len(EXPECTED_ATTRIBUTES)
            else:
                gate_groups = set(sub["gate_group"].dropna())
                expected = len(EXPECTED_CONFIGS) * len(EXPECTED_SEEDS) * len(gate_groups)
            record_check(
                len(sub) == expected,
                f"{dataset}/{label}: cardinalita' {expected}",
                f"{dataset}/{label}: cardinalita' {len(sub)} invece di {expected}",
                report_lines, error_lines,
            )

        for label, df, column, expected in [
            ("RQ1", rq1, "configuration", EXPECTED_CONFIGS),
            ("RQ2", rq2, "configuration", EXPECTED_CONFIGS),
            ("RQ3", rq3, "configuration", EXPECTED_CONFIGS),
        ]:
            values = set(df.loc[df["dataset"] == dataset, column].dropna())
            record_check(
                values == expected,
                f"{dataset}/{label}: configurazioni complete",
                f"{dataset}/{label}: configurazioni {sorted(values)} invece di {sorted(expected)}",
                report_lines, error_lines,
            )

        seeds = set(rq3.loc[rq3["dataset"] == dataset, "seed"].dropna().astype(int))
        record_check(
            seeds == EXPECTED_SEEDS,
            f"{dataset}: seed completi",
            f"{dataset}: seed {sorted(seeds)} invece di {sorted(EXPECTED_SEEDS)}",
            report_lines, error_lines,
        )

    return frames


def validate_matrix(path, report_lines, error_lines):
    if not check_file_exists(path, report_lines, error_lines):
        return
    try:
        matrix = pd.read_csv(path, index_col=0)
        numeric = matrix.to_numpy(dtype=float)
        record_check(
            matrix.shape[0] == matrix.shape[1],
            f"{path}: matrice quadrata",
            f"{path}: matrice non quadrata {matrix.shape}",
            report_lines, error_lines,
        )
        record_check(
            np.allclose(numeric, numeric.T, atol=1e-6),
            f"{path}: matrice simmetrica",
            f"{path}: matrice non simmetrica",
            report_lines, error_lines,
        )
        record_check(
            np.isfinite(numeric).all() and (numeric >= 0).all() and (numeric <= 1).all(),
            f"{path}: valori p-validi",
            f"{path}: valori non finiti o fuori da [0,1]",
            report_lines, error_lines,
        )
    except Exception as exc:
        msg = f"[ERROR] {path}: errore nella validazione matrice: {exc!r}"
        report_lines.append(msg)
        error_lines.append(msg)


def validate_statistical_results(root, report_lines, error_lines):
    results = Path(root) / "StatisticalResults"
    for dataset, ddir in DATASET_DIRS.items():
        for metric in ["f1", "accuracy"]:
            validate_matrix(results / "RQ1" / ddir / f"{metric}_NemenyiTestResults.csv", report_lines, error_lines)
        for attribute in ["age", "sex", "overall"]:
            for metric in ["spd", "eod", "aod"]:
                validate_matrix(results / "RQ2" / ddir / attribute / f"{metric}_NemenyiTestResults.csv", report_lines, error_lines)
        validate_matrix(results / "RQ3" / ddir / "gate" / "svqx_value_NemenyiTestResults.csv", report_lines, error_lines)
        for metric in ["svqx_concentration", "entanglement_score", "qsalto_a1_estimate"]:
            validate_matrix(results / "RQ3" / ddir / "run" / f"{metric}_NemenyiTestResults.csv", report_lines, error_lines)

        gate_friedman = results / "RQ3" / ddir / "gate" / "friedman.csv"
        if check_file_exists(gate_friedman, report_lines, error_lines):
            df = pd.read_csv(gate_friedman)
            record_check(
                "n_blocks" in df and not df.empty and (df["n_blocks"] > 0).all(),
                f"{dataset}/RQ3 gate: n_blocks presente",
                f"{dataset}/RQ3 gate: n_blocks inatteso",
                report_lines, error_lines,
            )

        stability = results / "RQ3" / ddir / "gate" / "svqx_stability_spearman.csv"
        if check_file_exists(stability, report_lines, error_lines):
            df = pd.read_csv(stability)
            expected = {"dataset", "configuration", "spearman_rho_7_17", "spearman_p_7_17", "spearman_rho_7_27", "spearman_p_7_27", "spearman_rho_17_27", "spearman_p_17_27", "mean_rho"}
            record_check(expected <= set(df.columns), f"{dataset}: colonne stabilita' presenti", f"{dataset}: colonne stabilita' mancanti", report_lines, error_lines)
            record_check(set(df["configuration"]) == EXPECTED_CONFIGS, f"{dataset}: configurazioni stabilita' complete", f"{dataset}: configurazioni stabilita' incomplete", report_lines, error_lines)
            record_check(df["mean_rho"].between(-1, 1).all(), f"{dataset}: mean_rho in [-1,1]", f"{dataset}: mean_rho fuori intervallo", report_lines, error_lines)
            if not (df["mean_rho"] > 0.5).any():
                report_lines.append(f"[WARN] {dataset}: nessuna configurazione supera mean_rho > 0.5 (non bloccante)")

        complexity = results / "RQ3" / ddir / "run" / "complexity_correlations.csv"
        if check_file_exists(complexity, report_lines, error_lines):
            df = pd.read_csv(complexity)
            expected_x = {"cnot_count", "n_params", "n_layers"}
            expected_y = {"aod_mean", "spd_mean", "eod_mean", "entanglement_score"}
            record_check(set(df["metric_x"]) == expected_x, f"{dataset}: predittori complessita' completi", f"{dataset}: predittori complessita' incompleti", report_lines, error_lines)
            record_check(set(df["metric_y"]) == expected_y, f"{dataset}: outcome complessita' completi", f"{dataset}: outcome complessita' incompleti", report_lines, error_lines)
            record_check(len(df) == 12, f"{dataset}: 12 correlazioni", f"{dataset}: {len(df)} correlazioni invece di 12", report_lines, error_lines)
            record_check(df["correlation"].between(-1, 1).all() and df["p_value"].between(0, 1).all(), f"{dataset}: correlazioni/p-value validi", f"{dataset}: correlazioni/p-value non validi", report_lines, error_lines)


def validate_merge(root, frames, report_lines, error_lines):
    if not all(k in frames for k in ["RQ2", "RQ3"]):
        return
    rq2, rq3 = frames["RQ2"], frames["RQ3"]
    for dataset in DATASET_DIRS:
        sub3 = rq3[rq3["dataset"] == dataset]
        sub2 = rq2[rq2["dataset"] == dataset]
        run_df = sub3[["dataset", "configuration", "seed", "svqx_concentration", "entanglement_score", "qsalto_a1_estimate", "cnot_count", "n_params", "n_layers"]].drop_duplicates(["dataset", "configuration", "seed"])
        rq2_agg = sub2.groupby(["dataset", "configuration", "seed"])[["aod", "spd", "eod"]].mean().reset_index().rename(columns={"aod": "aod_mean", "spd": "spd_mean", "eod": "eod_mean"})
        try:
            merged = run_df.merge(rq2_agg, on=["dataset", "configuration", "seed"], how="left", validate="one_to_one")
            for col in ["aod_mean", "spd_mean", "eod_mean"]:
                record_check(
                    col in merged and not merged[col].isna().any(),
                    f"{dataset}: merge RQ2-RQ3 completo per {col}",
                    f"{dataset}: NaN o colonna mancante dopo merge per {col}",
                    report_lines, error_lines,
                )
        except Exception as exc:
            msg = f"[ERROR] {dataset}: merge RQ2-RQ3 fallito: {exc!r}"
            report_lines.append(msg)
            error_lines.append(msg)


def run(root="Quantum_Dataset_Generation_Online_Appendix"):
    root = Path(root)
    out_dir = root / "ValidationResults"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["=== Validation suite  ===", f"Root: {root}", ""]
    error_lines = []
    frames = validate_raw(root, report_lines, error_lines)
    report_lines.extend(["", "## Risultati statistici"])
    validate_statistical_results(root, report_lines, error_lines)
    report_lines.extend(["", "## Merge RQ2-RQ3"])
    validate_merge(root, frames, report_lines, error_lines)
    report_lines.extend(["", "## Esito finale"])
    if error_lines:
        report_lines.append(f"[FAIL] {len(error_lines)} errori rilevati")
        (out_dir / "validation_errors.txt").write_text("\n".join(error_lines), encoding="utf-8")
    else:
        report_lines.append("[PASS] Nessun errore rilevato")
    (out_dir / "validation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report scritto: {out_dir / 'validation_report.txt'}")
    if error_lines:
        print(f"[FAIL] {len(error_lines)} errori rilevati")
    else:
        print("[PASS] Nessun errore rilevato")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Quantum_Dataset_Generation_Online_Appendix")
    run(**vars(parser.parse_args()))