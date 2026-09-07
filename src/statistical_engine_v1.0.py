"""
statistical_engine.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Motore di analisi statistica inferenziale per la valutazione di esperimenti QML.

Questo modulo costituisce il motore statistico del framework. Esegue l'analisi rigorosa
delle metriche di utilità, equità (fairness) e complessità computazionale relative ai
VQC, implementando test non parametrici in conformità con le linee guida di Demšar.

Il sistema automatizza il flusso di validazione statistica, partendo dal test di normalità 
fino al calcolo dei test omnibus e post-hoc (Friedman e Nemenyi), esportando matrici di 
correlazione e tabelle diagnostiche pronte per la pubblicazione scientifica.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, shapiro, norm, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05
DATASET_DIRS = {
    "german_credit": "German",
    "heart_disease": "Heart",
    "student_performance": "Student",
}


def safe_shapiro(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or len(np.unique(x)) <= 1:
        return np.nan, np.nan
    s, p = shapiro(x)
    return float(s), float(p)


def mean_std(df, groups, metrics, path):
    metrics = [m for m in metrics if m in df.columns]
    df.groupby(groups)[metrics].agg(["mean", "std"]).to_csv(path)


def shapiro_table(df, groups, metrics, path):
    """
    Solleva ValueError se una metrica richiesta manca o e' interamente NaN/inf.
    """

    missing = []
    all_nan = []
    for m in metrics:
        if m not in df.columns:
            missing.append(m)
        elif df[m].apply(lambda x: np.isfinite(x)).sum() == 0:
            all_nan.append(m)
    if missing:
        raise ValueError(
            f"shapiro_table: metriche richieste mancanti nel DataFrame: {missing}. "
            f"Colonne disponibili: {list(df.columns)}"
        )
    if all_nan:
        raise ValueError(
            f"shapiro_table: metriche richieste con tutti i valori NaN/inf: {all_nan}. "
            f"Verifica i dati grezzi prima di procedere."
        )

    rows = []
    for keys, g in df.groupby(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys))
        for metric in metrics:
            n = int(g[metric].notna().sum())
            stat, p = safe_shapiro(g[metric].dropna().to_numpy())
            rows.append({**base, "metric": metric, "n": n,
                         "statistic": stat, "p_value": p,
                         "note": "descriptive only; n<8" if n < 8 else "diagnostic"})
    pd.DataFrame(rows).to_csv(path, index=False)


def aligned_pivot(df, metric, index_cols):
    return df.pivot_table(index=index_cols, columns="configuration",
                          values=metric, aggfunc="mean").dropna()


def friedman_table(df, metrics, index_cols):
    """
    Test di Friedman omnibus, un test per (dataset, metrica), con blocchi
    definiti da index_cols. Riceve in input un sottoinsieme di df
    gia' filtrato alla grana statistica corretta.
    """
    rows = []
    for dataset, g in df.groupby("dataset"):
        for metric in metrics:
            pvt = aligned_pivot(g, metric, index_cols)
            if pvt.shape[0] >= 2 and pvt.shape[1] >= 3:
                test = friedmanchisquare(*[pvt[c] for c in pvt.columns])
                rows.append({"dataset": dataset, "metric": metric,
                             "statistic": float(test.statistic),
                             "p_value_raw": float(test.pvalue),
                             "n_blocks": int(len(pvt)),
                             "n_configurations": int(pvt.shape[1]),
                             "unit": "+".join(index_cols)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_value_holm"] = multipletests(out.p_value_raw, method="holm")[1]
    out["p_value_fdr_bh"] = multipletests(out.p_value_raw, method="fdr_bh")[1]
    out["significant_raw"] = out.p_value_raw < ALPHA
    out["significant_holm"] = out.p_value_holm < ALPHA
    out["significant_fdr_bh"] = out.p_value_fdr_bh < ALPHA
    return out


def nemenyi_matrix(df, metric, index_cols):
    pvt = aligned_pivot(df, metric, index_cols)
    if pvt.shape[0] < 2 or pvt.shape[1] < 3:
        return pd.DataFrame()
    cols = list(pvt.columns)
    ranks = pvt.rank(axis=1, method="average")
    avg = ranks.mean(axis=0).to_numpy(float)
    k, n = len(cols), len(ranks)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    mat = np.ones((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            z = abs(avg[i] - avg[j]) / se
            p = min(1.0, float(2.0 * norm.sf(z)))
            mat[i, j] = mat[j, i] = p
    return pd.DataFrame(mat, index=cols, columns=cols)


def write_posthoc(df, metrics, out_dir, index_cols):
    for metric in metrics:
        matrix = nemenyi_matrix(df, metric, index_cols)
        if not matrix.empty:
            matrix.to_csv(out_dir / f"{metric}_NemenyiTestResults.csv")


def process_rq1(df, root):
    metrics = ["f1", "accuracy"]
    for dataset, sub in df.groupby("dataset"):
        dest = root / "RQ1" / DATASET_DIRS[dataset]
        dest.mkdir(parents=True, exist_ok=True)
        groups = ["dataset", "configuration", "model"]
        mean_std(sub, groups, metrics, dest / "rq1_mean_std.csv")
        shapiro_table(sub, groups, metrics, dest / "shapiro.csv")
        friedman_table(sub, metrics, ["seed", "model"]).to_csv(dest / "friedman.csv", index=False)
        write_posthoc(sub, metrics, dest, ["seed", "model"])


def process_rq2(df, root):
    metrics = ["spd", "eod", "aod", "trace_distance_input", "qsp_diff_output"]
    has_attribute = "attribute" in df.columns
    groups = ["dataset", "configuration", "model", "attribute"] if has_attribute \
        else ["dataset", "configuration", "model"]

    for dataset, sub in df.groupby("dataset"):
        dest = root / "RQ2" / DATASET_DIRS[dataset]
        dest.mkdir(parents=True, exist_ok=True)
        mean_std(sub, groups, metrics, dest / "rq2_mean_std.csv")
        shapiro_table(sub, groups, metrics, dest / "shapiro.csv")

        if has_attribute:
            per_attribute_friedman = []
            for attribute, sub_attr in sub.groupby("attribute"):
                attr_dest = dest / attribute
                attr_dest.mkdir(parents=True, exist_ok=True)

                friedman_attr = friedman_table(sub_attr, metrics, ["seed", "model"])
                if not friedman_attr.empty:
                    friedman_attr.insert(0, "attribute", attribute)
                    friedman_attr.to_csv(attr_dest / "friedman.csv", index=False)
                    per_attribute_friedman.append(friedman_attr)

                write_posthoc(sub_attr, metrics, attr_dest, ["seed", "model"])

            if per_attribute_friedman:
                combined = pd.concat(
                    [f for f in per_attribute_friedman if not f.empty],
                    ignore_index=True,
                )
            else:
                combined = pd.DataFrame()
            combined.to_csv(dest / "friedman.csv", index=False)
        else:
            friedman_table(sub, metrics, ["seed", "model"]).to_csv(dest / "friedman.csv", index=False)
            write_posthoc(sub, metrics, dest, ["seed", "model"])


def process_rq3(df, root, rq2_df):
    gate_metrics = ["svqx_value"]
    run_metrics = ["svqx_concentration", "entanglement_score", "qsalto_a1_estimate"]

    for dataset, sub in df.groupby("dataset"):
        base = root / "RQ3" / DATASET_DIRS[dataset]
        gate = base / "gate"
        run = base / "run"
        gate.mkdir(parents=True, exist_ok=True)
        run.mkdir(parents=True, exist_ok=True)


        mean_std(sub, ["dataset", "configuration", "gate_group"], gate_metrics,
                 gate / "rq3_gate_mean_std.csv")
        shapiro_table(sub, ["dataset", "configuration", "gate_group"], gate_metrics,
                      gate / "rq3_gate_shapiro.csv")
        friedman_table(sub, gate_metrics, ["seed", "gate_group"]).to_csv(gate / "friedman.csv", index=False)
        write_posthoc(sub, gate_metrics, gate, ["seed", "gate_group"])

        # Stabilita' inter-seed SVQX: per ogni configurazione, pivot
        # gate_group x seed, poi Spearman tra le colonne dei seed.
        stability_rows = []
        for config, cfg_sub in sub.groupby("configuration"):
            pvt = cfg_sub.pivot(index="gate_group", columns="seed", values="svqx_value")
            if pvt.shape[1] == 3:  # 3 seed
                seeds = sorted(pvt.columns)
                rho_7_17, p_7_17 = spearmanr(pvt[seeds[0]], pvt[seeds[1]])
                rho_7_27, p_7_27 = spearmanr(pvt[seeds[0]], pvt[seeds[2]])
                rho_17_27, p_17_27 = spearmanr(pvt[seeds[1]], pvt[seeds[2]])
                mean_rho = np.mean([rho_7_17, rho_7_27, rho_17_27])
                stability_rows.append({
                    "dataset": dataset,
                    "configuration": config,
                    "spearman_rho_7_17": rho_7_17,
                    "spearman_p_7_17": p_7_17,
                    "spearman_rho_7_27": rho_7_27,
                    "spearman_p_7_27": p_7_27,
                    "spearman_rho_17_27": rho_17_27,
                    "spearman_p_17_27": p_17_27,
                    "mean_rho": mean_rho,
                })
        if stability_rows:
            pd.DataFrame(stability_rows).to_csv(gate / "svqx_stability_spearman.csv", index=False)

        run_cols = ["dataset", "configuration", "seed"]
        complexity_cols = [c for c in ["cnot_count", "n_params", "n_layers"] if c in sub.columns]
        run_df = sub[run_cols + complexity_cols + run_metrics].drop_duplicates(run_cols)

        rq2_sub = rq2_df[rq2_df["dataset"] == dataset].copy()
        rq2_agg = rq2_sub.groupby(["dataset", "configuration", "seed"])[["aod", "spd", "eod"]].mean().reset_index()
        rq2_agg = rq2_agg.rename(columns={"aod": "aod_mean", "spd": "spd_mean", "eod": "eod_mean"})
        run_df = run_df.merge(rq2_agg, on=["dataset", "configuration", "seed"], how="left")

        cols_for_corr = ["cnot_count", "n_params", "n_layers", "aod_mean", "spd_mean", "eod_mean", "entanglement_score"]
        cols_available = [c for c in cols_for_corr if c in run_df.columns]
        run_df_clean = run_df[cols_available].dropna()
        print(f"  [{dataset}] run_df pre-dropna: {len(run_df)} righe, "
              f"post-dropna: {len(run_df_clean)} righe "
              f"(colonne per correlazioni: {cols_available})")

        mean_std(run_df, ["dataset", "configuration"], run_metrics,
                 run / "rq3_run_mean_std.csv")
        shapiro_table(run_df, ["dataset", "configuration"], run_metrics,
                      run / "rq3_run_shapiro.csv")
        friedman_table(run_df, run_metrics, ["seed"]).to_csv(run / "friedman.csv", index=False)
        write_posthoc(run_df, run_metrics, run, ["seed"])

        complexity_rows = []
        metric_y_list = ["aod_mean", "spd_mean", "eod_mean", "entanglement_score"]
        for metric_y in metric_y_list:
            if metric_y not in run_df_clean.columns:
                continue
            for metric_x in ["cnot_count", "n_params", "n_layers"]:
                if metric_x not in run_df_clean.columns:
                    continue
                x = run_df_clean[metric_x].to_numpy()
                y = run_df_clean[metric_y].to_numpy()
                _, px = safe_shapiro(x)
                _, py = safe_shapiro(y)
                if np.isnan(px) or np.isnan(py) or px < 0.05 or py < 0.05:
                    corr, p = spearmanr(x, y)
                    method = "spearman"
                else:
                    corr, p = pearsonr(x, y)
                    method = "pearson"
                complexity_rows.append({
                    "dataset": dataset,
                    "metric_x": metric_x,
                    "metric_y": metric_y,
                    "correlation": corr,
                    "p_value": p,
                    "method": method,
                })
        if complexity_rows:
            pd.DataFrame(complexity_rows).to_csv(run / "complexity_correlations.csv", index=False)
        else:
            print(f"  [WARN] {dataset}: complexity_correlations.csv non generato "
                  f"(colonne disponibili in run_df_clean: {list(run_df_clean.columns)})")


def run(root="Quantum_Dataset_Generation_Online_Appendix"):
    root = Path(root)
    analysis = root / "Analysis"
    out = root / "StatisticalResults"
    out.mkdir(parents=True, exist_ok=True)

    rq1_path = analysis / "RQ1_data" / "rq1_models_results.csv"
    rq2_path = analysis / "RQ2_data" / "rq2_all_fairness_results_multiattribute.csv"
    rq3_path = analysis / "RQ3_data" / "rq3_all_results.csv"

    for label, path in [("RQ1", rq1_path), ("RQ2", rq2_path), ("RQ3", rq3_path)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label}: file raw non trovato: {path}. "
                f"Verifica di aver eseguito pipeline_orchestrator.py."
            )

    rq1 = pd.read_csv(rq1_path)
    rq2 = pd.read_csv(rq2_path)
    rq3 = pd.read_csv(rq3_path)

    
    
    # -------------------------------------------------------------------------
    # ELABORAZIONE RQ1
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("ELABORAZIONE RESEARCH QUESTION 1: Downstream Utility (TSTR)")
    print("-" * 80)
    print("[PROCESS] Calcolo medie aritmetiche e deviazioni standard globali...")
    print("[PROCESS] Esecuzione test diagnostico di normalità di Shapiro-Wilk...")
    print("[PROCESS] Test Omnibus di Friedman su 9 blocchi (seed x model)...")
    print("[PROCESS] Calcolo delle matrici post-hoc pairwise di Nemenyi per F1 ed Accuracy...")
    process_rq1(rq1, out)
    
    print("[OK] Statistiche RQ1 salvate in: StatisticalResults/RQ1/")
    
    # -------------------------------------------------------------------------
    # ELABORAZIONE RQ2
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("ELABORAZIONE RESEARCH QUESTION 2: Downstream Fairness & Quantum Diagnostics")
    print("-" * 80)
    print("[PROCESS] Analisi disaggregata per attributi sensibili: [age, sex, overall]...")
    print("[PROCESS] Calcolo medie e test di Shapiro-Wilk per metriche classiche e quantistiche...")
    print("[PROCESS] Test di Friedman ed estrazione matrici post-hoc Nemenyi per ciascun attributo...")
    process_rq2(rq2, out)
    
    print("[OK] Statistiche RQ2 salvate in: StatisticalResults/RQ2/")
    
    # -------------------------------------------------------------------------
    # ELABORAZIONE RQ3
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("ELABORAZIONE RESEARCH QUESTION 3: Explainability (SVQX), Entanglement & Complessità")
    print("-" * 80)
    print("[PROCESS] Ispezione causale dei Quantum Shapley Values (SVQX) a livello di gate...")
    print("[PROCESS] Calcolo stabilità inter-seed dei ranking SVQX (Spearman rho)...")
    print("[PROCESS] Integrazione delle metriche fisiche globali (Purezza Q-SALTO, entanglement)...")
    print("[PROCESS] Calcolo delle 12 correlazioni di complessità architetturale (3x4)...")
    process_rq3(rq3, out, rq2)
    
    print("[OK] Statistiche RQ3 salvate in: StatisticalResults/RQ3/")
    
    # Banner di Chiusura
    print("\n" + "=" * 80)
    print("                ELABORAZIONE STATISTICA COMPLETATA CON SUCCESSO                 ")
    print("=" * 80)
    print(f"[PASS] Tutti i report CSV sono stati correttamente compilati in StatisticalResults:")
    print("       - RQ1/  --> Tabelle di utility, Shapiro-Wilk e test post-hoc Nemenyi")
    print("       - RQ2/  --> Tabelle di fairness e diagnostica quantistica disaggregate")
    print("       - RQ3/  --> Tabelle SVQX, stabilità, purezza Q-SALTO e 12 correlazioni")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Quantum_Dataset_Generation_Online_Appendix")
    run(**vars(parser.parse_args()))