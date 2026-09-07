"""
visualization_suite.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Suite di visualizzazione grafica e reporting per l'ispezione dei risultati QML.

Questo modulo costituisce il motore di rendering grafico del framework. Trasforma i dati 
sperimentali e i risultati dei test statistici (prodotti dal motore inferenziale) in 
visualizzazioni ad alta risoluzione, tra cui heatmap per i test post-hoc di Nemenyi, 
grafici a barre per la stabilità SVQX e scatter plot di complessità architetturale.

La suite automatizza la generazione del catalogo visivo dei risultati, garantendo 
piena compatibilità cross-environment grazie alla rimozione di dipendenze grafiche 
non supportate ed esportando i grafici in una struttura di directory standardizzata.

Compatibile con:
- pipeline_orchestrator.py
- statistical_engine.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_DIRS = {
    "german_credit": "German",
    "heart_disease": "Heart",
    "student_performance": "Student",
}


def plot_nemenyi_heatmap(matrix, out_path, title_suffix=""):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="viridis_r",
        vmin=0, vmax=1, square=True,
        cbar_kws={"label": "p-value (Nemenyi post-hoc)"},
    )
    plt.title(f"Nemenyi post-hoc {title_suffix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_svqx_stability(stability_df, out_path):
    """Bar plot robusto della stabilita' inter-seed SVQX."""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=stability_df,
        x="configuration",
        y="mean_rho",
        palette="viridis",
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    plt.ylim(-1, 1)
    plt.ylabel("Mean pairwise Spearman rho (gate groups)")
    plt.xlabel("Configurazione")
    plt.title("Stabilita' inter-seed del ranking SVQX (gate groups)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_complexity_scatter(run_df, metric_x, metric_y, out_path):
    plot_df = run_df[[metric_x, metric_y]].dropna()
    if len(plot_df) < 2:
        print(f" [SKIP] campione insufficiente per {metric_x} vs {metric_y}")
        return
    plt.figure(figsize=(6, 5))
    sns.regplot(
        data=plot_df,
        x=metric_x,
        y=metric_y,
        scatter_kws={"s": 40, "alpha": 0.7},
        line_kws={"color": "red"},
    )
    plt.xlabel(metric_x.replace("_", " ").title())
    plt.ylabel(metric_y.replace("_", " ").title())
    plt.title(f"Correlazione: {metric_x} vs {metric_y}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def visualize_rq1(root):
    root = Path(root)
    results = root / "StatisticalResults"
    viz = root / "VisualizationResults"
    for dataset, ddir in DATASET_DIRS.items():
        dest = viz / "RQ1" / ddir
        dest.mkdir(parents=True, exist_ok=True)
        for metric in ["f1", "accuracy"]:
            path = results / "RQ1" / ddir / f"{metric}_NemenyiTestResults.csv"
            if not path.exists():
                print(f" [SKIP] {path} non trovato")
                continue
            matrix = pd.read_csv(path, index_col=0)
            out_png = dest / f"{metric}_NemenyiTestResults_heatmap.png"
            plot_nemenyi_heatmap(matrix, out_png, f"RQ1 {dataset} - {metric.upper()}")
            print(f" [OK] {out_png}")


def visualize_rq2(root):
    root = Path(root)
    results = root / "StatisticalResults"
    viz = root / "VisualizationResults"
    for dataset, ddir in DATASET_DIRS.items():
        base_dest = viz / "RQ2" / ddir
        base_dest.mkdir(parents=True, exist_ok=True)
        for metric in ["spd", "eod", "aod"]:
            for attribute in ["age", "sex", "overall"]:
                attr_dest = base_dest / attribute
                attr_dest.mkdir(parents=True, exist_ok=True)
                path = results / "RQ2" / ddir / attribute / f"{metric}_NemenyiTestResults.csv"
                if not path.exists():
                    continue
                matrix = pd.read_csv(path, index_col=0)
                out_png = attr_dest / f"{metric}_NemenyiTestResults_heatmap.png"
                plot_nemenyi_heatmap(matrix, out_png, f"RQ2 {dataset} {attribute} - {metric.upper()}")
                print(f" [OK] {out_png}")


def load_run_df(rq3, rq2, dataset):
    sub = rq3[rq3["dataset"] == dataset].copy()
    run_cols = ["dataset", "configuration", "seed"]
    complexity_cols = [c for c in ["cnot_count", "n_params", "n_layers"] if c in sub.columns]
    run_metrics = ["svqx_concentration", "entanglement_score", "qsalto_a1_estimate"]
    required = run_cols + complexity_cols + run_metrics
    missing = [c for c in run_cols + run_metrics if c not in sub.columns]
    if missing:
        raise ValueError(f"{dataset}: colonne RQ3 mancanti: {missing}")
    run_df = sub[required].drop_duplicates(run_cols)

    rq2_sub = rq2[rq2["dataset"] == dataset].copy()
    required_rq2 = ["dataset", "configuration", "seed", "aod", "spd", "eod"]
    missing = [c for c in required_rq2 if c not in rq2_sub.columns]
    if missing:
        raise ValueError(f"{dataset}: colonne RQ2 mancanti: {missing}")
    rq2_agg = (
        rq2_sub.groupby(["dataset", "configuration", "seed"])[["aod", "spd", "eod"]]
        .mean()
        .reset_index()
        .rename(columns={"aod": "aod_mean", "spd": "spd_mean", "eod": "eod_mean"})
    )
    return run_df.merge(
        rq2_agg,
        on=["dataset", "configuration", "seed"],
        how="left",
        validate="one_to_one",
    )


def visualize_rq3(root):
    root = Path(root)
    results = root / "StatisticalResults"
    viz = root / "VisualizationResults"
    analysis = root / "Analysis"

    rq3_path = analysis / "RQ3_data" / "rq3_all_results.csv"
    rq2_path = analysis / "RQ2_data" / "rq2_all_fairness_results_multiattribute.csv"
    if not (rq3_path.exists() and rq2_path.exists()):
        print(" [SKIP] raw RQ2/RQ3 non trovati")
        return
    rq3 = pd.read_csv(rq3_path)
    rq2 = pd.read_csv(rq2_path)

    for dataset, ddir in DATASET_DIRS.items():
        dest = viz / "RQ3" / ddir
        dest.mkdir(parents=True, exist_ok=True)

        nemenyi_path = results / "RQ3" / ddir / "gate" / "svqx_value_NemenyiTestResults.csv"
        if nemenyi_path.exists():
            matrix = pd.read_csv(nemenyi_path, index_col=0)
            out_png = dest / "svqx_NemenyiTestResults_heatmap.png"
            plot_nemenyi_heatmap(matrix, out_png, f"RQ3 {dataset} (gate-level)")
            print(f" [OK] {out_png}")
        else:
            print(f" [SKIP] {nemenyi_path} non trovato")

        stability_path = results / "RQ3" / ddir / "gate" / "svqx_stability_spearman.csv"
        if stability_path.exists():
            stability_df = pd.read_csv(stability_path)
            if not stability_df.empty:
                out_png = dest / "svqx_stability_bars.png"
                plot_svqx_stability(stability_df, out_png)
                print(f" [OK] {out_png}")
        else:
            print(f" [SKIP] {stability_path} non trovato")

        run_df = load_run_df(rq3, rq2, dataset)
        fairness_cols = ["aod_mean", "spd_mean", "eod_mean"]
        corr_cols = ["cnot_count", "n_params", "n_layers"] + fairness_cols + ["entanglement_score"]
        missing = [c for c in corr_cols if c not in run_df.columns]
        if missing:
            print(f" [WARN] {dataset}: colonne mancanti per scatter: {missing}")
            continue
        complete_df = run_df[corr_cols].dropna()
        if complete_df.empty:
            print(f" [WARN] {dataset}: nessuna riga completa per scatter")
            continue
        for metric_y in fairness_cols + ["entanglement_score"]:
            for metric_x in ["cnot_count", "n_params", "n_layers"]:
                out_png = dest / f"complexity_scatter_{metric_x}_{metric_y}.png"
                plot_complexity_scatter(complete_df, metric_x, metric_y, out_png)
                print(f" [OK] {out_png}")


def run(root="Quantum_Dataset_Generation_Online_Appendix"):
    root = Path(root)
    viz = root / "VisualizationResults"
    viz.mkdir(parents=True, exist_ok=True)
    print("=== Visualization suite ===")
    visualize_rq1(root)
    visualize_rq2(root)
    visualize_rq3(root)
    print(f"\nVisualizzazioni completate: {viz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Quantum_Dataset_Generation_Online_Appendix")
    run(**vars(parser.parse_args()))