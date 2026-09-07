"""
pipeline_orchestrator.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Orchestratore della pipeline sperimentale per la generazione e valutazione VQC.

Questo modulo costituisce il punto d'ingresso e il coordinatore (orchestrator) dell'intero
framework sperimentale. Gestisce l'esecuzione sistematica dei test sui tre dataset di
riferimento (German Credit, Heart Disease, Student Performance) attraverso le diverse
configurazioni di ansatz quantistici, profondità di layer e seed di addestramento.

Il sistema coordina il flusso sequenziale che va dalla preparazione dei dati, all'addestramento
dei modelli quantistici variazionali, fino all'esportazione dei dataset generati e alle relative
valutazioni downstream di utilità e fairness.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Quantum_Dataset_Generation_Online_Appendix"
ANALYSIS = OUTPUT / "Analysis"
STATE_PATH = OUTPUT / "run_manifest.json"

DATASETS = ["german_credit", "heart_disease", "student_performance"]
SEEDS = [7, 17, 27]
BATCH_FRAC = 0.25
EPOCHS = 15
LEARNING_RATE = 0.4
LAMBDA_FAIR = 5.0

CONFIGS = [
    {"name": "L1_linear", "n_layers": 1, "entangle": "linear", "encoding_strategy": "single"},
    {"name": "L2_linear", "n_layers": 2, "entangle": "linear", "encoding_strategy": "single"},
    {"name": "L1_all_to_all", "n_layers": 1, "entangle": "all_to_all", "encoding_strategy": "single"},
    {"name": "L2_all_to_all", "n_layers": 2, "entangle": "all_to_all", "encoding_strategy": "single"},
    {"name": "L2_re_upload", "n_layers": 2, "entangle": "linear", "encoding_strategy": "re_upload"},
    {"name": "L2_all_to_all_re_upload", "n_layers": 2, "entangle": "all_to_all", "encoding_strategy": "re_upload"},
]


def load_module(alias, filename):
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Modulo richiesto non trovato: {path}")
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


data_pipeline = load_module("data_pipeline", "data_pipeline.py")
dataset_export = load_module("dataset_export", "dataset_export.py")
evaluation = load_module("evaluation", "evaluation.py")
sys.modules["evaluation"] = evaluation
quantum_vqc = load_module("quantum_vqc", "quantum_vqc.py")
bell_sampling = load_module("bell_sampling", "bell_sampling.py")
entanglement = load_module("entanglement", "entanglement_analysis.py")
svqx = load_module("svqx", "svqx.py")


def load_manifest():
    if STATE_PATH.exists():
        with STATE_PATH.open() as f:
            return json.load(f)
    return {"version": "1.4", "batch_frac": BATCH_FRAC, "completed": {}, "failed": {}}


def save_manifest(manifest):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temp = STATE_PATH.with_suffix(".tmp")
    with temp.open("w") as f:
        json.dump(manifest, f, indent=2)
    temp.replace(STATE_PATH)


def run_key(dataset, config, seed):
    return f"{dataset}|{config}|seed{seed}"


def load_existing_run_keys(rq1_path: Path) -> set[str]:
    """
    Usato per rendere il resume idempotente anche se il manifest 
    non e' stato salvato per una run che aveva già scritto le sue
    righe sui CSV (crash tra append e save).
    """
    if not rq1_path.exists():
        return set()
    df = pd.read_csv(rq1_path, usecols=["dataset", "configuration", "seed"])
    df = df.drop_duplicates()
    return {
        run_key(row.dataset, row.configuration, int(row.seed))
        for row in df.itertuples(index=False)
    }


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def count_cnot_gates(gates: list[dict]) -> int:
    return sum(1 for g in gates if g["kind"] == "CNOT")


def run_single(bundle, x_scaled, scale_params, config, seed):
    idx = np.arange(len(x_scaled))
    idx_train, idx_test = train_test_split(
        idx, test_size=.30, stratify=bundle.A_overall, random_state=seed
    )

    x_train = x_scaled[idx_train]
    y_train = bundle.Y[idx_train]
    a_train = bundle.A_overall[idx_train]

    x_test = bundle.X[idx_test]
    y_test = bundle.Y[idx_test]
    a_overall_test = bundle.A_overall[idx_test]
    a_age_test = bundle.A_age[idx_test]
    a_sex_test = bundle.A_sex[idx_test]

    generator = quantum_vqc.VQCGenerator(
        d_features=x_train.shape[1],
        n_layers=config["n_layers"],
        entangle=config["entangle"],
        encoding_strategy=config["encoding_strategy"],
        seed=seed,
    )

    structural_metadata = {
        "n_layers": config["n_layers"],
        "entangle": config["entangle"],
        "encoding_strategy": config["encoding_strategy"],
        "n_qubits": generator.n_qubits,
        "n_params": generator.n_params,
        "cnot_count": count_cnot_gates(generator.gates),
    }

    history = generator.fit(
        x_train, y_train, a_train, lam_f=LAMBDA_FAIR,
        epochs=EPOCHS, lr=LEARNING_RATE,
        batch_frac=BATCH_FRAC, verbose=False,
    )
    x_synth, y_synth, a_synth = generator.reconstruct_dataset(
        x_train, y_train, a_train, scale_params
    )

    attributes_test = {
        "age": a_age_test,
        "sex": a_sex_test,
        "overall": a_overall_test,
    }
    multi = evaluation.rq1_rq2_evaluation_multi_attribute(
        x_synth, y_synth, x_test, y_test, attributes_test,
        seed=seed, light=True,
    )

    qfair = bell_sampling.bias_pair_diagnostic(generator, x_train, y_train, a_train)
    groups = svqx.build_gate_groups(generator.gates, generator.n_qubits)
    n = min(15, len(x_train))
    phi = svqx.svqx_exact(
        generator, generator.theta, groups,
        x_train[:n], y_train[:n], a_train[:n],
    )
    concentration = svqx.svqx_concentration(phi)
    encoded = generator._encode_input(
        x_train[0], float(y_train[0]), float(a_train[0])
    )
    ereport = entanglement.analyze_entanglement(generator, encoded, n_shots=2048)

    rq1, rq2 = [], []
    for model_name, payload in multi.items():
        utility = payload["utility"]
        rq1.append({
            "dataset": bundle.name,
            "configuration": config["name"],
            "seed": seed,
            "model": model_name,
            "batch_frac": BATCH_FRAC,
            **structural_metadata,
            "f1": utility["F1"],
            "accuracy": utility["Accuracy"],
            "final_loss": history[-1][0],
            "best_loss": min(h[0] for h in history),
        })
        for fairness_row in payload["fairness"]:
            rq2.append({
                "dataset": bundle.name,
                "configuration": config["name"],
                "seed": seed,
                "model": model_name,
                "batch_frac": BATCH_FRAC,
                **structural_metadata,
                "attribute": fairness_row["attribute"],
                "spd": fairness_row["spd"],
                "eod": fairness_row["eod"],
                "aod": fairness_row["aod"],
                "trace_distance_input": qfair["trace_distance_input_states"],
                "qsp_diff_output": qfair["qsp_diff_output"],
                "is_bias_pair_candidate": qfair["is_bias_pair_candidate"],
            })

    rq3 = [{
        "dataset": bundle.name,
        "configuration": config["name"],
        "seed": seed,
        "batch_frac": BATCH_FRAC,
        **structural_metadata,
        "gate_group": g,
        "svqx_value": v,
        "svqx_concentration": concentration,
        "entanglement_score": ereport.entanglement_score,
        "qsalto_a1_estimate": ereport.qsalto_a1_estimate,
    } for g, v in phi.items()]

    dataset_export.save_synthetic_dataset(
        x_synth, y_synth, a_synth, bundle.feature_names, bundle.target_name,
        bundle.sensitive_names["overall"], bundle.name, config["name"], seed,
        output_dir=str(OUTPUT / "Datasets"),
    )

    return rq1, rq2, rq3


def main():
    manifest = load_manifest()
    if manifest.get("batch_frac") != BATCH_FRAC:
        raise RuntimeError(
            "Manifest esistente con batch_frac diverso: archiviarlo prima di procedere."
        )

    rq1_path = ANALYSIS / "RQ1_data" / "rq1_models_results.csv"
    rq2_path = ANALYSIS / "RQ2_data" / "rq2_all_fairness_results_multiattribute.csv"
    rq3_path = ANALYSIS / "RQ3_data" / "rq3_all_results.csv"
    start = time.perf_counter()

    existing_keys = load_existing_run_keys(rq1_path)
    healed = 0
    for key in existing_keys:
        if key not in manifest["completed"]:
            manifest["completed"][key] = {
                "status": "completed_self_healed",
                "timestamp": time.time(),
                "note": "Chiave presente nei CSV raw ma assente dal manifest "
                        "(probabile crash tra append e save_manifest in una run "
                        "precedente). Marcata completata per evitare duplicazione.",
            }
            healed += 1
    if healed:
        print(f"[SELF-HEAL] {healed} chiavi presenti nei CSV ma assenti dal manifest: "
              f"marcate come completate per evitare ri-esecuzione e duplicazione.")
        save_manifest(manifest)

    for dataset_name in DATASETS:
        bundle = data_pipeline.load_dataset(dataset_name)
        x_scaled, scale_params = data_pipeline.scale_to_angle(bundle.X)
        dataset_export.save_real_dataset_reference(
            bundle.X, bundle.Y, bundle.A_overall, bundle.feature_names,
            bundle.target_name, bundle.sensitive_names["overall"], dataset_name,
            output_dir=str(OUTPUT / "Datasets"),
        )

        for config in CONFIGS:
            for seed in SEEDS:
                key = run_key(dataset_name, config["name"], seed)
                if key in manifest["completed"]:
                    print(f"[SKIP] {key} già completata")
                    continue
                print(f"[START] {key}")
                try:
                    rq1, rq2, rq3 = run_single(
                        bundle, x_scaled, scale_params, config, seed
                    )
                    append_rows(rq1_path, rq1)
                    append_rows(rq2_path, rq2)
                    append_rows(rq3_path, rq3)
                    manifest["completed"][key] = {
                        "status": "completed", "timestamp": time.time()
                    }
                    manifest["failed"].pop(key, None)
                    save_manifest(manifest)
                    print(f"[DONE] {key}")
                except Exception as exc:
                    manifest["failed"][key] = {
                        "status": "failed", "error": repr(exc),
                        "timestamp": time.time(),
                    }
                    save_manifest(manifest)
                    raise

    n_total = len(DATASETS) * len(CONFIGS) * len(SEEDS)
    print(f"Pipeline completata/resumata in {time.perf_counter() - start:.1f}s")
    print(f"Manifest: {STATE_PATH}")
    print(f"CSV: {rq1_path}, {rq2_path}, {rq3_path}")
    print(
        f"Griglia: {len(DATASETS)} dataset x {len(CONFIGS)} configurazioni "
        f"x {len(SEEDS)} seed = {n_total} run totali"
    )


if __name__ == "__main__":
    main()