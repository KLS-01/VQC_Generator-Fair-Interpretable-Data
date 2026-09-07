"""
evaluation.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Modulo di valutazione multi-attributo per utilità downstream ed equità algoritmica.

Questo modulo definisce la suite di valutazione del framework. Bilancia la misurazione
dell'utilità predittiva (predictive utility) dei dataset sintetici generati dai VQC con
un rigoroso audit di responsabilità sociale ed equità (fairness) multi-attributo.

Il sistema addestra molteplici classificatori downstream (tra cui Random Forest e Reti 
Neurali a complessità architetturale variabile) secondo il protocollo "Train on Synthetic, 
Test on Real" (TSTR). Successivamente, analizza le prestazioni di utilità classica (accuratezza, 
F1-score) e quantifica l'impatto disparato (disparate impact) e le deviazioni di equità 
disaggregate per attributi sensibili quali età, sesso e le loro combinazioni intersezionali.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score


def get_downstream_models(seed=0, light=False):
    if light:
        return {
            "RandomForest": RandomForestClassifier(n_estimators=60, random_state=seed),
            "XGBoost_stub": GradientBoostingClassifier(n_estimators=40, random_state=seed),
            "MLP": MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=200, random_state=seed),
        }
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "XGBoost_stub": GradientBoostingClassifier(random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=seed),
    }


def fairness_metrics(y_true, y_pred, a):
    y_true, y_pred, a = np.asarray(y_true), np.asarray(y_pred), np.asarray(a)

    def p1(mask):
        return float(y_pred[mask].mean()) if mask.sum() > 0 else np.nan

    spd = p1(a == 1) - p1(a == 0)
    pos1, pos0 = (a == 1) & (y_true == 1), (a == 0) & (y_true == 1)
    eod = p1(pos1) - p1(pos0)
    neg1, neg0 = (a == 1) & (y_true == 0), (a == 0) & (y_true == 0)
    aod = 0.5 * ((p1(neg1) - p1(neg0)) + eod)
    return spd, eod, aod


def fairness_metrics_by_attribute(y_true, y_pred, attributes: dict):
    """attributes: {"age": A_age_test, "sex": A_sex_test, "overall": A_overall_test}."""
    rows = []
    for attribute_name, a in attributes.items():
        if a is None:
            continue
        if len(np.unique(a)) < 2:
            raise ValueError(
                f"Attributo sensibile '{attribute_name}' non ha due gruppi distinti "
                "nel test set: impossibile calcolare SPD/EOD/AOD senza fallback."
            )
        spd, eod, aod = fairness_metrics(y_true, y_pred, a)
        rows.append({"attribute": attribute_name, "spd": spd, "eod": eod, "aod": aod})
    return rows


def rq1_rq2_evaluation_multi_attribute(
    X_train, Y_train, X_test_real, Y_test_real,
    attributes_test: dict, seed=0, light=False, label="",
):
    """
    Esegue il fit una sola volta per modello, valuta l'utility una volta e la fairness per singolo attributo.

    attributes_test: {"age": A_age_test, "sex": A_sex_test, "overall": A_overall_test}
    Returns: dict[model_name] -> {"utility": {...}, "fairness": [rows]}
    """
    models = get_downstream_models(seed, light=light)
    results = {}
    for name, clf in models.items():
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test_real)
        f1 = f1_score(Y_test_real, y_pred, zero_division=0)
        acc = accuracy_score(Y_test_real, y_pred)
        fairness_rows = fairness_metrics_by_attribute(Y_test_real, y_pred, attributes_test)
        results[name] = {
            "utility": {"F1": f1, "Accuracy": acc},
            "fairness": fairness_rows,
        }
        if label:
            print(f"--- {label} / {name} ---")
            print(f"  F1={f1:.4f} Accuracy={acc:.4f}")
            for row in fairness_rows:
                print(f"  attribute={row['attribute']} SPD={row['spd']:.4f} "
                      f"EOD={row['eod']:.4f} AOD={row['aod']:.4f}")
    return results


def rq1_rq2_evaluation(X_train, Y_train, X_test_real, Y_test_real, A_test_real,
                        seed=0, light=False, label=""):
    """Legacy: evaluation a singolo attributo"""
    models = get_downstream_models(seed, light=light)
    results = {}
    for name, clf in models.items():
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test_real)
        f1 = f1_score(Y_test_real, y_pred, zero_division=0)
        acc = accuracy_score(Y_test_real, y_pred)
        spd, eod, aod = fairness_metrics(Y_test_real, y_pred, A_test_real)
        results[name] = dict(F1=f1, Accuracy=acc, SPD=spd, EOD=eod, AOD=aod)
        if label:
            print(f"--- {label} ---")
            for k, v in results.items():
                print(k, {m: round(val, 4) for m, val in v.items()})
    return results
