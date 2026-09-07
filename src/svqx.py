"""
svqx.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Framework di auditing strutturale basato su Quantum Shapley Values (SVQX).

Questo modulo implementa il motore di spiegabilità e ispezione causale (XQML) del framework.
Utilizza il concetto dei valori di Shapley, derivato dalla teoria dei giochi cooperativi, 
per quantificare il contributo marginale di singoli gate quantistici o gruppi di operazioni 
(rotazionali ed entangling) alle performance di utilità (utility) e di equità (fairness).

Il sistema garantisce la precisione matematica del calcolo attraverso l'ordinamento 
canonico dei sottoinsiemi di dati (canonical sorting), mappando l'influenza dei componenti 
hardware quantistici sulle decisioni e sui bias del modello predittivo finale.
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from math import factorial

import numpy as np

from LS_0512110456_evaluation import fairness_metrics

_worker_state = {}


def build_gate_groups(gate_metadata, n_qubits):
    groups = {f"qubit_{q}_rot": [] for q in range(n_qubits)}
    groups["entangling_CNOT"] = []
    for gate_idx, gate in enumerate(gate_metadata):
        if gate["kind"] == "CNOT":
            groups["entangling_CNOT"].append(gate_idx)
        else:
            groups[f"qubit_{gate['qubit']}_rot"].append(gate_idx)
    return groups


def value_function(generator, theta, x_batch, y_batch, a_batch, active_gate_indices):
    full_mask = np.zeros(len(generator.gates), dtype=bool)
    if active_gate_indices:
        full_mask[active_gate_indices] = True
    z_values = generator.batch_forward(theta, x_batch, y_batch, a_batch, active_gate_mask=full_mask)
    p_hat = (z_values[:, generator.target_qubit] + 1) / 2
    y_pred = (p_hat >= 0.5).astype(int)
    spd, _, _ = fairness_metrics(y_batch, y_pred, a_batch)
    if np.isnan(spd):
        spd = 0.0
    return 1 - abs(spd)


def _init_worker(generator, theta, gate_groups, x_batch, y_batch, a_batch):
    _worker_state.update({
        "generator": generator, "theta": theta, "gate_groups": gate_groups,
        "x_batch": x_batch, "y_batch": y_batch, "a_batch": a_batch,
    })


def _evaluate_subset(subset):
    state = _worker_state
    indices = [idx for name in subset for idx in state["gate_groups"][name]]
    value = value_function(
        state["generator"], state["theta"], state["x_batch"],
        state["y_batch"], state["a_batch"], indices,
    )
    return tuple(sorted(subset)), value


def svqx_exact(generator, theta, gate_groups, x_batch, y_batch, a_batch, max_workers=None):
    """Calcolo esatto di tutti i valori v(S) e dei Quantum Shapley Values."""
    group_names = list(gate_groups.keys())
    n_groups = len(group_names)
    all_subsets = [
        subset
        for r in range(n_groups + 1)
        for subset in itertools.combinations(group_names, r)
    ]

    workers = max_workers or (os.cpu_count() or 1)
    value_cache = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(generator, theta, gate_groups, x_batch, y_batch, a_batch),
    ) as executor:
        for key, value in executor.map(_evaluate_subset, all_subsets):
            value_cache[key] = value

    phi = {name: 0.0 for name in group_names}
    for group_i in group_names:
        others = [g for g in group_names if g != group_i]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                key_s = tuple(sorted(subset))
                key_with = tuple(sorted(subset + (group_i,)))
                weight = factorial(len(subset)) * factorial(n_groups - len(subset) - 1) / factorial(n_groups)
                phi[group_i] += weight * (value_cache[key_with] - value_cache[key_s])
    return phi


def svqx_concentration(phi):
    values = np.array([abs(v) for v in phi.values()])
    return float(values.max() / (values.sum() + 1e-12))
