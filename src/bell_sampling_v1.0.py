"""
bell_sampling.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Framework diagnostico quantistico per la misurazione della fairness nativa.

Questo modulo implementa metriche di equità quantistiche native (quantum-native fairness metrics)
e protocolli di diagnostica dello stato quantistico per valutare le disparità demografiche .
Invece di affidarsi esclusivamente a classificatori classici downstream, il sistema ispeziona
lo stato quantistico misto emerso dal Variational Quantum Circuit (VQC) per verificare come i
diversi gruppi demografici vengano mappati all'interno dello spazio di Hilbert.

Il modulo adotta il formalismo degli operatori di densità e calcola la distanza di traccia (Trace Distance)
tra stati quantistici misti ridotti per quantificare la distinguibilità geometrica ex-ante dei gruppi.
Inoltre, implementa il protocollo di Bell sampling a due copie per stimare la sovrapposizione delle tracce
(trace overlap) senza richiedere una tomografia quantistica di stato completa, riducendo drasticamente
l'overhead di misurazione su hardware NISQ.

Metrica di fairness quantum-native per RQ2, basata sul protocollo di Bell sampling
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

PARALLEL_THRESHOLD = 40  # numero minimo di campioni nel gruppo per attivare il worker pool


class BellSamplingError(RuntimeError):
    """
    Eccezione bloccante sollevata quando il calcolo della metrica di
    fairness quantum-native fallisce per QUALUNQUE motivo. Nessun
    fallback silenzioso su NaN
    """


@dataclass(frozen=True)
class BellSamplingResult:
    """Esito di una stima di sovrapposizione/distanza via Bell sampling."""

    trace_overlap: float
    trace_distance_upper_bound: float
    n_shots: int | None



def build_bell_sampling_circuit(theta_a: float, phi_a: float, theta_b: float, phi_b: float) -> QuantumCircuit:
    """Circuito di Bell sampling per due stati puri a un qubit, parametrizzati dagli angoli di Bloch."""
    qc = QuantumCircuit(2, name="bell_sampling")
    qc.u(theta_a, phi_a, 0, 0)
    qc.u(theta_b, phi_b, 0, 1)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.measure_all()
    return qc


def estimate_trace_overlap_via_shots(
    theta_a: float, phi_a: float, theta_b: float, phi_b: float, n_shots: int = 8192,
) -> BellSamplingResult:
    """Stima Tr[rho_a rho_b] eseguendo il circuito di Bell sampling su un simulatore a shot."""
    try:
        from qiskit_aer import AerSimulator

        qc = build_bell_sampling_circuit(theta_a, phi_a, theta_b, phi_b)
        simulator = AerSimulator()
        job = simulator.run(qc, shots=n_shots)
        counts = job.result().get_counts()

        total = sum(counts.values())
        p_11 = counts.get("11", 0) / total
        trace_overlap = 1 - 2 * p_11
    except Exception as exc:  # noqa: BLE001
        raise BellSamplingError(
            f"Stima via shot del Bell sampling fallita (theta_a={theta_a}, phi_a={phi_a}, "
            f"theta_b={theta_b}, phi_b={phi_b}, n_shots={n_shots}): {type(exc).__name__}: {exc}"
        ) from exc
    return BellSamplingResult(trace_overlap=trace_overlap, trace_distance_upper_bound=float("nan"),
                               n_shots=n_shots)


_worker_state: dict = {}


def _init_density_worker(generator, X: np.ndarray, Y: np.ndarray, A: np.ndarray, qubit_index: int) -> None:
    _worker_state["generator"] = generator
    _worker_state["X"] = X
    _worker_state["Y"] = Y
    _worker_state["A"] = A
    _worker_state["qubit_index"] = qubit_index


def _compute_single_reduced_density(i: int) -> tuple[int, np.ndarray]:
    """Calcola la matrice densità ridotta per UN campione"""
    generator = _worker_state["generator"]
    X, Y, A = _worker_state["X"], _worker_state["Y"], _worker_state["A"]
    qubit_index = _worker_state["qubit_index"]
    n_qubits = generator.n_qubits
    indices_to_trace_out = [q for q in range(n_qubits) if q != qubit_index]

    x_full = generator._encode_input(X[i], Y[i], A[i])
    bound_circuit = generator.bound_full_circuit(x_full)
    full_state = Statevector.from_instruction(bound_circuit)
    reduced = partial_trace(full_state, indices_to_trace_out)
    return i, reduced.data


def reduced_density_matrix_for_group(
    vqc_generator, X: np.ndarray, Y: np.ndarray, A: np.ndarray, group_value: int, qubit_index: int,
    max_workers: int | None = None,
) -> DensityMatrix:

    mask = A == group_value
    if not np.any(mask):
        raise BellSamplingError(
            f"Nessun campione con attributo sensibile A={group_value} nel batch fornito "
            f"(batch size={len(A)}). Impossibile costruire l'operatore di densità del gruppo."
        )

    sample_indices = np.where(mask)[0]  # ordine crescente deterministico
    n_qubits = vqc_generator.n_qubits
    accumulated = np.zeros((2, 2), dtype=complex)

    if len(sample_indices) >= PARALLEL_THRESHOLD:
        resolved_max_workers = max_workers or (os.cpu_count() or 1)
        results: dict[int, np.ndarray] = {}
        try:
            with ProcessPoolExecutor(
                max_workers=resolved_max_workers, initializer=_init_density_worker,
                initargs=(vqc_generator, X, Y, A, qubit_index),
            ) as executor:
                for i, data in executor.map(_compute_single_reduced_density, sample_indices):
                    results[i] = data
        except Exception as exc:  # noqa: BLE001
            raise BellSamplingError(
                f"Calcolo parallelo delle matrici densità ridotte fallito per il gruppo A={group_value}, "
                f"qubit={qubit_index}: {type(exc).__name__}: {exc}"
            ) from exc

        for i in sample_indices:  # stesso ordine del ciclo sequenziale -> somma bit-identica
            accumulated += results[i]
    else:
        for i in sample_indices:
            try:
                x_full = vqc_generator._encode_input(X[i], Y[i], A[i])
                bound_circuit = vqc_generator.bound_full_circuit(x_full)
                full_state = Statevector.from_instruction(bound_circuit)
                reduced = partial_trace(full_state, [q for q in range(n_qubits) if q != qubit_index])
            except Exception as exc:  # noqa: BLE001
                raise BellSamplingError(
                    f"Costruzione dell'operatore di densità ridotto fallita per il campione indice {i} "
                    f"(gruppo A={group_value}, qubit={qubit_index}): {type(exc).__name__}: {exc}"
                ) from exc
            accumulated += reduced.data

    accumulated /= mask.sum()
    return DensityMatrix(accumulated)


def trace_overlap(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    try:
        return float(np.real(np.trace(rho.data @ sigma.data)))
    except Exception as exc:  # noqa: BLE001
        raise BellSamplingError(f"Calcolo di Tr[rho*sigma] fallito: {type(exc).__name__}: {exc}") from exc


def trace_distance(rho: DensityMatrix, sigma: DensityMatrix) -> float:
    try:
        diff = rho.data - sigma.data
        eigenvalues = np.linalg.eigvalsh(diff)
        return float(0.5 * np.sum(np.abs(eigenvalues)))
    except Exception as exc:  # noqa: BLE001
        raise BellSamplingError(
            f"Calcolo della distanza di traccia fallito (diagonalizzazione non riuscita): "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def quantum_statistical_parity_diff(
    vqc_generator, X: np.ndarray, Y: np.ndarray, A: np.ndarray, qubit_index: int | None = None,
) -> float:
    """QSP-diff = |Tr(Z rho1) - Tr(Z rho0)|"""
    if qubit_index is None:
        qubit_index = vqc_generator.target_qubit

    rho_1 = reduced_density_matrix_for_group(vqc_generator, X, Y, A, group_value=1, qubit_index=qubit_index)
    rho_0 = reduced_density_matrix_for_group(vqc_generator, X, Y, A, group_value=0, qubit_index=qubit_index)

    try:
        z_observable = np.array([[1, 0], [0, -1]], dtype=complex)
        exp_1 = float(np.real(np.trace(z_observable @ rho_1.data)))
        exp_0 = float(np.real(np.trace(z_observable @ rho_0.data)))
        return abs(exp_1 - exp_0)
    except Exception as exc:  # noqa: BLE001
        raise BellSamplingError(
            f"Calcolo di QSP-diff fallito sul qubit {qubit_index}: {type(exc).__name__}: {exc}"
        ) from exc


def bias_pair_diagnostic(
    vqc_generator, X: np.ndarray, Y: np.ndarray, A: np.ndarray, qubit_index: int | None = None,
) -> dict:
    """Verifica operativa della condizione di bias-pair di Guan et al."""
    if qubit_index is None:
        qubit_index = vqc_generator.target_qubit

    rho_1 = reduced_density_matrix_for_group(vqc_generator, X, Y, A, group_value=1, qubit_index=qubit_index)
    rho_0 = reduced_density_matrix_for_group(vqc_generator, X, Y, A, group_value=0, qubit_index=qubit_index)

    d_tr = trace_distance(rho_1, rho_0)
    qsp_diff = quantum_statistical_parity_diff(vqc_generator, X, Y, A, qubit_index)

    return {
        "trace_distance_input_states": d_tr,
        "qsp_diff_output": qsp_diff,
        "is_bias_pair_candidate": bool(d_tr < 0.1 and qsp_diff > 0.1),
    }
