"""
entanglement_analysis.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Modulo di analisi e quantificazione dell'entanglement nei circuiti quantistici.

Questo modulo fornisce gli strumenti matematici e diagnostici per quantificare l'entanglement 
globale e locale all'interno dei Circuiti Quantistici Variazionali (VQC). Utilizza la libreria 
'qsalto' per estrarre metriche di correlazione quantistica, consentendo di studiare la relazione 
tra la complessità fisica dell'ansatz e l'equità (fairness) dei dati generati.

Il sistema implementa protocolli avanzati quali le misure di Bell a due copie (two-copy Bell 
measurements) e controlli di purezza diagnostici, estraendo accuratamente il coefficiente di 
Shor-Laflamme per mappare l'architettura fisica delle correlazioni quantistiche.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace


class EntanglementAnalysisError(RuntimeError):
    """
    Eccezione bloccante sollevata quando qsalto non è disponibile
    o non si comporta in modo desiderabile
    """


@dataclass(frozen=True)
class EntanglementReport:
    """Esito della quantificazione di entanglement per una configurazione VQC (RQ3)."""

    method: str  # sempre "qsalto" in questa versione (nessun fallback automatico)
    average_subsystem_purity: float  # calcolata comunque come riferimento diagnostico (Eq. 2.x, Sez. 2.4.1)
    entanglement_score: float  # 1 - purezza media, normalizzato in [0, 1]
    n_qubits: int
    qsalto_a1_estimate: float  # coefficiente Shor-Laflamme a_1 estratto da qsalto (indice [1])


def build_two_copy_bell_circuit(n_qubits: int) -> QuantumCircuit:
    """
    Costruisce il circuito a 2n qubit per la misura di Bell a due copie
    per ciascuna coppia (j, j+n) si applica     CNOT + Hadamard e si misura.
    """
    qc = QuantumCircuit(2 * n_qubits, name="two_copy_bell_sampling")
    for j in range(n_qubits):
        qc.cx(j, j + n_qubits)
        qc.h(j)
    qc.measure_all()
    return qc


def run_two_copy_bell_measurement(
    state_preparation_circuit: QuantumCircuit, n_qubits: int, n_shots: int = 4096,
) -> np.ndarray:
    """Esegue il protocollo a due copie; ritorna l'array dei conteggi di singoletti m per ogni shot."""
    from qiskit_aer import AerSimulator

    full_circuit = QuantumCircuit(2 * n_qubits)
    full_circuit.compose(state_preparation_circuit, qubits=range(n_qubits), inplace=True)
    full_circuit.compose(state_preparation_circuit, qubits=range(n_qubits, 2 * n_qubits), inplace=True)
    full_circuit.compose(build_two_copy_bell_circuit(n_qubits), inplace=True)

    simulator = AerSimulator()
    job = simulator.run(full_circuit, shots=n_shots)
    counts = job.result().get_counts()

    m_values = []
    for bitstring, count in counts.items():
        clean_bits = bitstring.replace(" ", "")
        n_ones_in_bell_outcomes = sum(
            1 for j in range(n_qubits) if clean_bits[-(j + 1)] == "1" and clean_bits[-(j + n_qubits + 1)] == "1"
        )
        m_values.extend([n_ones_in_bell_outcomes] * count)
    return np.array(m_values)


def classical_average_subsystem_purity(statevector: Statevector, n_qubits: int) -> float:
    """
    Tr[rho_j^2] mediata su tutti i sottosistemi a singolo qubit j
    """
    purities = []
    for j in range(n_qubits):
        traced_out = [q for q in range(n_qubits) if q != j]
        rho_j = partial_trace(statevector, traced_out)
        purities.append(float(np.real(np.trace(rho_j.data @ rho_j.data))))
    return float(np.mean(purities))


def analyze_entanglement(vqc_generator, sample_x: np.ndarray, n_shots: int = 4096) -> EntanglementReport:
    """
    Quantifica l'entanglement del circuito VQC per UN singolo campione
    di input, usando ESCLUSIVAMENTE qsalto.
    Se qsalto non è disponibile o l'estrazione dello scalare fallisce,
    solleva l'eccezione 'EntanglementAnalysisError'
    """
    n_qubits = vqc_generator.n_qubits
    bound_circuit = vqc_generator.bound_full_circuit(sample_x)
    statevector = Statevector.from_instruction(bound_circuit)
    classical_purity = classical_average_subsystem_purity(statevector, n_qubits)

    try:
        import qsalto

        m_samples = run_two_copy_bell_measurement(bound_circuit, n_qubits, n_shots)
        m_distribution = np.bincount(m_samples, minlength=n_qubits + 1) / len(m_samples)

        estimators = qsalto.single_shot_estimators(n_qubits)
        a_enumerator_weights = np.asarray(estimators[0])


        estimates_vector = np.dot(a_enumerator_weights, m_distribution)
        a_estimate = float(estimates_vector[1])
    except Exception as exc:  # noqa: BLE001
        raise EntanglementAnalysisError(
            f"qsalto.single_shot_estimators({n_qubits}) non ha prodotto una stima valida "
            f"({type(exc).__name__}: {exc}). Verifica 'pip install qsalto'"
        ) from exc

    return EntanglementReport(
        method="qsalto", average_subsystem_purity=classical_purity,
        entanglement_score=1.0 - classical_purity, n_qubits=n_qubits, qsalto_a1_estimate=a_estimate,
    )
