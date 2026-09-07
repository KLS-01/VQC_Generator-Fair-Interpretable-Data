"""
quantum_vqc.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Questo modulo definisce le architetture di circuiti quantistici parametrizzati
utilizzate per generare dataset tabulari sintetici equi e interpretabili.
Mappa le feature classiche nello spazio quantistico di Hilbert e ottimizza
i parametri sotto vincoli multi-obiettivo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector

EncodingStrategy = Literal["single", "re_upload"]
EntanglementTopology = Literal["linear", "all_to_all", "none"]


def _build_z_bit_matrix(n_qubits: int) -> np.ndarray:
    indices = np.arange(2 ** n_qubits)
    return ((indices[:, None] >> np.arange(n_qubits)) & 1).astype(np.float64)


@dataclass
class VQCCircuitTemplate:
    n_qubits: int
    n_layers: int
    entangle: EntanglementTopology = "linear"
    encoding_strategy: EncodingStrategy = "single"

    circuit: QuantumCircuit = field(init=False)
    x_params: ParameterVector = field(init=False)
    theta_params: ParameterVector = field(init=False)
    gate_metadata: list[dict] = field(init=False)

    def __post_init__(self) -> None:
        self.x_params = ParameterVector("x", self.n_qubits)
        self.theta_params = ParameterVector("theta", self.n_layers * self.n_qubits * 2)
        self.circuit, self.gate_metadata = self._build()

    def _build(self) -> tuple[QuantumCircuit, list[dict]]:
        qc = QuantumCircuit(self.n_qubits, name="VQC_generator")
        metadata: list[dict] = []

        for q in range(self.n_qubits):
            qc.ry(self.x_params[q], q)
            metadata.append({"kind": "Ry", "qubit": q, "role": "encoding", "layer": 0})

        theta_idx = 0
        for layer in range(self.n_layers):
            if self.encoding_strategy == "re_upload" and layer > 0:
                for q in range(self.n_qubits):
                    qc.ry(self.x_params[q], q)
                    metadata.append({"kind": "Ry", "qubit": q, "role": "encoding", "layer": layer})

            for q in range(self.n_qubits):
                qc.ry(self.theta_params[theta_idx], q)
                metadata.append({"kind": "Ry", "qubit": q, "role": "ansatz", "layer": layer + 1,
                                  "param_idx": theta_idx})
                theta_idx += 1
                qc.rz(self.theta_params[theta_idx], q)
                metadata.append({"kind": "Rz", "qubit": q, "role": "ansatz", "layer": layer + 1,
                                  "param_idx": theta_idx})
                theta_idx += 1

            if self.entangle == "linear":
                for q in range(self.n_qubits - 1):
                    qc.cx(q, q + 1)
                    metadata.append({"kind": "CNOT", "control": q, "target": q + 1,
                                      "role": "ansatz", "layer": layer + 1})
            elif self.entangle == "all_to_all":
                for q1 in range(self.n_qubits):
                    for q2 in range(q1 + 1, self.n_qubits):
                        qc.cx(q1, q2)
                        metadata.append({"kind": "CNOT", "control": q1, "target": q2,
                                          "role": "ansatz", "layer": layer + 1})
            elif self.entangle != "none":
                raise ValueError(f"Topologia di entanglement non supportata: {self.entangle}")

        return qc, metadata

    @property
    def n_params(self) -> int:
        return len(self.theta_params)

    def bind(self, x_values: np.ndarray, theta_values: np.ndarray) -> QuantumCircuit:
        mapping = dict(zip(self.x_params, x_values)) | dict(zip(self.theta_params, theta_values))
        return self.circuit.assign_parameters(mapping)


class VQCGenerator:
    """Incapsula S2+S3+S4 e l'ottimizzazione vincolata alla fairness (Eq. 3.5-3.8)."""

    def __init__(
        self,
        d_features: int,
        n_layers: int = 2,
        entangle: EntanglementTopology = "linear",
        encoding_strategy: EncodingStrategy = "single",
        seed: int = 0,
    ) -> None:
        self.d = d_features
        self.n_qubits = d_features + 2
        self.target_qubit = d_features
        self.sensitive_qubit = d_features + 1

        self.template = VQCCircuitTemplate(
            n_qubits=self.n_qubits, n_layers=n_layers, entangle=entangle,
            encoding_strategy=encoding_strategy,
        )
        self.n_params = self.template.n_params
        self.gates = self.template.gate_metadata

        # --- REQUISITO ESPLICITO: RNG locale deterministico ---
        self._rng = np.random.RandomState(seed)
        self.theta = self._rng.uniform(-np.pi, np.pi, self.n_params)

        self._full_circuit_cache: QuantumCircuit = self.template.circuit
        self._restricted_circuit_cache: dict[tuple[bool, ...], QuantumCircuit] = {}
        self._bit_matrix = _build_z_bit_matrix(self.n_qubits)

    def _encode_input(self, x_feats: np.ndarray, y_bit: float, a_bit: float) -> np.ndarray:
        x_full = np.zeros(self.n_qubits)
        x_full[: self.d] = x_feats
        x_full[self.target_qubit] = y_bit * np.pi
        x_full[self.sensitive_qubit] = a_bit * np.pi
        return x_full

    def _bind_filtered(self, circuit: QuantumCircuit, x_values: np.ndarray, theta_values: np.ndarray) -> QuantumCircuit:
        present_params = set(circuit.parameters)
        full_mapping = dict(zip(self.template.x_params, x_values))
        full_mapping.update(dict(zip(self.template.theta_params, theta_values)))
        filtered_mapping = {p: v for p, v in full_mapping.items() if p in present_params}
        return circuit.assign_parameters(filtered_mapping)

    def _circuit_for_mask(self, active_gate_mask: np.ndarray | None) -> QuantumCircuit:
        if active_gate_mask is None:
            return self._full_circuit_cache

        mask_key = tuple(bool(v) for v in active_gate_mask)
        cached = self._restricted_circuit_cache.get(mask_key)
        if cached is not None:
            return cached

        qc = QuantumCircuit(self.n_qubits, name="VQC_generator_restricted")
        for gi, gate_meta in enumerate(self.gates):
            if not active_gate_mask[gi]:
                continue
            if gate_meta["kind"] == "Ry":
                param = (self.template.x_params[gate_meta["qubit"]] if gate_meta["role"] == "encoding"
                         else self.template.theta_params[gate_meta["param_idx"]])
                qc.ry(param, gate_meta["qubit"])
            elif gate_meta["kind"] == "Rz":
                qc.rz(self.template.theta_params[gate_meta["param_idx"]], gate_meta["qubit"])
            elif gate_meta["kind"] == "CNOT":
                qc.cx(gate_meta["control"], gate_meta["target"])

        self._restricted_circuit_cache[mask_key] = qc
        return qc

    def _bind_batch(
        self, circuit: QuantumCircuit, X: np.ndarray, Y: np.ndarray, A: np.ndarray, theta: np.ndarray,
    ) -> list[QuantumCircuit]:
        return [self._bind_filtered(circuit, self._encode_input(X[i], Y[i], A[i]), theta) for i in range(len(X))]

    def _estimate_bound_circuits(self, circuits: list[QuantumCircuit]) -> np.ndarray:
        if not circuits:
            return np.empty((0, self.n_qubits))
        results = np.empty((len(circuits), self.n_qubits))
        for idx, circuit in enumerate(circuits):
            probs = Statevector.from_instruction(circuit).probabilities()
            results[idx] = 1.0 - 2.0 * (probs @ self._bit_matrix)
        return results

    def batch_forward(
        self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray, A: np.ndarray,
        active_gate_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        circuit = self._circuit_for_mask(active_gate_mask)
        return self._estimate_bound_circuits(self._bind_batch(circuit, X, Y, A, theta))

    @staticmethod
    def _loss_from_Z(
        Z: np.ndarray, X: np.ndarray, A: np.ndarray, d: int, target_qubit: int, lam_f: float,
    ) -> tuple[float, float, float]:
        L_fid = float(np.mean((Z[:, :d] - np.cos(X)) ** 2))
        p_hat = (Z[:, target_qubit] + 1) / 2
        g1, g0 = p_hat[A == 1], p_hat[A == 0]
        spd_proxy = (g1.mean() if len(g1) else 0.0) - (g0.mean() if len(g0) else 0.0)
        L_fair = float(spd_proxy ** 2)
        return L_fid + lam_f * L_fair, L_fid, L_fair

    def loss_total(
        self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray, A: np.ndarray,
        lam_f: float = 1.0, active_gate_mask: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        Z = self.batch_forward(theta, X, Y, A, active_gate_mask)
        return self._loss_from_Z(Z, X, A, self.d, self.target_qubit, lam_f)

    def parameter_shift_gradient(
        self, theta: np.ndarray, X: np.ndarray, Y: np.ndarray, A: np.ndarray,
        lam_f: float = 1.0, batch_frac: float = 0.35,
    ) -> np.ndarray:
        """Eq. (3.6): Parameter-Shift Rule. Campionamento del mini-batch tramite RNG locale deterministico."""
        if batch_frac < 1.0:
            # --- REQUISITO ESPLICITO: campionamento del mini-batch tramite RNG locale ---
            idx = self._rng.choice(
                len(X),
                size=max(1, int(len(X) * batch_frac)),
                replace=False
            )
            Xb, Yb, Ab = X[idx], Y[idx], A[idx]
        else:
            Xb, Yb, Ab = X, Y, A

        all_circuits: list[QuantumCircuit] = []
        for k in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[k] += np.pi / 2
            theta_minus = theta.copy()
            theta_minus[k] -= np.pi / 2
            all_circuits.extend(self._bind_batch(self.template.circuit, Xb, Yb, Ab, theta_plus))
            all_circuits.extend(self._bind_batch(self.template.circuit, Xb, Yb, Ab, theta_minus))

        Z_all = self._estimate_bound_circuits(all_circuits)
        batch_size = len(Xb)

        grad = np.zeros(len(theta))
        for k in range(len(theta)):
            start_plus = 2 * k * batch_size
            start_minus = start_plus + batch_size
            Z_plus = Z_all[start_plus: start_plus + batch_size]
            Z_minus = Z_all[start_minus: start_minus + batch_size]
            loss_plus, _, _ = self._loss_from_Z(Z_plus, Xb, Ab, self.d, self.target_qubit, lam_f)
            loss_minus, _, _ = self._loss_from_Z(Z_minus, Xb, Ab, self.d, self.target_qubit, lam_f)
            grad[k] = 0.5 * (loss_plus - loss_minus)
        return grad

    def fit(
        self, X: np.ndarray, Y: np.ndarray, A: np.ndarray,
        lam_f: float = 5.0, epochs: int = 15, lr: float = 0.4, batch_frac: float = 0.35,
        verbose: bool = False,
    ) -> list[tuple[float, float, float]]:
        history: list[tuple[float, float, float]] = []
        for epoch in range(epochs):
            grad = self.parameter_shift_gradient(self.theta, X, Y, A, lam_f, batch_frac)
            self.theta -= lr * grad
            loss_epoch = self.loss_total(self.theta, X, Y, A, lam_f)
            history.append(loss_epoch)
            if verbose:
                l_tot, l_fid, l_fair = loss_epoch
                print(f"epoch {epoch:03d}  L_tot={l_tot:.5f}  L_fid={l_fid:.5f}  L_fair={l_fair:.6f}")
        return history

    def reconstruct_dataset(
        self, X: np.ndarray, Y: np.ndarray, A: np.ndarray, scale_params: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = self.batch_forward(self.theta, X, Y, A)
        x_min, x_max = scale_params
        X_synth = ((Z[:, : self.d] + 1) / 2) * (x_max - x_min) + x_min
        Y_synth = ((Z[:, self.target_qubit] + 1) / 2 >= 0.5).astype(int)
        A_synth = ((Z[:, self.sensitive_qubit] + 1) / 2 >= 0.5).astype(int)
        return X_synth, Y_synth, A_synth

    def bound_full_circuit(self, x_values: np.ndarray) -> QuantumCircuit:
        return self._bind_filtered(self._full_circuit_cache, x_values, self.theta)
