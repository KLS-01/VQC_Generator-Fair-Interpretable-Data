"""
dataset_export.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Modulo di esportazione e decodifica dei dataset sintetici generati da VQC.

Questo modulo gestisce la trasformazione finale dei dati sintetici generati dal calcolatore 
quantistico in formati classici strutturati e direttamente ispezionabili. Converte le 
rappresentazioni numeriche grezze e normalizzate derivanti dallo spazio di Hilbert quantistico, 
invertendo il preprocessing per ripristinare le unità di misura originarie, i nomi delle feature 
e i valori categoriali reali dei dataset di riferimento.

Il sistema funge da componente chiave per l'ispezione empirica esterna, integrando funzioni di 
analisi statistica distributiva e summary comparativi che permettono di validare l'integrità 
delle variabili sintetiche rispetto a quelle reali.


Convenzione di naming:
    <dataset>_generated_<configurazione>_<seed>.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def save_synthetic_dataset(
    x_synth: np.ndarray,
    y_synth: np.ndarray,
    a_synth: np.ndarray,
    feature_names: list[str],
    target_name: str,
    sensitive_name: str,
    dataset_name: str,
    configuration_label: str,
    seed: int,
    output_dir: str = "output/Datasets",
) -> str:
    """
    Salva un singolo dataset sintetico su disco
    Ritorna il path completo del file salvato.
    """
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    df = pd.DataFrame(x_synth, columns=feature_names)
    df[target_name] = y_synth
    df[sensitive_name] = a_synth

    filename = f"{dataset_name}_generated_{configuration_label}_seed{seed}.csv"
    filepath = os.path.join(dataset_output_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath


def save_real_dataset_reference(
    x_real: np.ndarray,
    y_real: np.ndarray,
    a_real: np.ndarray,
    feature_names: list[str],
    target_name: str,
    sensitive_name: str,
    dataset_name: str,
    output_dir: str = "output/Datasets",
) -> str:
    """
    Salva il dataset reale usato come riferimento, nello
    stesso formato dei sintetici, necessario per confronti diretti
    riga-per-riga o per rigenerare le figure di confronto distribuzionale
    """
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    df = pd.DataFrame(x_real, columns=feature_names)
    df[target_name] = y_real
    df[sensitive_name] = a_real

    filepath = os.path.join(dataset_output_dir, f"{dataset_name}_original.csv")
    df.to_csv(filepath, index=False)
    return filepath


def summarize_column_distribution(
    x_real: np.ndarray, x_synth: np.ndarray, feature_names: list[str],
) -> pd.DataFrame:
    """
    Confronto statistico sintetico vs. reale per ciascuna colonna
    (min, media, mediana, max, std)
    """
    rows = []
    for j, name in enumerate(feature_names):
        rows.append(dict(
            feature=name,
            real_min=x_real[:, j].min(), real_mean=x_real[:, j].mean(),
            real_median=np.median(x_real[:, j]), real_max=x_real[:, j].max(),
            real_std=x_real[:, j].std(),
            synth_min=x_synth[:, j].min(), synth_mean=x_synth[:, j].mean(),
            synth_median=np.median(x_synth[:, j]), synth_max=x_synth[:, j].max(),
            synth_std=x_synth[:, j].std(),
        ))
    return pd.DataFrame(rows)
