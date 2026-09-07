"""
data_pipeline.py
==============================================
Studente: Leonardo Schiavo
Matricola: 0512110456

Pipeline di preprocessing e feature selection per l'allineamento dei dati QML.

Questo modulo gestisce il flusso di ingegneria dei dati (data pipeline) del framework.
Si occupa del caricamento, della pulizia, della normalizzazione e della codifica dei
dataset di riferimento (German Credit, Heart Disease, Student Performance), garantendo
la consistenza dei gruppi demografici attraverso la standardizzazione delle soglie degli
attributi protetti (es. l'attributo sensibile dell'età).

Il modulo implementa criteri avanzati di selezione delle feature basati sull'Informazione
Mutua (Mutual Information), riducendo la dimensionalità dello spazio delle feature classiche
per renderlo compatibile con i vincoli fisici e il footprint di qubit dei VQC. Inoltre,
assicura un allineamento strutturale tra la funzione di perdita (loss function) del modello
e le metriche di valutazione downstream.


A_sex     : attributo sensibile di genere
A_age     : attributo sensibile di età (soglie documentate sotto)
A_overall : attributo intersectional:
            privileged_groups = [{'sex':1, 'age':1} => A_overall = (A_sex == 1) & (A_age == 1)

Soglie di età
    german_credit        : age >= 25
    heart_disease        : age >= 55
    student_performance  : age >= 18

A_overall e' l'attributo passato al training del VQC
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif

logger = logging.getLogger("thesis_vqc.data_pipeline")

DATASET_UCI_IDS: dict[str, int] = {
    "heart_disease": 45,
    "german_credit": 144,
    "student_performance": 320,
}

AGE_THRESHOLDS: dict[str, float] = {
    "german_credit": 25.0,
    "heart_disease": 55.0,
    "student_performance": 18.0,
}

FEATURE_SELECTION_METHOD: str = "mutual_info"
MAX_FEATURES_FOR_QUANTUM: int = 8


class DatasetLoadError(RuntimeError):
    """Eccezione bloccante: nessun fallback automatico su dati fittizi."""


@dataclass(frozen=True)
class DatasetBundle:
    X: np.ndarray
    Y: np.ndarray
    A_overall: np.ndarray
    A_age: np.ndarray
    A_sex: np.ndarray
    name: str
    feature_names: list[str]
    target_name: str
    sensitive_names: Dict[str, str]
    selection_method: str
    selected_original_indices: np.ndarray | None
    n_features_original: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_features_original", self.X.shape[1])

    @property
    def A(self) -> np.ndarray:
        """Attributo usato per addestrare il VQC: l'intersectional A_overall."""
        return self.A_overall


def generate_synthetic_proxy_dataset(
    n_samples: int = 300, d_features: int = 8, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SOLO per smoke-test di sviluppo -- mai invocato automaticamente da load_dataset."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, d_features))
    A = rng.binomial(1, 0.5, size=n_samples)
    logits = X.sum(axis=1) + 1.5 * A - 1.0
    probs = 1 / (1 + np.exp(-logits))
    Y = rng.binomial(1, probs)
    return X, Y, A


def _validate_two_groups(a: np.ndarray, dataset_name: str, attribute_name: str) -> None:
    if len(np.unique(a)) < 2:
        raise DatasetLoadError(
            f"Dataset '{dataset_name}': attributo '{attribute_name}' non ha due gruppi "
            "distinti. Nessun fallback: verificare la soglia o la codifica sorgente."
        )


def _load_heart_disease_raw():
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=DATASET_UCI_IDS["heart_disease"])
    df = data.data.features.copy()
    df["target"] = data.data.targets.iloc[:, 0]
    df = df.dropna()

    feature_cols = ["age", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "slope"]
    X = df[feature_cols].to_numpy(dtype=float)
    Y = (df["target"].to_numpy(dtype=int) > 0).astype(int)

    A_sex = df["sex"].to_numpy(dtype=int)
    A_age = (df["age"].to_numpy(dtype=float) >= AGE_THRESHOLDS["heart_disease"]).astype(int)
    A_overall = ((A_sex == 1) & (A_age == 1)).astype(int)

    _validate_two_groups(A_sex, "heart_disease", "sex")
    _validate_two_groups(A_age, "heart_disease", "age")
    _validate_two_groups(A_overall, "heart_disease", "overall")

    return X, Y, A_overall, A_age, A_sex, feature_cols, "heart_disease_present"


def _load_german_credit_raw():
    """
    Attribute9 (personal_status) codifica genere+stato civile con stringhe
    categoriche: viene usato esclusivamente per derivare A_sex 
    male_codes = {A91, A93, A94}).
    """
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=DATASET_UCI_IDS["german_credit"])
    df = data.data.features.copy()
    df["target"] = data.data.targets.iloc[:, 0]

    numeric_map = {
        "Attribute2": "duration", "Attribute5": "credit_amount", "Attribute13": "age",
        "Attribute8": "installment_rate", "Attribute11": "residence_since",
        "Attribute16": "existing_credits", "Attribute18": "num_dependents",
    }
    missing = [c for c in numeric_map if c not in df.columns]
    if missing:
        raise DatasetLoadError(
            f"Colonne attese non trovate in German Credit: {missing}. "
            f"Colonne disponibili: {list(df.columns)}."
        )
    df = df.rename(columns=numeric_map)

    feature_cols = ["duration", "credit_amount", "installment_rate", "residence_since",
                     "age", "existing_credits", "num_dependents"]

    male_codes = {"A91", "A93", "A94"}
    A_sex = df["Attribute9"].isin(male_codes).astype(int).to_numpy()
    A_age = (df["age"].to_numpy(dtype=float) >= AGE_THRESHOLDS["german_credit"]).astype(int)
    A_overall = ((A_sex == 1) & (A_age == 1)).astype(int)

    _validate_two_groups(A_sex, "german_credit", "sex")
    _validate_two_groups(A_age, "german_credit", "age")
    _validate_two_groups(A_overall, "german_credit", "overall")

    X = df[feature_cols].to_numpy(dtype=float)
    Y = (df["target"].to_numpy(dtype=int) == 1).astype(int)

    return X, Y, A_overall, A_age, A_sex, feature_cols, "credit_risk_good"


def _load_student_performance_raw():
    """
    Target: G3 >= 10
    """
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=DATASET_UCI_IDS["student_performance"])
    features_df = data.data.features.copy()
    targets_df = data.data.targets.copy()

    missing_targets = [c for c in ("G1", "G2", "G3") if c not in targets_df.columns]
    if missing_targets:
        raise DatasetLoadError(
            f"Colonne target attese non trovate in Student Performance: {missing_targets}. "
            f"Colonne disponibili in data.targets: {list(targets_df.columns)}."
        )

    df = features_df.join(targets_df[["G1", "G2", "G3"]])

    feature_cols = ["age", "studytime", "failures", "absences", "G1", "G2", "goout", "health"]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise DatasetLoadError(
            f"Colonne feature attese non trovate in Student Performance dopo il join: "
            f"{missing_features}. Colonne disponibili: {list(df.columns)}."
        )

    X = df[feature_cols].to_numpy(dtype=float)
    Y = (df["G3"].to_numpy(dtype=float) >= 10).astype(int)

    A_sex = (df["sex"].astype(str).str.upper() == "M").astype(int).to_numpy()
    A_age = (df["age"].to_numpy(dtype=float) >= AGE_THRESHOLDS["student_performance"]).astype(int)
    A_overall = ((A_sex == 1) & (A_age == 1)).astype(int)

    _validate_two_groups(A_sex, "student_performance", "sex")
    _validate_two_groups(A_age, "student_performance", "age")
    _validate_two_groups(A_overall, "student_performance", "overall")

    return X, Y, A_overall, A_age, A_sex, feature_cols, "pass_fail"


_RAW_LOADERS: dict[str, Callable[[], tuple]] = {
    "heart_disease": _load_heart_disease_raw,
    "german_credit": _load_german_credit_raw,
    "student_performance": _load_student_performance_raw,
}


def _select_features_mutual_information(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], max_features: int,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    if X.shape[1] <= max_features:
        logger.info("Nessuna riduzione necessaria: %d feature <= %d.", X.shape[1], max_features)
        return X, feature_names, None

    selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
    X_reduced = selector.fit_transform(X, y)
    selected_idx = selector.get_support(indices=True)
    selected_names = [feature_names[i] for i in selected_idx]
    logger.info("Mutual Information: selezionate %d/%d feature: %s",
                max_features, X.shape[1], selected_names)
    return X_reduced, selected_names, selected_idx


def load_dataset(name: str) -> DatasetBundle:
    if name not in _RAW_LOADERS:
        raise ValueError(f"Dataset '{name}' non configurato. Opzioni: {list(_RAW_LOADERS.keys())}")

    try:
        X, Y, A_overall, A_age, A_sex, feature_names, target_name = _RAW_LOADERS[name]()
    except DatasetLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatasetLoadError(
            f"Impossibile caricare il dataset REALE '{name}' da UCI ({type(exc).__name__}: {exc}). "
            f"Verifica: (1) di aver eseguito 'pip install ucimlrepo'; (2) di avere connessione a "
            f"internet (necessaria su Colab); (3) che l'ID UCI del dataset sia ancora valido. "
        ) from exc

    X_reduced, selected_names, selected_idx = _select_features_mutual_information(
        X, Y, feature_names, MAX_FEATURES_FOR_QUANTUM
    )

    logger.info("Dataset reale '%s' caricato e pre-processato (%d istanze, %d feature finali).",
                name, X_reduced.shape[0], X_reduced.shape[1])

    return DatasetBundle(
        X=X_reduced, Y=Y, A_overall=A_overall, A_age=A_age, A_sex=A_sex,
        name=name, feature_names=selected_names, target_name=target_name,
        sensitive_names={"overall": "overall", "age": "age", "sex": "sex"},
        selection_method=FEATURE_SELECTION_METHOD,
        selected_original_indices=selected_idx,
    )


def scale_to_angle(X: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    X_scaled = np.pi * (X - x_min) / (x_max - x_min + 1e-12)
    return X_scaled, (x_min, x_max)


def stratified_split(
    X: np.ndarray, Y: np.ndarray, A: np.ndarray,
    test_size: float = 0.15, val_size: float = 0.15, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    strat = Y.astype(str) + "_" + A.astype(str)
    idx_all = np.arange(len(X))

    idx_train_val, idx_test = train_test_split(idx_all, test_size=test_size, stratify=strat, random_state=seed)
    strat_tv = strat[idx_train_val]
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=val_size / (1 - test_size), stratify=strat_tv, random_state=seed
    )
    return idx_train, idx_val, idx_test
