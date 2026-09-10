# VQC Generator: Fair Interpretable Data
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/KLS-01/VQC_Generator-Fair-Interpretable-Data/blob/main/LICENSE)

**A cura di:** [Leonardo Schiavo](https://github.com/KLS-01/)

**Tecnologie usate:**

<p align="left">
  <!-- Python -->
  <a href="https://www.python.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="45" height="40" />
  </a>
  <!-- Qiskit -->
  <a href="https://qiskit.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/Qiskit/qiskit.org/3633ec63a67ca44b2ec293682ec06535e253a633/static/images/qiskit-logo.svg" alt="qiskit" width="45" height="40" />
  </a>
  <!-- NumPy -->
  <a href="https://numpy.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="numpy" width="45" height="40" />
  </a>
  <!-- Pandas -->
  <a href="https://pandas.pydata.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="pandas" width="45" height="40" />
  </a>
  <!-- Scikit-Learn -->
  <a href="https://scikit-learn.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png" alt="scikit-learn" width="45" height="40" />
  </a>
  <!-- Qsalto -->
  <a href="https://github.com/Mc-Zen/qsalto" target="_blank" rel="noreferrer">
    <img src="https://github.com/Mc-Zen/qsalto/raw/main/docs/media/logo-dark.svg" alt="qsalto" height="40" />
  </a>
</p>

## 🎯 1. Obiettivo Scientifico e Razionale Metodologico

L’obiettivo del presente lavoro è progettare e valutare un sistema gene rativo basato su VQC per la produzione di dati sintetici tabulari fair analizzandone e ottimizzandone il trade-off tridimensionale tra espressività dell’ansatz (**utility**), mitigazione del bias (**fairness**) e interpretabilità del circuito (**explainability**). 

### 1.1 Perché i VQC: Ottimizzazione Vincolata ed Esplicitabilità dei Parametri

La peculiarità dell'approccio VQC risiede invece in due proprietà strutturali assenti nei modelli generativi linguistici classici:

* **Fairness come Vincolo Variazionale Esplicito:** A differenza dei modelli basati su prompt (black-box ed euristici), i VQC permettono di incorporare vincoli matematici formali di equità direttamente nella funzione di funzione di costo globale:

$$\mathcal{L}_{\text{total}}(\boldsymbol{\theta}) = \mathcal{L}_{\text{fidelity}}(\boldsymbol{\theta}) + \lambda_{\text{fair}} \cdot \left[\Delta P_{\text{proxy}}(\boldsymbol{\theta})\right]^2$$

I gradienti analitici $\nabla_{\boldsymbol{\theta}}\mathcal{L}$ vengono calcolati in modo esatto nello spazio degli stati tramite **Parameter-Shift Rule** su spazio di Hilbert, consentendo una convergenza analitica congiunta tra fedeltà distributiva ed equità statistica (mitigazione del bias).

* **Trasparenza e Ispezionabilità Causale:** I VQC consentono di analizzare esplicitamente il contributo dei singoli gate e parametri del circuito tramite metodi analitici quantistici (Quantum Shapley Values, entanglement globale Meyer-Wallach, purezza con qsalto), permettendo di quantificare il trade-off tra espressività architetturale, complessità circuitale e stabilità delle spiegazioni.

## 🧩 2. File Sorgente Python (`.py`) e Architettura Modulare

| File Sorgente | Modulo / Ruolo | Responsabilità Operativa |
| --- | --- | --- |
| `data_pipeline.py` | **Data Ingestion & Preprocessing** | Download deterministico da UCI ML Repo, selezione $k$-best feature tramite Mutual Information ($k \le 8$), binarizzazione soglie protette, derivazione $A_{\text{overall}}$ e scaling angolare $[0, \pi]$ per rotazioni quantistiche.|
| `quantum_vqc.py` | **Quantum Generator Engine** | Template circuitale parametrizzato (Ry-Rz rotazioni + CNOT entangling), codifica Parameter-Shift Rule esatta, ottimizzatore variazionale con loss di fairness vincolata ($\lambda_{\text{fair}}=5.0$). |
| `dataset_export.py` | **Dataset Exporter** | Esportazione dei dataset reali di riferimento e dei dataset sintetici generati dai VQC con ricostruzione delle scale originali. |
| `evaluation.py` | **Downstream Evaluator** | Protocollo TSTR (*Train on Synthetic, Test on Real*) su 3 classificatori (`RandomForest`, `XGBoost_stub`, `MLP`); calcolo utility ($F_1$, Accuracy) e fairness multi-attributo (SPD, EOD, AOD).|
| `bell_sampling.py` | **Quantum Native Diagnostics** | Bell State Measurement a due copie; calcolo della Trace Distance sugli stati di input, Quantum Statistical Parity (`qsp_diff_output`) e identificazione delle Bias Pairs.|
| `entanglement_analysis.py` | **Entanglement Engine** | Calcolo dell'entanglement multipartito (indice di Meyer-Wallach) e stima del coefficiente Shor-Laflamme $a_1$ (purezza media dei singoli qubit) mediante protocollo di Bell sampling a due copie (2048 shot) tramite la libreria `qsalto`. |
| `svqx.py` | **Quantum Explainability (XAI)** | Calcolo esatto dei Quantum Shapley Values (SVQX) a livello di singoli gate groups (rotazioni per qubit e blocchi CNOT).|
| `pipeline_orchestrator.py` | **End-to-End Orchestrator** | Gestione della griglia $3 \times 6 \times 3 = 54\text{ run}$, checkpointing atomico transazionale con manifest `.json`, meccanismo *self-healing* per resume idempotente senza duplicazioni, esportazione raw unificata con iniezione di 6 covariate strutturali.|
| `dataset_analysis.py` | **Descriptive Analytics** | Estrazione parametri statistici (Min, Max, Quartili, Std, IQR) su feature continue e frequenze marginali per classi target e sensibili.|
| `statistical_engine.py` | **Inferential Statistics Engine** | Shapiro-Wilk diagnostico bloccante, test omnibus di Friedman con correzione Holm / FDR-BH, matrici post-hoc Nemenyi, stabilità inter-seed SVQX e 12 correlazioni di complessità architetturale.|
| `visualization_suite.py` | **Visual Analytics Suite** | Generazione di 75 figure PNG ad alta risoluzione ($300\text{ DPI}$): Heatmap di performance e Nemenyi, barplot stabilità SVQX, scatter plot con bande di confidenza $95\%$.|
| `validation_suite.py` | **Quality Gate & Compliance** | Validazione formale cumulativa: verifica shape, unicità chiavi primarie, range matematici delle matrici, merge relazionali $1:1$ RQ2-RQ3.|

## 🔬 3. Disegno Sperimentale e Scelte di Codifica

```
3 Dataset UCI × 6 Configurazioni VQC × 3 Seed Locali = 54 Run Totali
```

```
           ┌───────────────────────────────────────────────┐
           │              PIPELINE DI RICERCA              │
           └───────────────────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
     │  German Credit  │  │  Heart Disease  │  │  Student Perf.  │
     │     (9 Qubit)   │  │    (10 Qubit)   │  │    (10 Qubit)   │
     └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
      ┌────────────────────────────┴────────────────────────────┐
      │    6 Configurazioni VQC: L1/L2 × Linear/All-to-All      │
      │    Single Encoding & Data Re-Uploading (3 Seed Ciascuna)│
      └────────────────────────────┬────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
     │ RQ1: Utility    │  │ RQ2: Fairness   │  │ RQ3: Quantum    │
     │ F1, Accuracy    │  │ SPD, EOD, AOD   │  │ SVQX, qsalto,   │
     │ TSTR Protocol   │  │ Multi-Attributo │  │ Complessità     │
     └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 3.1 Le 6 configurazioni circuitali

| ID Configurazione | Profondità ($L$) | Topologia Entanglement | Strategia Encoding | Parametri ($N_q=10$) | Porte CNOT ($N_q=10$) |
| --- | --- | --- | --- | --- | --- |
| `L1_linear` | 1 | Lineare (nearest neighbor) | Single Encoding | 20 | 9 |
| `L1_all_to_all` | 1 | Completa (All-to-All) | Single Encoding | 20 | 45 |
| `L2_linear` | 2 | Lineare (nearest neighbor) | Single Encoding | 40 | 18 |
| `L2_all_to_all` | 2 | Completa (All-to-All) | Single Encoding | 40 | 90 |
| `L2_re_upload` | 2 | Lineare (nearest neighbor) | Data Re-Uploading | 40 | 18 |
| `L2_all_to_all_re_upload` | 2 | Completa (All-to-All) | Data Re-Uploading | 40 | 90 |

### 3.2 Allocazione del Budget di Qubit per Dataset

Il simulatore esegue calcoli su vettori di stato esatti ($2^{N_q}$ ampiezze complesse). Il vincolo fisico è:

$$N_{\text{qubits}} = d_{\text{features}} + 1_{\text{target}} + 1_{\text{sensitive}}$$

* **German Credit (UCI 144)**: $7\text{ feature} + 1\text{ target} + 1\text{ sensibile} = \mathbf{9\text{ qubit}}$ ($512\text{ ampiezze}$). Gate groups analizzati: **10** (9 rotazioni qubit + 1 blocco CNOT).

* **Heart Disease (UCI 45)**: $8\text{ feature} + 1\text{ target} + 1\text{ sensibile} = \mathbf{10\text{ qubit}}$ ($1024\text{ ampiezze}$). Gate groups analizzati: **11** (10 rotazioni qubit + 1 blocco CNOT).

* **Student Performance (UCI 320)**: $8\text{ feature} + 1\text{ target} + 1\text{ sensibile} = \mathbf{10\text{ qubit}}$ ($1024\text{ ampiezze}$). Gate groups analizzati: **11** (10 rotazioni qubit + 1 blocco CNOT).

### 3.3 Codifica degli Attributi Sensibili e Scelte di Protezione

* **Binarizzazione a Soglia Fissa ($A_{\text{age}}$):** German Credit ($\text{Age} \ge 25$), Heart Disease ($\text{Age} \ge 55$), Student Performance ($\text{Age} \ge 18$).

* **Attributo Demografico ($A_{\text{sex}}$):** Genere codificato dal dominio d'origine.

* **Attributo Intersezionale ($A_{\text{overall}}$):** Calcolato come $A_{\text{overall}} = (A_{\text{sex}} == 1) \land (A_{\text{age}} == 1)$.

* **Perché $A_{\text{overall}}$ è l'unico attributo ottimizzato ed esportato:** Ottimizzare il circuito sull'intersezione previene il *fairness gerrymandering* (un modello può risultare equo su genere ed età presi singolarmente pur discriminando sistematicamente il sottogruppo congiunto).

* **Fairness Through Unawareness:** L'attributo `sex` non entra mai nella matrice predittiva $X$ per prevenire il *disparate treatment* diretto da parte dei classificatori classici. `age` è presente sia come feature continua predittiva in $X$ sia come base di discretizzazione per $A_{\text{age}}$.

### 3.4 Iperparametri di Addestramento e Protocollo di Valutazione

* **Ottimizzazione VQC:** `EPOCHS = 15`, `LEARNING_RATE = 0.4`, `LAMBDA_FAIR = 5.0`, `BATCH_FRAC = 0.25`.

* **Replicabilità:** Seed globali e locali `SEEDS = [7, 17, 27]`.

* **Split Dati:** $70\%$ Training (ricostruzione sintetica) e $30\%$ Test set reale stratificato su $A_{\text{overall}}$.

## 📊 4. Research Questions (RQ) e Struttura Metriche
```
                 ┌─────────────────────────────────────────┐
                 │     QUADRO DELLE RESEARCH QUESTIONS     │
                 └────────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│       RQ1        │        │       RQ2        │        │       RQ3        │
│ Downstream       │        │ Downstream       │        │ Quantum Audit,   │
│ Utility (TSTR)   │        │ Fairness         │        │ XAI & Complexity │
├──────────────────┤        ├──────────────────┤        ├──────────────────┤
│ • F1-Score       │        │ • SPD (Demogr.)  │        │ • SVQX Gate XAI  │
│ • Accuracy       │        │ • EOD (Eq. Odds) │        │ • Ranking Stab.  │
│ • Random Forest  │        │ • AOD (Avg Odds) │        │ • Meyer-Wallach  │
│ • XGBoost stub   │        │ • Age, Sex, Over.│        │ • qsalto Purity  │
│ • MLP Classifier │        │ • Trace Distance │        │ • 12 Correlazioni│
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

### RQ1: Downstream Utility (Protocollo TSTR)

* **Obiettivo:** Verificare se i dati generati dal VQC l'utilità predittiva del dominio addestrando classificatori standard sui soli dati sintetici e testandoli sul test set reale (*Train on Synthetic, Test on Real*)..

* **Metriche:** F1-Score (macro/weighted), Accuracy.

* **Classificatori Downstream:** `RandomForest`, `XGBoost_stub` (GradientBoosting), `MLP` (Multi-Layer Perceptron).

* **File Raw Generato:** `Analysis/RQ1_data/rq1_models_results.csv` ($162\text{ righe} = 54\text{ run} \times 3\text{ modelli}$).

* **Output Statistici:** `rq1_mean_std.csv`, `shapiro.csv`, `friedman.csv` ($9\text{ blocchi}$ `seed × model`), `f1_NemenyiTestResults.csv`, `accuracy_NemenyiTestResults.csv`.

* **Output Grafici:** `f1_NemenyiTestResults_heatmap.png`, `accuracy_NemenyiTestResults_heatmap.png`.

### RQ2: Downstream Multi-Attribute Fairness & Quantum Diagnostics

* **Obiettivo:** Rispondere alla valutazione indiretta della fairness, misurando la mitigazione del bias sui singoli attributi protetti (`age`, `sex`) e sulla loro congiunzione intersezionale (`overall`), correlando l'equità con la geometria dello spazio di Hilbert

* **Metriche di Fairness Classica:**
  * Statistical Parity Difference: $\text{SPD} = P(\hat{Y}=1 \mid A=0) - P(\hat{Y}=1 \mid A=1)$
  * Equal Opportunity Difference: $\text{EOD} = P(\hat{Y}=1 \mid A=0, Y=1) - P(\hat{Y}=1 \mid A=1, Y=1)$
  * Average Odds Difference: $\text{AOD} = \frac{1}{2} \left( \vert{}\text{FPR}_{A=0} - \text{FPR}_{A=1}\vert{} + \vert{}\text{TPR}_{A=0} - \text{TPR}_{A=1}\vert{} \right)$

* **Metriche Quantum-Native:**
  * `trace_distance_input`: Distanza di traccia tra le matrici densità degli stati di input protetti e non protetti.
  * `qsp_diff_output`: Differenza di parità statistica quantistica misurata direttamente all'uscita del circuito.
  * `is_bias_pair_candidate`: Flag booleano di identificazione delle coppie di bias (coppie con elevata divergenza geometrica).

* **File Raw Generato:** `Analysis/RQ2_data/rq2_all_fairness_results_multiattribute.csv` ($486\text{ righe} = 54\text{ run} \times 3\text{ modelli} \times 3\text{ attributi}$).

* **Output Statistici:** `rq2_mean_std.csv`, `shapiro.csv`, `friedman.csv` (per dataset e per singolo attributo a $9\text{ blocchi}$), matrici Nemenyi per ciascuna combinazione metrica-attributo.

* **Output Grafici:** Heatmap Nemenyi disaggregate per `age`, `sex` e `overall` su SPD, EOD, AOD.

### RQ3: Quantum Structural Audit, Spiegabilità Causale (SVQX) e Complessità

* **Obiettivo:** Ispezionare i meccanismi interni del VQC, misurare il contributo causale dei singoli gate, quantificare la stabilità delle spiegazioni e correlare la complessità circuitale con fairness ed entanglement. Indagare l'effetto di profondità ($L$), topologia di entanglement ed encoding sulle proprietà quantistiche pure (Meyer-Wallach, qsalto) e verificare il trade-off tra espressività architetturale, stabilità delle spiegazioni (SVQX) e mitigazione del bias.

* **Metriche di Spiegabilità e Circuito:**
  * Quantum Shapley Values (SVQX): Attribuzione di importanza a ciascun gate group $g (contributo causale esatto di ogni gate group).
  * Indice di Concentrazione Gini SVQX: Misura di concentrazione/dispersione dell'attribuzione causale nel circuito.
  * Stabilità Inter-Seed SVQX: Concordanza media di rango pairwise di Spearman ($\bar{\rho}$) tra i ranking dei gate group nelle repliche con seed 7, 17 e 27.
  * Meyer-Wallach Entanglement Score: Misura dell'entanglement globale medio a molti corpi generato dal circuito.
  * Stime di Purezza con qsalto ($A_1$ estimate): Stima di purezza tramite Bell Sampling a 2 copie con $2048\text{ shot}$.

* **Analisi di Complessità e Trade (12 Coppie per Dataset):**
  * Predittori di Complessità ($X$): `cnot_count`, `n_params`, `n_layers`.
  * Outcome ($Y$): `aod_mean`, `spd_mean`, `eod_mean`, `entanglement_score`.
  * Metodo: Correlazione di Pearson (se normalità Shapiro verificata) o Spearman (se non normale).

* **File Raw Generato:** `Analysis/RQ3_data/rq3_all_results.csv` ($576\text{ righe} = 18\text{ run}\times 10\text{ gate} + 36\text{ run}\times 11\text{ gate}$).

* **Output Statistici:** `rq3_gate_mean_std.csv`, `rq3_gate_shapiro.csv`, `gate/friedman.csv` ($30/33\text{ blocchi}$), `svqx_value_NemenyiTestResults.csv`, `svqx_stability_spearman.csv`, `rq3_run_mean_std.csv`, `rq3_run_shapiro.csv`, `run/friedman.csv`, `complexity_correlations.csv`.

* **Output Grafici:** `svqx_NemenyiTestResults_heatmap.png`, `svqx_stability_bars.png` (barplot in $[-1, 1]$), $36\text{ scatter plot}$ `complexity_scatter_<metric_x>_<metric_y>.png` con retta di regressione e banda di confidenza $95\%$.

## 📁 5. Gerarchia "\artifacts"

```text
Quantum_Dataset_Generation_Online_Appendix/
Datasets/
├── german_credit/                             (19 CSV: 1 reale + 18 sintetici)
├── heart_disease/                             (19 CSV: 1 reale + 18 sintetici)
└── student_performance/                       (19 CSV: 1 reale + 18 sintetici)

DatasetsAnalysis/
├── german_credit/{age,overall,target}_statistics.csv
├── heart_disease/{age,overall,target}_statistics.csv
└── student_performance/{age,overall,target}_statistics.csv

Analysis/
├── RQ1_data/rq1_models_results.csv
├── RQ2_data/rq2_all_fairness_results_multiattribute.csv
└── RQ3_data/rq3_all_results.csv

StatisticalResults/
├── RQ1/<German|Heart|Student>/
│   ├── rq1_mean_std.csv, shapiro.csv, friedman.csv
│   └── {f1,accuracy}_NemenyiTestResults.csv
├── RQ2/<German|Heart|Student>/
│   ├── rq2_mean_std.csv, shapiro.csv, friedman.csv
│   └── {age,sex,overall}/
│       ├── friedman.csv
│       └── {spd,eod,aod,
│           trace_distance_input,
│           qsp_diff_output}_NemenyiTestResults.csv
└── RQ3/<German|Heart|Student>/
    ├── gate/
    │   ├── rq3_gate_mean_std.csv, rq3_gate_shapiro.csv, friedman.csv
    │   ├── svqx_value_NemenyiTestResults.csv
    │   └── svqx_stability_spearman.csv
    └── run/
        ├── rq3_run_mean_std.csv, rq3_run_shapiro.csv, friedman.csv
        ├── {svqx_concentration,
        │   entanglement_score,
        │   qsalto_a1_estimate}_NemenyiTestResults.csv
        └── complexity_correlations.csv

VisualizationResults/
├── RQ1/<German|Heart|Student>/
│   └── {f1,accuracy}_NemenyiTestResults_heatmap.png
├── RQ2/<German|Heart|Student>/<age|sex|overall>/
│   └── {spd,eod,aod}_NemenyiTestResults_heatmap.png
└── RQ3/<German|Heart|Student>/
    ├── svqx_NemenyiTestResults_heatmap.png
    ├── svqx_stability_bars.png 
    └── complexity_scatter_{cnot_count,
                           n_params,
                           n_layers}_{aod_mean,
                                     spd_mean,
                                     eod_mean,
                                     entanglement_score}.png
    
ValidationResults/
├── validation_report.txt    (formale di validazione: [PASS])
└── validation_errors.txt    (Tracciamento errori: vuoto a pipeline completata)
```

## ⚡ 6. Pipeline di Esecuzione e Guida Operativa

### 6.1 Dipendenze Software Esatte

Per garantire la piena riproducibilità scientifica ed evitare disallineamenti tra versioni di librerie, si raccomanda di utilizzare le seguenti versioni dei pacchetti:

```text
qiskit==2.5.2
qiskit-aer==0.17.2
ucimlrepo==0.0.7
qsalto==0.2.2
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.1.3
scipy==1.16.3
statsmodels==0.14.6
matplotlib==3.10.0
seaborn==0.13.2
rustworkx==0.18.1
stevedore==5.9.1
```

Comando di installazione a prova di invecchiamento:

```bash
pip install qiskit==2.5.2 qiskit-aer==0.17.2 ucimlrepo==0.0.7 qsalto==0.2.2 scikit-learn==1.6.1 pandas==2.2.3 numpy==2.1.3 scipy==1.16.3 statsmodels==0.14.6 matplotlib==3.10.0 seaborn==0.13.2
```

### 6.2 Sequenza Operativa di Esecuzione

Per utilizzare la pipeline, lanciare i seguenti comandi:

```bash
# 1. Esecuzione della computazione quantistica end-to-end (54 run)
python -u pipeline_orchestrator.py

# 2. Estrazione delle statistiche descrittive sui dataset esportati
python -u dataset_analysis.py

# 3. Elaborazione del motore statistico inferenziale (Friedman, Nemenyi, Correlazioni)
python -u statistical_engine.py

# 4. Generazione delle 75 visualizzazioni grafiche ad alta risoluzione
python -u visualization_suite.py

# 5. Audit finale e certificazione di conformità
python -u validation_suite.py
```

