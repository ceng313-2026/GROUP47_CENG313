# GROUP47_CENG313 — CAFA 6 Protein Function Prediction

Course project for **CENG313 Introduction to Data Science** (group **47**).

**Kaggle competition:** [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
— predicting the Gene Ontology (GO) terms of a protein from its amino-acid sequence.

**Current Kaggle score:** TODO (late submission; public / private score), see [Kaggle submission history](#kaggle-submission-history).

> The competition is closed. TODO, keep the sentence that applies:
> *Late submissions are accepted, so the score above comes from a late submission.* /
> *Late submissions are not accepted, so models are evaluated on a held-out 20% split of the training data (`random_state=42`) with the competition metric.*

## Group members

| Name | Student ID | GitHub username |
|---|---|---|
| Ekin Boran İBİLİ | 23118080073 | gecerpiyon |
| Berat DEMİR | 23118080762 | BeratDemir2 |

## The competition

Proteins carry out most of the work in a cell, but only a small share of the known protein sequences
has an experimentally confirmed function. CAFA (Critical Assessment of Functional Annotation) asks
participants to predict these functions computationally.

- **Input:** the amino-acid sequence of a protein (a string over 20 letters) and its species.
- **Output:** a score in (0, 1] for each GO term the protein might have. One protein has many terms, so
  this is a multi-label problem with tens of thousands of possible labels.
- **Three sub-ontologies**, evaluated separately: Molecular Function (MFO), Biological Process (BPO)
  and Cellular Component (CCO).
- **Hierarchy:** GO terms form a directed acyclic graph. A term implies all of its ancestors.
- **Metric:** protein-centric F-max, weighted by the information accretion (IA) of each term and
  averaged over the three sub-ontologies. Rare, specific terms count more than generic ones.

## Dataset

| File | Content |
|---|---|
| `train_sequences.fasta` | amino-acid sequences of the training proteins |
| `train_terms.tsv` | ground truth: protein id, GO term, aspect |
| `train_taxonomy.tsv` | species (NCBI taxon id) of each training protein |
| `go-basic.obo` | the Gene Ontology graph |
| `IA.tsv` | information accretion weight of each GO term |
| `testsuperset.fasta` | sequences to predict |
| `testsuperset-taxon-list.tsv` | species names of the test set |

Key numbers (TODO: copy from section 2.2 of the Assignment 1 notebook after running it):

| Quantity | Value |
|---|---|
| Training proteins | TODO |
| Annotation rows | TODO |
| Distinct GO terms in train | TODO |
| Species in train | TODO |
| Test proteins | TODO |

**How to download the data.** The data is not stored in this repository (`data/` is in `.gitignore`).
Join the competition on Kaggle and accept its rules, create an API key file (`kaggle.json`), and run the
data cell of the notebook: it calls `kaggle competitions download -c cafa-6-protein-function-prediction`
and unzips the files into `data/`. Alternatively download the files by hand from the competition's *Data*
tab and unzip them into `data/`.

## Repository layout

```
GROUP47_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb   Assignment 1: EDA and baseline models
├── assignment2_solution.ipynb       Assignment 2: proposed solution (due December)
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py        loading FASTA / TSV files, GO graph, label propagation, label matrices
│   ├── models.py      k-mer featuriser and the baseline models
│   └── utils.py       plot style, competition metric (IA-weighted F-max), submission writer
├── submissions/       screenshots of Kaggle submission scores
└── data/              not committed (downloaded from Kaggle)
```

The notebooks import everything reusable from `src/`.

## Approach

### Assignment 1 — EDA and baselines

**Data overview and EDA** (notebook sections 2–3): shapes, data types, missing values and duplicates;
sequence length and outliers, amino-acid composition, species distribution, labels per aspect and per
protein, GO hierarchy and label propagation, long tail of term frequencies, IA weights, relationships
between sequence length / species and the labels, train/test comparison.

**Preprocessing** (section 4): labels are propagated up the GO graph; 20% of the labelled training proteins are held out
for validation (random split, seed 42); only the most frequent terms of each aspect are predicted
(600 BPO, 200 CCO, 300 MFO), while recall is still measured against all true terms.

**Baselines:**

| Model | Description |
|---|---|
| Naive | training frequency of each term, identical for every protein |
| k-NN | label transfer from the 10 most similar training sequences (cosine on 3-mer TF-IDF) |
| MLP | one hidden layer (512 units) on 3-mer TF-IDF, one sigmoid output per term |
| MLP + propagation | MLP scores made consistent with the GO hierarchy |

### Assignment 2 — proposed solution

The proposed solution will be developed in Assignment 2. It will focus on improving the baseline
models and evaluating the improvements against the Assignment 1 results.

## Results

Validation set (held-out 20%), IA-weighted F-max (TODO: copy from notebook section 5.5 after running on the full data):

| Model | BPO | CCO | MFO | Mean |
|---|---|---|---|---|
| Naive (term frequency) | TODO | TODO | TODO | TODO |
| k-NN (k=10) | TODO | TODO | TODO | TODO |
| MLP | TODO | TODO | TODO | TODO |
| MLP + propagation | TODO | TODO | TODO | TODO |

Main findings: TODO (2–4 sentences).

## How to run

### Google Colab (recommended)

1. Open `assignment1_eda_baseline.ipynb` in Colab (*File → Open notebook → GitHub*).
2. In the first code cell set `REPO_URL` to this repository
   (`https://github.com/ceng313-2026/GROUP47_CENG313.git`). The cell clones it so that `src/` can be imported.
3. Accept the competition rules once on the Kaggle competition page, then create an API key file
   (*Kaggle → Settings → API → Legacy API Credentials → Create Legacy API Key*). The data cell asks you
   to upload the resulting `kaggle.json`.
4. *Runtime → Run all*. No GPU is needed.

### Local

```bash
git clone https://github.com/ceng313-2026/GROUP47_CENG313.git
cd GROUP47_CENG313
pip install -r requirements.txt
# put kaggle.json in ~/.kaggle/ , or place the unzipped competition files in ./data
jupyter notebook assignment1_eda_baseline.ipynb
```

The data directory can be changed with the environment variable `CAFA6_DATA_DIR`.
For a quick test run set `MAX_PROTEINS = 5000` in the configuration cell of section 4.

## Kaggle submission history

| Date | Model | Validation F-max (mean) | Kaggle public | Kaggle private | Screenshot |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | `submissions/TODO.png` |

## Progress log

| Date | Who | What was done |
|---|---|---|
| 2026-11-10 | TODO | Studied the competition, its data files and the evaluation metric; planned Assignment 1 |
| 2026-11-13 | TODO | Created the repository and the README |
| 2026-11-13 | TODO | Joined the course organization and the Kaggle competition |
| 2026-11-13 | TODO | Added the source modules (`src/`) and the Assignment 1 notebook |
| TODO | TODO | TODO: ran the notebook on the full data, wrote the interpretations, made the Kaggle submission |

## References

- CAFA 6 competition page: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction
- Gene Ontology: http://geneontology.org
- TODO: add every paper, notebook or tool you actually used.
