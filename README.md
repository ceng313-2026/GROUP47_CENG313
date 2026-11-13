# GROUP47_CENG313 — CAFA 6 Protein Function Prediction

Course project for **CENG313 Introduction to Data Science** (group **47**).

**Kaggle competition:** [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
— predicting the Gene Ontology (GO) terms of a protein from its amino-acid sequence.

**Current Kaggle score:** not available. The competition is closed and no longer accepts submissions
(see [Kaggle submission history](#kaggle-submission-history)).

**Current held-out score:** mean IA-weighted F-max **0.421** (MLP + propagation, Assignment 1 baseline).

> The competition is closed and late submissions are not accepted, so all models are evaluated on a
> held-out 20% split of the training data (`random_state=42`) with the competition metric.

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

Key numbers (from section 2 of the Assignment 1 notebook):

| Quantity | Value |
|---|---|
| Training proteins | 82,404 |
| Annotation rows (protein, GO term) | 537,027 (3,564,990 after propagation) |
| Distinct GO terms in train | 26,125 (29,386 after propagation) |
| Species in train | 1,381 |
| Test proteins | 224,309 (8,453 species) |
| Missing values / duplicate ids | none |

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

Held-out validation set (20% of the training proteins, 16,481 proteins), IA-weighted F-max
(notebook section 5.5):

| Model | BPO | CCO | MFO | Mean |
|---|---|---|---|---|
| Naive (term frequency) | 0.224 | 0.359 | 0.364 | 0.316 |
| k-NN (k=10) | 0.269 | 0.416 | 0.430 | 0.372 |
| MLP | 0.288 | 0.479 | 0.484 | 0.417 |
| **MLP + propagation** | **0.292** | **0.482** | **0.490** | **0.421** |

For reference: random scores give 0.056, and a perfect prediction of the 1,100 terms we keep would give 0.873.

Main findings: the naive baseline is already strong (0.316) because a few generic GO terms occur in most
proteins. Using the sequence helps: the MLP with hierarchy propagation is 33% better than the naive
baseline. Biological Process is clearly the hardest aspect (0.292), which matches the EDA: it has the
largest vocabulary, the deepest terms and the longest tail. The best baseline reaches less than half of
the upper bound of its own label space, so there is a lot of room for Assignment 2.

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

| Date | Model | Validation F-max (mean) | Kaggle public | Kaggle private | Notes |
|---|---|---|---|---|---|
| 2026-11-13 | MLP + propagation (Assignment 1 baseline) | 0.421 | not available | not available | Late submission attempted with the Kaggle API and rejected (`400 Bad Request`); the submit button on the competition page is disabled. See notebook section 6 and `submissions/`. |

## Progress log

| Date | What was done |
|---|---|
| 2026-11-10 | Studied the competition, its data files and the evaluation metric; planned Assignment 1 |
| 2026-11-13 | Created the repository and the README; joined the course organization and the Kaggle competition |
| 2026-11-13 | Added the source modules (`src/`) and the Assignment 1 notebook; transferred the repository to `ceng313-2026` |
| 2026-11-13 | Ran the notebook on the full data (EDA, preprocessing, three baselines), wrote the interpretations |
| 2026-11-13 | Tried a Kaggle late submission (rejected, competition closed); reported the held-out evaluation instead |

## References

- CAFA 6 competition page: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction
- Gene Ontology: http://geneontology.org
- Public write-up used for the comparison in notebook section 6: https://nihalsulaiman.blogspot.com/2026/01/cafa-6-protein-function-prediction-bio.html
- Libraries: NumPy, pandas, SciPy, scikit-learn, Matplotlib; Kaggle API for the data download.
- AI assistance: Claude (Anthropic) was used to help write the code in `src/`, the notebook and its text.
  No Kaggle notebook or cell was copied.
