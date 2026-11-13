"""Loading and preprocessing the CAFA 6 competition data.

Part 1 reads the competition files into pandas / scipy structures.
Part 2 handles the Gene Ontology graph and propagates labels along it.

The competition archive contains (folder names may differ between downloads,
so files are located by name, not by fixed path):

    train_sequences.fasta        amino-acid sequences of the training proteins
    train_terms.tsv              EntryID, GO term, aspect (ground-truth labels)
    train_taxonomy.tsv           EntryID, NCBI taxon id
    go-basic.obo                 Gene Ontology graph
    IA.tsv                       information accretion weight of every GO term
    testsuperset.fasta           sequences we have to predict for
    testsuperset-taxon-list.tsv  taxon id -> species name for the test set
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# The 20 standard amino acids (one-letter codes).
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# The aspect column is written either as one letter or as a three-letter code.
ASPECT_ALIASES = {
    "P": "BPO", "BPO": "BPO", "BP": "BPO", "biological_process": "BPO",
    "C": "CCO", "CCO": "CCO", "CC": "CCO", "cellular_component": "CCO",
    "F": "MFO", "MFO": "MFO", "MF": "MFO", "molecular_function": "MFO",
}

_OX_PATTERN = re.compile(r"\bOX=(\d+)")


def find_file(data_dir, filename: str) -> Path:
    """Return the path of `filename` anywhere below `data_dir` (case-insensitive)."""
    data_dir = Path(data_dir)
    matches = [p for p in data_dir.rglob("*") if p.is_file() and p.name.lower() == filename.lower()]
    if not matches:
        raise FileNotFoundError(f"{filename} was not found under {data_dir.resolve()}")
    return sorted(matches)[0]


def parse_fasta_header(header: str):
    """Extract (EntryID, taxon id or None) from a FASTA header line (without '>').

    Handles both header styles used in the competition:
        sp|P12345|NAME_HUMAN Protein name OS=Homo sapiens OX=9606 GN=... (train)
        P12345 9606                                                     (test)
    """
    tokens = header.split()
    first = tokens[0]
    parts = first.split("|")
    entry_id = parts[1] if len(parts) >= 3 else first

    taxon = None
    match = _OX_PATTERN.search(header)
    if match:
        taxon = int(match.group(1))
    elif len(tokens) > 1 and tokens[1].isdigit():
        taxon = int(tokens[1])
    return entry_id, taxon


def read_fasta(path) -> pd.DataFrame:
    """Read a FASTA file into a DataFrame: EntryID, taxonomyID, sequence, length."""
    ids, taxa, seqs = [], [], []
    chunks: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if ids:
                    seqs.append("".join(chunks))
                entry_id, taxon = parse_fasta_header(line[1:])
                ids.append(entry_id)
                taxa.append(taxon)
                chunks = []
            else:
                chunks.append(line)
    if ids:
        seqs.append("".join(chunks))

    df = pd.DataFrame({"EntryID": ids, "sequence": seqs})
    df["taxonomyID"] = pd.array(taxa, dtype="Int64")
    df["length"] = df["sequence"].str.len().astype(int)
    return df[["EntryID", "taxonomyID", "sequence", "length"]]


def _read_tsv_no_header(path, n_cols: int, is_data_row) -> pd.DataFrame:
    """Read a TSV as strings and drop the first row if it is a header line."""
    df = None
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, sep="\t", header=None, dtype=str, encoding=encoding,
                             keep_default_na=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"could not decode {path}")
    df = df.iloc[:, :n_cols]
    if len(df) and not is_data_row(df.iloc[0]):
        df = df.iloc[1:]
    return df.reset_index(drop=True)


def load_train_terms(path) -> pd.DataFrame:
    """Ground-truth annotations: one row per (EntryID, GO term), plus the aspect."""
    df = _read_tsv_no_header(path, 3, lambda row: str(row.iloc[1]).startswith("GO:"))
    df.columns = ["EntryID", "term", "aspect"][: df.shape[1]]
    if "aspect" in df.columns:
        df["aspect"] = df["aspect"].map(ASPECT_ALIASES).fillna(df["aspect"])
    return df


def load_taxonomy(path) -> pd.DataFrame:
    """Taxon id of every training protein."""
    df = _read_tsv_no_header(path, 2, lambda row: str(row.iloc[1]).strip().isdigit())
    df.columns = ["EntryID", "taxonomyID"]
    df["taxonomyID"] = df["taxonomyID"].astype(int)
    return df


def load_ia(path) -> pd.Series:
    """Information accretion (IA) weight per GO term, as a Series indexed by term."""
    df = _read_tsv_no_header(path, 2, lambda row: str(row.iloc[0]).startswith("GO:"))
    df.columns = ["term", "IA"]
    return df.set_index("term")["IA"].astype(float)


def load_test_taxon_list(path) -> pd.DataFrame:
    """Taxon id -> species name table shipped with the test superset."""
    df = _read_tsv_no_header(path, 2, lambda row: str(row.iloc[0]).strip().isdigit())
    df.columns = ["taxonomyID", "species"]
    df["taxonomyID"] = df["taxonomyID"].astype(int)
    return df


def load_cafa6(data_dir, load_test: bool = True) -> dict:
    """Load every competition table. Returns a dict of DataFrames / Series / paths."""
    data = {
        "train_seq": read_fasta(find_file(data_dir, "train_sequences.fasta")),
        "train_terms": load_train_terms(find_file(data_dir, "train_terms.tsv")),
        "train_tax": load_taxonomy(find_file(data_dir, "train_taxonomy.tsv")),
        "ia": load_ia(find_file(data_dir, "IA.tsv")),
        "obo_path": find_file(data_dir, "go-basic.obo"),
    }
    if load_test:
        data["test_seq"] = read_fasta(find_file(data_dir, "testsuperset.fasta"))
        try:
            data["test_taxa"] = load_test_taxon_list(find_file(data_dir, "testsuperset-taxon-list.tsv"))
        except FileNotFoundError:
            data["test_taxa"] = pd.DataFrame(columns=["taxonomyID", "species"])
    return data


def amino_acid_composition(sequences, alphabet=AMINO_ACIDS) -> pd.Series:
    """Relative frequency of each residue over a collection of sequences.

    Residues outside `alphabet` (X, U, B, Z, O ...) are summed into 'other'.
    """
    counts = dict.fromkeys(alphabet, 0)
    total = 0
    for seq in sequences:
        total += len(seq)
        for aa in alphabet:
            counts[aa] += seq.count(aa)
    result = pd.Series(counts, dtype=float)
    result["other"] = total - result.sum()
    return result / max(total, 1)


def build_label_matrix(terms: pd.DataFrame, protein_ids, term_ids) -> sparse.csr_matrix:
    """Binary matrix Y[i, j] = 1 if protein i is annotated with term j.

    Rows follow `protein_ids`, columns follow `term_ids`; annotations that
    refer to anything outside these two lists are ignored.
    """
    protein_index = {p: i for i, p in enumerate(protein_ids)}
    term_index = {t: j for j, t in enumerate(term_ids)}
    rows = terms["EntryID"].map(protein_index)
    cols = terms["term"].map(term_index)
    keep = (rows.notna() & cols.notna()).to_numpy()
    rows = rows.to_numpy()[keep].astype(np.int64)
    cols = cols.to_numpy()[keep].astype(np.int64)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(protein_index), len(term_index)),
    )
    matrix.data[:] = 1.0  # duplicates were summed by the constructor
    return matrix


# ---------------------------------------------------------------------------
# Gene Ontology (GO) graph and label propagation
#
# GO is a directed acyclic graph. If a protein has a term, it also has every
# ancestor of that term ("true path rule"). CAFA follows `is_a` and `part_of`
# edges, so we do the same.
# ---------------------------------------------------------------------------

ASPECTS = ["BPO", "CCO", "MFO"]
ASPECT_FULL_NAMES = {
    "BPO": "Biological Process",
    "CCO": "Cellular Component",
    "MFO": "Molecular Function",
}
NAMESPACE_TO_ASPECT = {
    "biological_process": "BPO",
    "cellular_component": "CCO",
    "molecular_function": "MFO",
}
ROOT_TERMS = {"BPO": "GO:0008150", "CCO": "GO:0005575", "MFO": "GO:0003674"}


class GeneOntology:
    """Minimal in-memory GO graph (no external dependency)."""

    def __init__(self, names, namespaces, parents, obsolete, alt_ids):
        self.names = names            # term -> human readable name
        self.namespaces = namespaces  # term -> BPO / CCO / MFO
        self.parents = parents        # term -> set of direct parents
        self.obsolete = obsolete      # set of obsolete terms
        self.alt_ids = alt_ids        # alternative id -> primary id
        self._ancestors: dict[str, frozenset] = {}
        self._depth: dict[str, int] = {}

    @classmethod
    def from_obo(cls, path, relationships=("is_a", "part_of")) -> "GeneOntology":
        names, namespaces, parents, alt_ids = {}, {}, {}, {}
        obsolete = set()
        stanza = None  # fields of the [Term] block we are currently inside

        def flush(block):
            if not block or "id" not in block:
                return
            term = block["id"]
            names[term] = block.get("name", "")
            namespaces[term] = NAMESPACE_TO_ASPECT.get(block.get("namespace", ""), "")
            if block.get("obsolete"):
                obsolete.add(term)
                parents[term] = set()
            else:
                parents[term] = block["parents"]
            for alt in block["alt"]:
                alt_ids[alt] = term

        with open(path, encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line.startswith("["):
                    flush(stanza)
                    stanza = {"parents": set(), "alt": []} if line == "[Term]" else None
                    continue
                if stanza is None or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.split("!")[0].strip()
                if key == "id":
                    stanza["id"] = value
                elif key == "name":
                    stanza["name"] = value
                elif key == "namespace":
                    stanza["namespace"] = value
                elif key == "alt_id":
                    stanza["alt"].append(value)
                elif key == "is_obsolete":
                    stanza["obsolete"] = value == "true"
                elif key == "is_a" and "is_a" in relationships:
                    stanza["parents"].add(value.split()[0])
                elif key == "relationship":
                    rel, target = value.split()[:2]
                    if rel in relationships:
                        stanza["parents"].add(target)
        flush(stanza)
        return cls(names, namespaces, parents, obsolete, alt_ids)

    def __contains__(self, term) -> bool:
        return term in self.parents

    def __len__(self) -> int:
        return len(self.parents)

    def canonical(self, term: str) -> str:
        """Map an alternative id to its primary id (unknown ids are returned as is)."""
        return self.alt_ids.get(term, term)

    def name(self, term: str) -> str:
        return self.names.get(self.canonical(term), "")

    def aspect(self, term: str) -> str:
        return self.namespaces.get(self.canonical(term), "")

    def ancestors(self, term: str, include_self: bool = False) -> frozenset:
        """All terms reachable by following parent edges (transitive closure)."""
        term = self.canonical(term)
        if term not in self._ancestors:
            # iterative DFS so that deep chains cannot hit the recursion limit
            seen, stack = set(), list(self.parents.get(term, ()))
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                cached = self._ancestors.get(current)
                if cached is not None:
                    seen |= cached
                else:
                    stack.extend(self.parents.get(current, ()))
            self._ancestors[term] = frozenset(seen)
        result = self._ancestors[term]
        return result | {term} if include_self else result

    def depth(self, term: str) -> int:
        """Length of the shortest path from the term up to its root (root = 0)."""
        term = self.canonical(term)
        if term not in self._depth:
            # breadth-first search upwards until a root (no parents) is reached
            frontier, level, seen = {term}, 0, {term}
            while frontier:
                if any(not self.parents.get(t) for t in frontier):
                    break
                frontier = {p for t in frontier for p in self.parents.get(t, ())} - seen
                seen |= frontier
                level += 1
            self._depth[term] = level
        return self._depth[term]


def propagate_annotations(terms: pd.DataFrame, go: GeneOntology) -> pd.DataFrame:
    """Add every ancestor of every annotated term (true path rule).

    Input and output: DataFrame with columns EntryID, term, aspect.
    Running it on already-propagated data changes nothing.
    """
    fallback_aspect = {}
    if "aspect" in terms.columns:
        fallback_aspect = dict(zip(terms["term"], terms["aspect"]))

    out_ids, out_terms = [], []
    for entry_id, group in terms.groupby("EntryID", sort=False)["term"]:
        full = set()
        for term in set(group):
            full |= go.ancestors(term, include_self=True)
        out_ids.extend([entry_id] * len(full))
        out_terms.extend(full)

    result = pd.DataFrame({"EntryID": out_ids, "term": out_terms})
    aspect_of = {t: (go.aspect(t) or fallback_aspect.get(t, "")) for t in set(out_terms)}
    result["aspect"] = result["term"].map(aspect_of)
    return result
