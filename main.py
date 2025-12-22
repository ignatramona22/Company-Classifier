from __future__ import annotations
import re, ast
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

GENERIC_TOKENS = {
    "service","services","manufacturing","manufacture","manufacturer","production",
    "processing","repair","repairs","installation","installations","maintenance",
    "and","or","the","other","all","misc","miscellaneous","general","contractor",
    "contractors","construction"
}

def _parse_tags(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, list):
        return " ".join(map(str, x))
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return " ".join(map(str, v))
        except Exception:
            return x
    return str(x)

def _clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\[\]\(\)\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9&/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize_filtered(s: str) -> List[str]:
    toks = [t for t in _clean_text(s).split() if len(t) > 2 and t not in GENERIC_TOKENS]
    return toks

@dataclass
class ClassifierConfig:
    word_ngram_range: Tuple[int,int] = (1,2)
    char_ngram_range: Tuple[int,int] = (3,5)
    max_features: int = 200_000
    min_df: int = 2

    seed_min_overlap: int = 2
    seed_min_label_coverage: float = 0.7
    seed_alt_coverage: float = 0.5
    seed_alt_overlap: int = 3
    seed_min_examples_for_centroid: int = 5

    add_label_min_score: float = 0.20
    add_label_close_ratio: float = 0.92

class InsuranceTaxonomyClassifier:
    def __init__(self, config: Optional[ClassifierConfig]=None):
        self.cfg = config or ClassifierConfig()
        self.word_vec: Optional[TfidfVectorizer] = None
        self.char_vec: Optional[TfidfVectorizer] = None
        self.taxonomy_labels: List[str] = []
        self.label_matrix: Optional[sp.csr_matrix] = None

    def _build_company_text(self, df: pd.DataFrame) -> pd.Series:
        tags = df.get("business_tags", pd.Series([""]*len(df))).apply(_parse_tags)
        s = (
            df.get("description","").fillna("") + " " +
            tags.fillna("") + " " +
            df.get("sector","").fillna("") + " " +
            df.get("category","").fillna("") + " " +
            df.get("niche","").fillna("")
        )
        return s.apply(_clean_text)

    def fit(self, companies: pd.DataFrame, taxonomy: pd.DataFrame) -> "InsuranceTaxonomyClassifier":
        self.taxonomy_labels = taxonomy["label"].astype(str).tolist()
        tax_text = [_clean_text(l) for l in self.taxonomy_labels]

        comp_text = self._build_company_text(companies)

        self.word_vec = TfidfVectorizer(
            ngram_range=self.cfg.word_ngram_range,
            min_df=self.cfg.min_df,
            max_features=self.cfg.max_features,
            stop_words="english",
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.cfg.char_ngram_range,
            min_df=self.cfg.min_df,
            max_features=self.cfg.max_features,
        )

        Xw = self.word_vec.fit_transform(comp_text.tolist() + tax_text)
        Xc = self.char_vec.fit_transform(comp_text.tolist() + tax_text)
        X = sp.hstack([Xw, Xc]).tocsr()

        X_comp = normalize(X[:len(companies)])
        X_tax  = normalize(X[len(companies):])

        comp_struct = (
            companies.get("sector","").fillna("") + " " +
            companies.get("category","").fillna("") + " " +
            companies.get("niche","").fillna("") + " " +
            companies.get("business_tags", pd.Series([""]*len(companies))).apply(_parse_tags).fillna("")
        ).apply(_clean_text)

        comp_tokens = [set(_tokenize_filtered(t)) for t in comp_struct.tolist()]
        label_tokens = [set(_tokenize_filtered(l)) for l in self.taxonomy_labels]

        seeds: Dict[int, List[int]] = {}
        for i, ct in enumerate(comp_tokens):
            if not ct:
                continue
            for j, lt in enumerate(label_tokens):
                if not lt:
                    continue
                inter = len(ct & lt)
                if inter >= self.cfg.seed_min_overlap:
                    coverage = inter / len(lt)
                    if coverage >= self.cfg.seed_min_label_coverage or (
                        inter >= self.cfg.seed_alt_overlap and coverage >= self.cfg.seed_alt_coverage
                    ):
                        seeds.setdefault(j, []).append(i)

        label_vecs = []
        for j in range(len(self.taxonomy_labels)):
            idxs = seeds.get(j, [])
            if len(idxs) >= self.cfg.seed_min_examples_for_centroid:
                centroid = X_comp[idxs].mean(axis=0)
                centroid = sp.csr_matrix(centroid)
            else:
                centroid = X_tax[j]
            label_vecs.append(normalize(centroid))

        self.label_matrix = sp.vstack(label_vecs).tocsr()
        return self

    def predict(self, companies: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        if self.word_vec is None or self.char_vec is None or self.label_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        comp_text = self._build_company_text(companies)
        Xw = self.word_vec.transform(comp_text.tolist())
        Xc = self.char_vec.transform(comp_text.tolist())
        X_comp = normalize(sp.hstack([Xw, Xc]).tocsr())

        sim = (X_comp @ self.label_matrix.T).toarray()
        top_idx = np.argsort(-sim, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(sim, top_idx, axis=1)

        out = companies.copy()
        labels_out = []
        conf_out = []
        alts_out = []

        for idxs, scores in zip(top_idx, top_scores):
            best = float(scores[0])
            chosen = [self.taxonomy_labels[int(idxs[0])]]
            for i in range(1, len(idxs)):
                if float(scores[i]) >= self.cfg.add_label_min_score and float(scores[i]) >= self.cfg.add_label_close_ratio * best:
                    chosen.append(self.taxonomy_labels[int(idxs[i])])

            labels_out.append("; ".join(chosen))
            conf_out.append(best)
            alts_out.append("; ".join([self.taxonomy_labels[int(i)] for i in idxs[1:4]]))

        out["insurance_label"] = labels_out
        out["label_confidence"] = conf_out
        out["label_alt_candidates"] = alts_out
        return out

if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument("--companies", required=True, help="CSV with companies")
    p.add_argument("--taxonomy", required=True, help="CSV with taxonomy labels")
    p.add_argument("--out", required=True, help="Output CSV path")
    args = p.parse_args()

    companies = pd.read_csv(args.companies)
    taxonomy = pd.read_csv(args.taxonomy)

    clf = InsuranceTaxonomyClassifier().fit(companies, taxonomy)
    pred = clf.predict(companies)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
