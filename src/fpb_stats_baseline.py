"""Financial PhraseBank (Malo et al. 2014) statistics + baselines.
Usage: python -E -P fpb_stats_baseline.py DATA_DIR RESULTS_DIR
DATA_DIR holds sentences_{50agree,66agree,75agree,allagree}.csv (from fpb_download.py)."""
import sys, os, json, statistics, collections, importlib.util
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

data_dir, res_dir = sys.argv[1], sys.argv[2]
os.makedirs(res_dir, exist_ok=True)
LABELS = ["negative", "neutral", "positive"]
SEED = 42
R = {"source": "HF repo takala/financial_phrasebank -> data/FinancialPhraseBank-v1.0.zip (raw txt parsed; HF loading script unsupported in datasets 5.x)"}

# 1. label distribution
dist = {}
for cfg in ["50agree", "66agree", "75agree", "allagree"]:
    p = os.path.join(data_dir, f"sentences_{cfg}.csv")
    if not os.path.exists(p): continue
    df = pd.read_csv(p)
    c = df["label"].value_counts()
    dist[cfg] = {"n": int(len(df)), **{l: {"count": int(c.get(l, 0)), "pct": round(100 * c.get(l, 0) / len(df), 2)} for l in LABELS}}
R["label_distribution"] = dist

# 2. 75agree stats
df = pd.read_csv(os.path.join(data_dir, "sentences_75agree.csv"))
wl = df["sentence"].str.split().str.len()
dups = int(df["sentence"].duplicated().sum())
R["stats_75agree"] = {"n": int(len(df)), "mean_words": round(float(wl.mean()), 2), "median_words": float(wl.median()),
                      "max_words": int(wl.max()), "min_words": int(wl.min()), "exact_duplicate_sentences": dups,
                      "n_after_dedup": int(len(df) - dups),
                      "duplicates_with_conflicting_labels": int((df.groupby("sentence")["label"].nunique() > 1).sum())}

# 3. stratified 80/10/10 split after dedup
dd = df.drop_duplicates("sentence").reset_index()  # keep original index in column 'index'
idx = np.arange(len(dd)); y = dd["label"].values
tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
dv, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=SEED)
split = {"seed": SEED, "subset": "sentences_75agree", "dedup": True,
         "note": "indices are row positions (0-based) in sentences_75agree.csv after drop_duplicates on sentence",
         "train": sorted(int(dd.loc[i, "index"]) for i in tr), "dev": sorted(int(dd.loc[i, "index"]) for i in dv),
         "test": sorted(int(dd.loc[i, "index"]) for i in te)}
with open(os.path.join(data_dir, "split_seed42.json"), "w") as f: json.dump(split, f)
R["split"] = {k: {"n": len(v), **dict(collections.Counter(y[globals()[g]]))} for k, v, g in
              [("train", tr, "tr"), ("dev", dv, "dv"), ("test", te, "te")]}
Xtr, ytr, Xte, yte = dd.loc[tr, "sentence"], y[tr], dd.loc[te, "sentence"], y[te]

# 4. baselines
maj = collections.Counter(ytr).most_common(1)[0][0]
pm = np.array([maj] * len(yte))
R["majority_baseline"] = {"class": maj, "accuracy": round(accuracy_score(yte, pm), 4), "macro_f1": round(f1_score(yte, pm, average="macro"), 4)}
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                    LogisticRegression(class_weight="balanced", max_iter=2000, C=10.0, random_state=SEED))
clf.fit(Xtr, ytr); pr = clf.predict(Xte)
rep = classification_report(yte, pr, labels=LABELS, output_dict=True)
R["tfidf_logreg"] = {"accuracy": round(accuracy_score(yte, pr), 4), "macro_f1": round(f1_score(yte, pr, average="macro"), 4),
                     "per_class_f1": {l: round(rep[l]["f1-score"], 4) for l in LABELS},
                     "confusion_matrix_rows_true_cols_pred": {"labels": LABELS, "matrix": confusion_matrix(yte, pr, labels=LABELS).tolist()},
                     "dev_accuracy": round(accuracy_score(y[dv], clf.predict(dd.loc[dv, "sentence"])), 4)}

# 5. FinBERT zero-shot (only if transformers+torch already present)
if importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"):
    try:
        from transformers import pipeline
        pipe = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
        pf = [o["label"].lower() for o in pipe(list(Xte), batch_size=32)]
        R["finbert_zero_shot"] = {"accuracy": round(accuracy_score(yte, pf), 4), "macro_f1": round(f1_score(yte, pf, average="macro"), 4),
                                  "per_class_f1": {l: round(v, 4) for l, v in zip(LABELS, f1_score(yte, pf, labels=LABELS, average=None))}}
    except Exception as e:
        R["finbert_zero_shot"] = {"skipped": f"error: {type(e).__name__}: {e}"[:300]}
else:
    R["finbert_zero_shot"] = {"skipped": "transformers/torch not installed"}

with open(os.path.join(res_dir, "fpb_baseline.json"), "w") as f: json.dump(R, f, indent=2)

# markdown
L = ["# Financial PhraseBank: statistics and baselines", "", f"Source: {R['source']}", "", "## Label distribution",
     "| subset | n | negative | neutral | positive |", "|---|---|---|---|---|"]
for k, v in dist.items():
    L.append(f"| {k} | {v['n']} | " + " | ".join(f"{v[l]['count']} ({v[l]['pct']}%)" for l in LABELS) + " |")
s = R["stats_75agree"]
L += ["", "## 75agree sentence stats", f"- words/sentence: mean {s['mean_words']}, median {s['median_words']}, max {s['max_words']}, min {s['min_words']}",
      f"- exact duplicate sentences: {s['exact_duplicate_sentences']} (conflicting labels: {s['duplicates_with_conflicting_labels']}); n after dedup: {s['n_after_dedup']}",
      "", "## Split (seed 42, stratified 80/10/10, after dedup)", "| split | n | negative | neutral | positive |", "|---|---|---|---|---|"]
for k, v in R["split"].items(): L.append(f"| {k} | {v['n']} | {v['negative']} | {v['neutral']} | {v['positive']} |")
L += ["", "## Baselines (test split)", "| model | accuracy | macro-F1 |", "|---|---|---|",
      f"| majority ({maj}) | {R['majority_baseline']['accuracy']} | {R['majority_baseline']['macro_f1']} |",
      f"| TF-IDF(1-2g)+LogReg balanced | {R['tfidf_logreg']['accuracy']} | {R['tfidf_logreg']['macro_f1']} |"]
fb = R["finbert_zero_shot"]
L.append(f"| FinBERT zero-shot | {fb['accuracy']} | {fb['macro_f1']} |" if "accuracy" in fb else f"| FinBERT zero-shot | skipped | {fb['skipped']} |")
L += ["", "TF-IDF+LogReg per-class F1: " + ", ".join(f"{l} {v}" for l, v in R["tfidf_logreg"]["per_class_f1"].items()),
      "", "Confusion matrix (rows=true, cols=pred; negative/neutral/positive):", "```"]
L += [str(r) for r in R["tfidf_logreg"]["confusion_matrix_rows_true_cols_pred"]["matrix"]] + ["```"]
with open(os.path.join(res_dir, "fpb_baseline.md"), "w", encoding="utf-8") as f: f.write("\n".join(L) + "\n")
print(json.dumps({k: R[k] for k in R if k != "label_distribution"}, indent=1))
print(json.dumps(dist))
