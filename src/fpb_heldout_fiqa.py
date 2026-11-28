"""Held-out sentiment check on FiQA 2018 headlines (outside Financial PhraseBank).
Scores TF-IDF+LogReg (trained on our PhraseBank train split), released FinBERT, and our fine-tuned FinBERT.
Usage: python -E -P fpb_heldout_fiqa.py FPB_DIR FIQA_DIR RUNS_DIR RESULTS_DIR"""
import sys, os, json
import numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

fpb_dir, fiqa_dir, runs_dir, res_dir = sys.argv[1:5]
LABELS = ["negative", "neutral", "positive"]; L2I = {l: i for i, l in enumerate(LABELS)}
dev = "cuda" if torch.cuda.is_available() else "cpu"
# FiQA: all splits, headlines only, thresholded score
fq = pd.concat([pd.read_csv(os.path.join(fiqa_dir, f)) for f in ["train.csv", "validation.csv", "test.csv"]])
fq = fq[fq["format"] == "headline"].drop_duplicates("sentence").reset_index(drop=True)
TH = 0.2
fq["gold"] = np.where(fq.sentiment_score > TH, "positive", np.where(fq.sentiment_score < -TH, "negative", "neutral"))
X = fq["sentence"].tolist(); y = [L2I[g] for g in fq["gold"]]
R = {"source": "pauri32/fiqa-2018 (FiQA 2018 task 1), headlines only, all splits, deduplicated", "threshold": TH,
     "n": len(fq), "label_counts": fq["gold"].value_counts().to_dict()}
def metrics(p):
    return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4),
            "per_class_f1": {l: round(float(v), 4) for l, v in zip(LABELS, f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0))},
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}
# TF-IDF baseline trained on our PhraseBank train split
df = pd.read_csv(os.path.join(fpb_dir, "sentences_75agree.csv")); sp = json.load(open(os.path.join(fpb_dir, "split_seed42.json")))
tr = df.loc[sp["train"]]
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0))
clf.fit(tr["sentence"], [L2I[l] for l in tr["label"]]); R["tfidf_logreg"] = metrics(clf.predict(X))
# FinBERT variants
name = "ProsusAI/finbert"; tok = AutoTokenizer.from_pretrained(name)
@torch.no_grad()
def predict(m):
    m.eval(); out = []
    for i in range(0, len(X), 64):
        b = {k: v.to(dev) for k, v in tok(X[i:i+64], truncation=True, max_length=96, padding=True, return_tensors="pt").items()}
        out.extend(m(**b).logits.argmax(-1).tolist())
    return out
m = AutoModelForSequenceClassification.from_pretrained(name).to(dev)
hf2ours = {i: L2I[l.lower()] for i, l in m.config.id2label.items()}
R["finbert_released"] = metrics([hf2ours[i] for i in predict(m)])
m.load_state_dict(torch.load(os.path.join(runs_dir, "finbert_best.pt"), map_location=dev))
p_ft = [hf2ours[i] for i in predict(m)]; R["finbert_finetuned"] = metrics(p_ft)
fq["finetuned"] = [LABELS[i] for i in p_ft]
fq[["sentence", "sentiment_score", "gold", "finetuned"]].to_csv(os.path.join(res_dir, "fiqa_heldout_preds.csv"), index=False)
json.dump(R, open(os.path.join(res_dir, "fiqa_heldout.json"), "w"), indent=2)
print(json.dumps(R, indent=1))
