"""FinBERT zero-shot + fine-tune on the saved seed-42 Financial PhraseBank split.
Usage: python -E -P fpb_finbert.py DATA_DIR RESULTS_DIR RUNS_DIR [--epochs 3]"""
import sys, os, json, time, argparse
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser(); ap.add_argument("data_dir"); ap.add_argument("res_dir"); ap.add_argument("runs_dir")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_len", type=int, default=96); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
LABELS = ["negative", "neutral", "positive"]; L2I = {l: i for i, l in enumerate(LABELS)}
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(os.path.join(a.data_dir, "sentences_75agree.csv"))
sp = json.load(open(os.path.join(a.data_dir, "split_seed42.json")))
D = {k: (df.loc[sp[k], "sentence"].tolist(), [L2I[x] for x in df.loc[sp[k], "label"]]) for k in ["train", "dev", "test"]}
name = "ProsusAI/finbert"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).to(dev)
# ProsusAI/finbert label order: 0 positive, 1 negative, 2 neutral
hf2ours = {model.config.label2id[l.lower()] if l.lower() in model.config.label2id else i: L2I[l.lower()] for i, l in model.config.id2label.items()}
hf2ours = {i: L2I[l.lower()] for i, l in model.config.id2label.items()}

def enc(texts): return tok(texts, truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
@torch.no_grad()
def predict(m, texts, bs=64):
    m.eval(); out = []
    for i in range(0, len(texts), bs):
        b = {k: v.to(dev) for k, v in enc(texts[i:i+bs]).items()}
        out.extend(m(**b).logits.argmax(-1).tolist())
    return out
def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0,1,2], zero_division=0)
    return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, average="macro"), 4),
            "macro_precision": round(float(pr.mean()), 4), "macro_recall": round(float(rc.mean()), 4),
            "per_class_f1": {l: round(float(v), 4) for l, v in zip(LABELS, f1)},
            "confusion": confusion_matrix(y, p, labels=[0,1,2]).tolist()}
R = {"model": name, "device": dev, "seed": a.seed, "max_len": a.max_len}
t0 = time.time()
# zero-shot (pretrained head, mapped to our label order)
for k in ["dev", "test"]:
    p = [hf2ours[i] for i in predict(model, D[k][0])]
    R[f"zero_shot_{k}"] = metrics(D[k][1], p)
    if k == "test": R["zero_shot_test_preds"] = p
R["zero_shot_seconds"] = round(time.time() - t0, 1)
print("zero-shot", json.dumps({k: R[k] for k in R if k.startswith("zero_shot_") and not k.endswith("preds")}), flush=True)

# fine-tune: re-init a model, remap its head to our order by re-labeling targets to HF order
hf_target = {v: k for k, v in hf2ours.items()}
ytr_hf = [hf_target[y] for y in D["train"][1]]
model = AutoModelForSequenceClassification.from_pretrained(name).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(ytr_hf) + a.bs - 1) // a.bs)
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
best, best_state, hist = -1, None, []
t0 = time.time()
for ep in range(a.epochs):
    model.train(); perm = np.random.permutation(len(ytr_hf)); tot = 0
    for i in range(0, len(perm), a.bs):
        idx = perm[i:i+a.bs]
        b = {k: v.to(dev) for k, v in enc([D["train"][0][j] for j in idx]).items()}
        y = torch.tensor([ytr_hf[j] for j in idx], device=dev)
        loss = model(**b, labels=y).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); tot += loss.item() * len(idx)
    pd_ = [hf2ours[i] for i in predict(model, D["dev"][0])]
    m = metrics(D["dev"][1], pd_); hist.append({"epoch": ep + 1, "train_loss": round(tot / len(ytr_hf), 4), "dev_macro_f1": m["macro_f1"], "dev_acc": m["accuracy"]})
    print(hist[-1], flush=True)
    if m["macro_f1"] > best:
        best = m["macro_f1"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; R["finetune_dev"] = m; R["best_epoch"] = ep + 1
model.load_state_dict(best_state)
p = [hf2ours[i] for i in predict(model, D["test"][0])]
R["finetune_test"] = metrics(D["test"][1], p); R["finetune_test_preds"] = p
R["finetune_history"] = hist; R["finetune_seconds"] = round(time.time() - t0, 1)
R["hparams"] = {"epochs": a.epochs, "lr": a.lr, "batch_size": a.bs, "max_len": a.max_len, "warmup": 0.1, "weight_decay": 0.01}
os.makedirs(a.runs_dir, exist_ok=True); torch.save(best_state, os.path.join(a.runs_dir, f"finbert_best{a.tag}.pt"))
# save test predictions with text for error analysis
pd.DataFrame({"sentence": D["test"][0], "gold": [LABELS[y] for y in D["test"][1]], "zero_shot": [LABELS[y] for y in R["zero_shot_test_preds"]], "finetuned": [LABELS[y] for y in p]}).to_csv(os.path.join(a.res_dir, f"fpb_finbert{a.tag}_test_preds.csv"), index=False)
json.dump(R, open(os.path.join(a.res_dir, f"fpb_finbert{a.tag}.json"), "w"), indent=2)
print("finetune test", json.dumps(R["finetune_test"]))
