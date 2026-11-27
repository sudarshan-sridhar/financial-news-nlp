"""BART-base on ECTSum, Checkpoint 3 ablations.
  --select head|salient   long-input strategy (salient = TF-IDF sentence selection against training summaries)
  --ckpt PATH             skip training, load weights (for decoding sweeps)
  --sweep                 decoding sweep (min_new_tokens x length_penalty) on val100, best setting applied to test
Usage: python -E -P ectsum_bart_v2.py DATA_FINAL_DIR RESULTS_DIR RUNS_DIR --tag NAME [...]"""
import sys, os, json, time, argparse, re, itertools
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("res_dir"); ap.add_argument("runs_dir")
ap.add_argument("--tag", required=True); ap.add_argument("--select", default="head", choices=["head", "salient"])
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--max_in", type=int, default=1024); ap.add_argument("--max_out", type=int, default=128)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--ckpt", default=""); ap.add_argument("--sweep", action="store_true")
ap.add_argument("--min_new", type=int, default=16); ap.add_argument("--len_pen", type=float, default=1.0); ap.add_argument("--bertscore", action="store_true")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def read(p):
    with open(p, encoding="utf-8", errors="replace") as f: return f.read()
def load(split):
    de, ds = os.path.join(a.root, split, "ects"), os.path.join(a.root, split, "gt_summaries")
    names = sorted(n for n in os.listdir(de) if os.path.exists(os.path.join(ds, n)))
    return [(n, read(os.path.join(de, n)), read(os.path.join(ds, n)).strip()) for n in names]
D = {s: load(s) for s in ["train", "val", "test"]}
name = "facebook/bart-base"
tok = AutoTokenizer.from_pretrained(name)

SENT = re.compile(r"(?<=[.!?])\s+")
def sents(t): return [s.strip() for s in SENT.split(" ".join(t.split())) if s.strip()]
if a.select == "salient":
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
    vec.fit([s for _, _, s in D["train"]] + [t for _, t, _ in D["train"]])
    ref_centroid = np.asarray(vec.transform([s for _, _, s in D["train"]]).mean(axis=0)).ravel()
    ref_centroid /= (np.linalg.norm(ref_centroid) + 1e-9)
    budget_words = 760
    def prepare(t):
        ss = sents(t)
        if not ss: return t
        M = vec.transform(ss); norms = np.sqrt(M.multiply(M).sum(axis=1)).A.ravel() + 1e-9
        sc = (M @ ref_centroid) / norms
        order = np.argsort(-sc); keep, used = set(), 0
        for i in order:
            w = len(ss[i].split())
            if used + w > budget_words and keep: continue
            keep.add(i); used += w
            if used >= budget_words: break
        return " ".join(ss[i] for i in sorted(keep))
else:
    def prepare(t): return " ".join(t.split())
t0 = time.time()
for s in D: D[s] = [(n, prepare(t), g) for n, t, g in D[s]]
print("prepared inputs", a.select, round(time.time() - t0), "s", flush=True)

model = AutoModelForSeq2SeqLM.from_pretrained(name).to(dev)
def batch(items):
    x = tok([t for _, t, _ in items], truncation=True, max_length=a.max_in, padding=True, return_tensors="pt")
    y = tok(text_target=[s.replace("\n", " ") for _, _, s in items], truncation=True, max_length=a.max_out, padding=True, return_tensors="pt")
    lab = y["input_ids"].clone(); lab[lab == tok.pad_token_id] = -100
    return {k: v.to(dev) for k, v in x.items()}, lab.to(dev)
@torch.no_grad()
def generate(items, min_new, len_pen, bs=8):
    model.eval(); outs = []
    for i in range(0, len(items), bs):
        x, _ = batch(items[i:i+bs])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            g = model.generate(**x, num_beams=4, max_new_tokens=a.max_out, min_new_tokens=min_new, no_repeat_ngram_size=3, length_penalty=len_pen)
        outs.extend(tok.batch_decode(g, skip_special_tokens=True))
    return outs
sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
def rouge(preds, refs):
    r = {k: [] for k in ["rouge1", "rouge2", "rougeL"]}
    for p, g in zip(preds, refs):
        s = sc.score(g, p)
        for k in r: r[k].append(s[k].fmeasure)
    return {k: round(100 * float(np.mean(v)), 2) for k, v in r.items()}

R = {"tag": a.tag, "model": name, "select": a.select, "seed": a.seed, "n_train": len(D["train"]), "n_test": len(D["test"])}
if a.ckpt:
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)); R["ckpt"] = a.ckpt
else:
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(D["train"]) + a.bs * a.accum - 1) // (a.bs * a.accum))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    hist, best, best_state = [], -1, None; t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(len(D["train"])); tot = 0; n = 0; opt.zero_grad()
        for bi, i in enumerate(range(0, len(perm), a.bs)):
            items = [D["train"][j] for j in perm[i:i+a.bs]]
            x, lab = batch(items)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = model(**x, labels=lab).loss / a.accum
            loss.backward(); tot += loss.item() * a.accum * len(items); n += len(items)
            if (bi + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad()
        vs = D["val"][:100]; rv = rouge(generate(vs, a.min_new, a.len_pen), [s for _, _, s in vs])
        hist.append({"epoch": ep + 1, "train_loss": round(tot / n, 4), "val100_rouge": rv, "elapsed_s": round(time.time() - t0)}); print(hist[-1], flush=True)
        if rv["rougeL"] > best:
            best = rv["rougeL"]; R["best_epoch"] = ep + 1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); R["history"] = hist; R["train_seconds"] = round(time.time() - t0)
    os.makedirs(a.runs_dir, exist_ok=True); torch.save(best_state, os.path.join(a.runs_dir, f"bart_{a.tag}.pt"))

min_new, len_pen = a.min_new, a.len_pen
if a.sweep:
    vs = D["val"][:100]; refs = [s for _, _, s in vs]; grid = []
    for mn, lp in itertools.product([16, 32, 48, 64], [1.0, 1.5, 2.0]):
        p = generate(vs, mn, lp); r = rouge(p, refs); r.update({"min_new": mn, "len_pen": lp, "pred_words": round(float(np.mean([len(x.split()) for x in p])), 1)})
        grid.append(r); print(r, flush=True)
    R["sweep_val100"] = grid
    bestg = max(grid, key=lambda g: g["rougeL"]); min_new, len_pen = bestg["min_new"], bestg["len_pen"]; R["sweep_best"] = bestg
t1 = time.time(); preds = generate(D["test"], min_new, len_pen); refs = [s for _, _, s in D["test"]]
R["decoding"] = {"min_new": min_new, "len_pen": len_pen, "beams": 4, "no_repeat_ngram": 3}
R["test_rouge"] = rouge(preds, refs); R["test_gen_seconds"] = round(time.time() - t1)
R["pred_words_mean"] = round(float(np.mean([len(p.split()) for p in preds])), 1)
print("TEST", a.tag, R["test_rouge"], flush=True)
if a.bertscore:
    from bert_score import score as bs
    P, Rr, F = bs(preds, refs, lang="en", rescale_with_baseline=False, device=dev); R["test_bertscore_f1"] = round(float(F.mean()), 4)
    print("BERTScore", R["test_bertscore_f1"], flush=True)
json.dump(R, open(os.path.join(a.res_dir, f"ectsum_bart_{a.tag}.json"), "w"), indent=2)
json.dump([{"doc": n, "reference": s, "bart": p} for (n, _, s), p in zip(D["test"], preds)], open(os.path.join(a.res_dir, f"ectsum_bart_{a.tag}_test_preds.json"), "w"), indent=1)
print("done", a.tag)
