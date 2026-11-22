"""Fine-tune BART-base on ECTSum (head truncation to 1024 tokens), evaluate ROUGE + BERTScore on test.
Usage: python -E -P ectsum_bart.py DATA_FINAL_DIR RESULTS_DIR RUNS_DIR [--epochs 2] [--max_test N]"""
import sys, os, json, time, argparse, re
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from rouge_score import rouge_scorer

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("res_dir"); ap.add_argument("runs_dir")
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--max_in", type=int, default=1024); ap.add_argument("--max_out", type=int, default=128)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max_test", type=int, default=0); ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--bertscore", action="store_true")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def read(p):
    with open(p, encoding="utf-8", errors="replace") as f: return f.read()
def load(split):
    de, ds = os.path.join(a.root, split, "ects"), os.path.join(a.root, split, "gt_summaries")
    names = sorted(n for n in os.listdir(de) if os.path.exists(os.path.join(ds, n)))
    return [(n, " ".join(read(os.path.join(de, n)).split()), read(os.path.join(ds, n)).strip()) for n in names]
D = {s: load(s) for s in ["train", "val", "test"]}
if a.max_train: D["train"] = D["train"][:a.max_train]
if a.max_test: D["test"] = D["test"][:a.max_test]
name = "facebook/bart-base"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).to(dev)

def batch(items):
    x = tok([t for _, t, _ in items], truncation=True, max_length=a.max_in, padding=True, return_tensors="pt")
    y = tok(text_target=[s.replace("\n", " ") for _, _, s in items], truncation=True, max_length=a.max_out, padding=True, return_tensors="pt")
    lab = y["input_ids"].clone(); lab[lab == tok.pad_token_id] = -100
    return {k: v.to(dev) for k, v in x.items()}, lab.to(dev)

@torch.no_grad()
def generate(items, bs=8):
    model.eval(); outs = []
    for i in range(0, len(items), bs):
        x, _ = batch(items[i:i+bs])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            g = model.generate(**x, num_beams=4, max_new_tokens=a.max_out, min_new_tokens=16, no_repeat_ngram_size=3, length_penalty=1.0)
        outs.extend(tok.batch_decode(g, skip_special_tokens=True))
    return outs
sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
def rouge(preds, refs):
    r = {k: [] for k in ["rouge1", "rouge2", "rougeL"]}
    for p, g in zip(preds, refs):
        s = sc.score(g, p)
        for k in r: r[k].append(s[k].fmeasure)
    return {k: round(100 * float(np.mean(v)), 2) for k, v in r.items()}

R = {"model": name, "device": dev, "seed": a.seed, "long_input": f"head truncation to {a.max_in} tokens", "n_train": len(D["train"]), "n_val": len(D["val"]), "n_test": len(D["test"])}
# zero-shot (pretrained, no fine-tune) on a small val sample for reference
t0 = time.time()
zs = generate(D["val"][:50]); R["pretrained_zero_shot_val50"] = rouge(zs, [s for _, _, s in D["val"][:50]])
print("pretrained zero-shot val50", R["pretrained_zero_shot_val50"], flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(D["train"]) + a.bs * a.accum - 1) // (a.bs * a.accum))
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = None
hist, best, best_state = [], -1, None
t0 = time.time()
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
        if bi % 100 == 0: print(f"ep{ep+1} step{bi} loss {tot/max(n,1):.4f} {time.time()-t0:.0f}s", flush=True)
    vs = D["val"][:100]; pv = generate(vs); rv = rouge(pv, [s for _, _, s in vs])
    hist.append({"epoch": ep + 1, "train_loss": round(tot / n, 4), "val100_rouge": rv, "elapsed_s": round(time.time() - t0)})
    print(hist[-1], flush=True)
    if rv["rougeL"] > best:
        best = rv["rougeL"]; R["best_epoch"] = ep + 1; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
os.makedirs(a.runs_dir, exist_ok=True); torch.save(best_state, os.path.join(a.runs_dir, "bart_base_ectsum_best.pt"))
t1 = time.time(); preds = generate(D["test"]); refs = [s for _, _, s in D["test"]]
R["test_rouge"] = rouge(preds, refs); R["test_gen_seconds"] = round(time.time() - t1); R["train_seconds"] = round(t1 - t0)
R["pred_words_mean"] = round(float(np.mean([len(p.split()) for p in preds])), 1)
print("TEST ROUGE", R["test_rouge"], flush=True)
if a.bertscore:
    from bert_score import score as bs
    P, Rr, F = bs(preds, refs, lang="en", rescale_with_baseline=False, device=dev)
    R["test_bertscore_f1"] = round(float(F.mean()), 4)
    print("BERTScore F1", R["test_bertscore_f1"], flush=True)
R["history"] = hist; R["hparams"] = {"epochs": a.epochs, "lr": a.lr, "batch_size": a.bs, "grad_accum": a.accum, "effective_batch": a.bs * a.accum, "max_in": a.max_in, "max_out": a.max_out, "beams": 4, "no_repeat_ngram": 3, "precision": "bf16 autocast"}
json.dump(R, open(os.path.join(a.res_dir, "ectsum_bart.json"), "w"), indent=2)
json.dump([{"doc": n, "reference": s, "bart": p} for (n, _, s), p in zip(D["test"], preds)], open(os.path.join(a.res_dir, "ectsum_bart_test_preds.json"), "w"), indent=1)
print("done")
