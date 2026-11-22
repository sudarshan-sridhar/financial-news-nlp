"""ECTSum dataset statistics + TF-IDF / Lead-k extractive baselines.
Usage: python -P -E ectsum_stats_baseline.py <data/final dir> <results dir> [--sample N] [--seed S]
"""
import argparse, json, math, os, re, statistics, sys, time
from collections import Counter

SPLITS = ["train", "val", "test"]

def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()

def load_split(root, split):
    d_e = os.path.join(root, split, "ects"); d_s = os.path.join(root, split, "gt_summaries")
    names = sorted(n for n in os.listdir(d_e) if os.path.exists(os.path.join(d_s, n)))
    return [(n, read(os.path.join(d_e, n)), read(os.path.join(d_s, n))) for n in names]

def wc(t): return len(t.split())

def stats(vals):
    return {"n": len(vals), "mean": round(statistics.mean(vals), 1), "median": statistics.median(vals),
            "min": min(vals), "max": max(vals)}

def sentences(text):
    # ECTSum transcripts are one sentence per line; fall back to regex split for long lines
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", line) if s.strip())
    return out

TOK = re.compile(r"[a-z0-9$%.]+")
def toks(s): return [t.strip(".") for t in TOK.findall(s.lower()) if t.strip(".")]

def tfidf_summary(sents, idf, k=4, max_words=50):
    scored = []
    for i, s in enumerate(sents):
        ts = toks(s)
        if len(ts) < 4: continue
        tf = Counter(ts)
        score = sum((1 + math.log(c)) * idf.get(t, idf["__default__"]) for t, c in tf.items())
        scored.append((score / math.sqrt(len(ts)), i))  # length-normalised
    scored.sort(reverse=True)
    chosen, words = [], 0
    for _, i in scored:
        if len(chosen) >= k or words >= max_words: break
        chosen.append(i); words += wc(sents[i])
    return "\n".join(sents[i] for i in sorted(chosen))

def lead_summary(sents, k=4): return "\n".join(sents[:k])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root"); ap.add_argument("results_dir")
    ap.add_argument("--sample", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    data = {s: load_split(a.data_root, s) for s in SPLITS}

    res = {"counts": {}, "transcript_words": {}, "summary_words": {}, "over_1024": {}, "bullets_per_summary": {}}
    all_t, all_s, all_b = [], [], []
    for s in SPLITS:
        tw = [wc(e) for _, e, _ in data[s]]; sw = [wc(g) for _, _, g in data[s]]
        bl = [len([l for l in g.splitlines() if l.strip()]) for _, _, g in data[s]]
        res["counts"][s] = len(data[s]); res["transcript_words"][s] = stats(tw); res["summary_words"][s] = stats(sw)
        res["over_1024"][s] = {"n": sum(x > 1024 for x in tw), "share": round(sum(x > 1024 for x in tw) / len(tw), 4)}
        res["bullets_per_summary"][s] = round(statistics.mean(bl), 2)
        all_t += tw; all_s += sw; all_b += bl
    res["counts"]["total"] = sum(res["counts"][s] for s in SPLITS)
    res["transcript_words"]["overall"] = stats(all_t); res["summary_words"]["overall"] = stats(all_s)
    res["over_1024"]["overall"] = {"n": sum(x > 1024 for x in all_t), "share": round(sum(x > 1024 for x in all_t) / len(all_t), 4)}
    res["bullets_per_summary"]["overall"] = round(statistics.mean(all_b), 2)

    # IDF from train transcripts (sentence-level documents)
    df = Counter(); nd = 0
    for _, e, _ in data["train"]:
        for s in sentences(e):
            df.update(set(toks(s))); nd += 1
    idf = {t: math.log((nd + 1) / (c + 1)) + 1 for t, c in df.items()}
    idf["__default__"] = math.log(nd + 1) + 1

    test = data["test"]
    if a.sample and a.sample < len(test):
        import random; random.Random(a.seed).shuffle(test); test = test[:a.sample]
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    agg = {"tfidf_top4": Counter(), "lead4": Counter()}
    for _, e, g in test:
        sents = sentences(e)
        for name, hyp in (("tfidf_top4", tfidf_summary(sents, idf)), ("lead4", lead_summary(sents))):
            sc = scorer.score(g, hyp)
            for m in sc: agg[name][m] += sc[m].fmeasure
    res["rouge_f1_test"] = {"n_docs": len(test), "sampled": bool(a.sample and a.sample < len(data["test"])),
                            **{k: {m: round(100 * v / len(test), 2) for m, v in c.items()} for k, c in agg.items()}}
    res["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(a.results_dir, exist_ok=True)
    with open(os.path.join(a.results_dir, "ectsum_baseline.json"), "w") as f: json.dump(res, f, indent=2)
    L = ["# ECTSum statistics and extractive baselines", "",
         "## Pairs per split", "", "| split | n |", "|---|---|"] + [f"| {s} | {res['counts'][s]} |" for s in SPLITS + ["total"]]
    for key, title in (("transcript_words", "Transcript word counts"), ("summary_words", "Summary word counts")):
        L += ["", f"## {title}", "", "| split | mean | median | min | max |", "|---|---|---|---|---|"]
        L += [f"| {s} | {d['mean']} | {d['median']} | {d['min']} | {d['max']} |" for s, d in res[key].items()]
    L += ["", "## Transcripts over 1024 whitespace tokens", "", "| split | n | share |", "|---|---|---|"]
    L += [f"| {s} | {d['n']} | {d['share']:.1%} |" for s, d in res["over_1024"].items()]
    L += ["", "## Mean bullet lines per summary", "", "| split | bullets |", "|---|---|"]
    L += [f"| {s} | {v} |" for s, v in res["bullets_per_summary"].items()]
    r = res["rouge_f1_test"]
    L += ["", f"## ROUGE F1 on test (n={r['n_docs']}{', sampled' if r['sampled'] else ''})", "",
          "| system | R-1 | R-2 | R-L |", "|---|---|---|---|"]
    L += [f"| {k} | {r[k]['rouge1']} | {r[k]['rouge2']} | {r[k]['rougeL']} |" for k in ("lead4", "tfidf_top4")]
    L += ["", f"Wall time: {res['wall_seconds']} s"]
    with open(os.path.join(a.results_dir, "ectsum_baseline.md"), "w") as f: f.write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
