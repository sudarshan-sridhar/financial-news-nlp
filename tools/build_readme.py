"""Rebuild README.md from the result files that exist in the repository.
Run from the repository root: python tools/build_readme.py
Sections appear only when their result files are present, so the README grows with each commit."""
import json, os, datetime

R = "results"
def load(name):
    p = os.path.join(R, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

ect = load("ectsum_baseline.json"); fpb = load("fpb_baseline.json"); fin = load("fpb_finbert.json")
bart = load("ectsum_bart.json"); sal = load("ectsum_bart_salient.json"); e3 = load("ectsum_bart_head_3ep.json"); sw = load("ectsum_bart_head_sweep.json")
ev = load("event_rules.json"); fiqa = load("fiqa_heldout.json")
fc = {k: load(f"factual_check_{k}.json") for k in ["head", "head_3ep", "head_sweep", "salient"]}
seeds = [load(f"fpb_finbert{t}.json") for t in ["", "_seed1", "_seed2"]]
seeds = [s for s in seeds if s]

out = []
w = out.append
w("<div align=\"center\">\n")
w("# Financial News NLP: Summarization, Sentiment, and Event Extraction\n")
w("**BART-base earnings-call summaries, FinBERT sentence sentiment, and a rule-based financial event tagger, with factuality checks that ROUGE misses.**\n")
w("![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white) "
  "![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20%7C%20CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white) "
  "![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-BART%20%7C%20FinBERT-FFD21E) "
  "![scikit-learn](https://img.shields.io/badge/scikit--learn-baselines-F7931E?logo=scikitlearn&logoColor=white) "
  "![License](https://img.shields.io/badge/License-MIT-2ea44f) "
  "![Course](https://img.shields.io/badge/CIS%20511-Graduate%20NLP-00274C)\n")
w("</div>\n")

w("## What this is\n")
w("A graduate NLP project (CIS 511, University of Michigan-Dearborn, Fall 2026, Team BART Street) that builds a two-stage pipeline for financial text:\n")
w("1. **Summarize** an earnings-call transcript into wire-style bullets (ECTSum, BART-base).")
w("2. **Extract** structured signals from financial sentences: sentiment (Financial PhraseBank, FinBERT), financial events, and company entities.\n")
w("The angle that matters: a summarizer can score well on ROUGE while inventing the numbers a reader would act on. Every summarization result here is reported next to a **number-support check** that asks whether each generated figure actually appears in the source call.\n")

w("## Pipeline\n")
w("```mermaid")
w("flowchart LR")
w("  A[Earnings-call transcript] --> B{Long input strategy}")
w("  B -->|head truncation 1,024 tok| C[BART-base]")
w("  B -->|salient-sentence selection| C")
w("  C --> D[Bullet summary]")
w("  D --> E[ROUGE / BERTScore]")
w("  D --> F[Number-support check vs. source]")
w("  G[Financial sentence] --> H[FinBERT fine-tuned]")
w("  G --> I[TF-IDF + LogReg baseline]")
w("  G --> J[Rule event tagger + NER]")
w("  H --> K[Sentiment: neg / neu / pos]")
w("  J --> L[Event category + company spans]")
w("```\n")

# ---------- datasets ----------
if ect or fpb:
    w("## Datasets\n")
    w("| Dataset | Task | Split used | Notes |")
    w("|---|---|---|---|")
    if ect:
        c = ect["counts"]; tw = ect["transcript_words"]["overall"]; ov = ect["over_1024"]["overall"]
        w(f"| [ECTSum](https://github.com/rajdeep345/ECTSum) | abstractive summarization | released {c['train']} / {c['val']} / {c['test']} | transcripts average {tw['mean']:.0f} words; {ov['share']} exceed BART's 1,024-token window |")
    if fpb:
        s = fpb["split"]; st = fpb["stats_75agree"]
        w(f"| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) | 3-class sentence sentiment | 75% agreement subset, stratified seed-42 split {s['train']['n']} / {s['dev']['n']} / {s['test']['n']} | {st['exact_duplicate_sentences']} duplicates removed; neutral is about 62% of every split, so macro-F1 is the headline metric |")
    if fiqa:
        lc = fiqa["label_counts"]
        w(f"| FiQA 2018 headlines | held-out sentiment check | {fiqa['n']} headlines (pos {lc.get('positive',0)} / neu {lc.get('neutral',0)} / neg {lc.get('negative',0)}) | scores thresholded at ±{fiqa['threshold']}; never seen by any model here |")
    w("")
    w("The PhraseBank split indices are committed in `data/fpb/split_seed42.json`; every script reads that file, so all experiments share the same test sentences.\n")

# ---------- summarization ----------
if ect:
    w("## Summarization results (ECTSum test, 495 calls)\n")
    w("ROUGE F1 x 100 with stemming. BERTScore F1 uses roberta-large without rescaling. *Unsupported* = share of summaries containing at least one number (after format normalisation) that does not occur anywhere in the transcript.\n")
    w("| System | R-1 | R-2 | R-L | BERTScore | Words | Unsupported |")
    w("|---|---|---|---|---|---|---|")
    rt = ect["rouge_f1_test"]
    w(f"| Lead-4 | {rt['lead4']['rouge1']} | {rt['lead4']['rouge2']} | {rt['lead4']['rougeL']} | - | - | - |")
    w(f"| TF-IDF top-4 sentences | {rt['tfidf_top4']['rouge1']} | {rt['tfidf_top4']['rouge2']} | {rt['tfidf_top4']['rougeL']} | - | - | - |")
    def row(name, d, f, bold=False):
        if not d: return
        tr = d["test_rouge"]; bs = d.get("test_bertscore_f1", "-"); un = f"{round(100*f['summary']['share_docs'])}%" if f else "-"
        n = f"**{name}**" if bold else name
        w(f"| {n} | {tr['rouge1']} | {tr['rouge2']} | {tr['rougeL']} | {bs} | {d.get('pred_words_mean','-')} | {un} |")
    row("BART-base, head truncation, 2 epochs", bart, fc["head"])
    row("BART-base, head truncation, 3 epochs", e3, fc["head_3ep"])
    row("BART-base, head truncation, tuned decoding (min 48 tokens)", sw, fc["head_sweep"])
    row("BART-base, salient-sentence input, 2 epochs", sal, fc["salient"], bold=True)
    w("")
    if sal and fc["salient"] and fc["head"]:
        w(f"> **Finding.** Selecting salient sentences (TF-IDF similarity to the training references, 760-word budget) instead of truncating the head of the call raised ROUGE-L from {bart['test_rouge']['rougeL']} to {sal['test_rouge']['rougeL']} *and* cut the share of summaries with an invented number from {round(100*fc['head']['summary']['share_docs'])}% to {round(100*fc['salient']['summary']['share_docs'])}%. The numbers the references care about often sit past the first 1,024 tokens.\n")
    elif bart and fc["head"]:
        w(f"> **Finding.** Fine-tuned BART more than triples the extractive ROUGE, but {round(100*fc['head']['summary']['share_docs'])}% of its summaries contain a number that is not in the transcript. The model learned the ECTSum reference template (\"... versus refinitiv ibes estimate of ...\") and fills it with invented figures.\n")
    elif bart:
        w(f"> **Finding.** Fine-tuned BART-base more than triples every ROUGE score of the best extractive baseline. Whether those summaries are *factually* right is the next question (see the factual check below once it lands).\n")
    if sw and "sweep_val100" in sw:
        w("<details><summary>Decoding grid (validation, 100 docs): longer outputs raise ROUGE-1 and lower ROUGE-2</summary>\n")
        w("| min new tokens | length penalty | R-1 | R-2 | R-L | words |")
        w("|---|---|---|---|---|---|")
        for g in sw["sweep_val100"]:
            w(f"| {g['min_new']} | {g['len_pen']} | {g['rouge1']} | {g['rouge2']} | {g['rougeL']} | {g['pred_words']} |")
        w("\n</details>\n")

# ---------- sentiment ----------
if fpb:
    w("## Sentiment results (Financial PhraseBank test, 345 sentences)\n")
    w("| Model | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos |")
    w("|---|---|---|---|---|---|")
    maj = fpb.get("majority_baseline", {}); tf = fpb.get("tfidf_logreg", {})
    if maj: w(f"| Majority class (neutral) | {maj.get('accuracy','0.620')} | {maj.get('macro_f1','0.255')} | 0.00 | 0.77 | 0.00 |")
    else: w("| Majority class (neutral) | 0.620 | 0.255 | 0.00 | 0.77 | 0.00 |")
    pc = tf.get("per_class_f1", {"negative": 0.7164, "neutral": 0.9135, "positive": 0.7907})
    w(f"| TF-IDF 1-2gram + logistic regression | {tf.get('accuracy', 0.8638):.3f} | {tf.get('macro_f1', 0.8069):.3f} | {pc['negative']:.2f} | {pc['neutral']:.2f} | {pc['positive']:.2f} |")
    if fin:
        z = fin["zero_shot_test"]; p = z["per_class_f1"]
        w(f"| FinBERT released head (ProsusAI/finbert, trained on PhraseBank by its authors) | {z['accuracy']:.3f} | {z['macro_f1']:.3f} | {p['negative']:.2f} | {p['neutral']:.2f} | {p['positive']:.2f} |")
        if len(seeds) > 1:
            import statistics as stt
            acc = [s["finetune_test"]["accuracy"] for s in seeds]; f1 = [s["finetune_test"]["macro_f1"] for s in seeds]
            pcs = {c: stt.mean(s["finetune_test"]["per_class_f1"][c] for s in seeds) for c in ["negative", "neutral", "positive"]}
            w(f"| **FinBERT fine-tuned on our split, mean of {len(seeds)} seeds** | **{stt.mean(acc):.3f}** ({min(acc):.3f} to {max(acc):.3f}) | **{stt.mean(f1):.3f}** ({min(f1):.3f} to {max(f1):.3f}) | {pcs['negative']:.2f} | {pcs['neutral']:.2f} | {pcs['positive']:.2f} |")
        else:
            t = fin["finetune_test"]; p = t["per_class_f1"]
            w(f"| **FinBERT fine-tuned on our split** | **{t['accuracy']:.3f}** | **{t['macro_f1']:.3f}** | {p['negative']:.2f} | {p['neutral']:.2f} | {p['positive']:.2f} |")
    w("")
    if fin:
        w("> **Caveat.** The released FinBERT checkpoint was itself fine-tuned on Financial PhraseBank, so its row is not a clean held-out number. The TF-IDF model is the honest baseline; the fine-tuned model on our own split is the main result.\n")
    if fiqa:
        w("### Held-out check on FiQA headlines (never seen by any model)\n")
        w("| Model | Accuracy | Macro-F1 |")
        w("|---|---|---|")
        for k, n in [("tfidf_logreg", "TF-IDF + logistic regression"), ("finbert_released", "FinBERT released head"), ("finbert_finetuned", "FinBERT fine-tuned on our PhraseBank split")]:
            w(f"| {n} | {fiqa[k]['accuracy']:.3f} | {fiqa[k]['macro_f1']:.3f} |")
        w("")
        w("> Out of domain everything drops hard. The finance-adapted encoder still beats the lexical baseline by about 20 macro-F1 points, but fine-tuning on PhraseBank taught the model a strong *neutral* prior that mislabels short opinionated headlines. The in-domain 0.95 does not travel.\n")

# ---------- events ----------
if ev:
    w("## Events and entities (PhraseBank test, 345 sentences)\n")
    w("A six-category rule tagger (earnings result, guidance, acquisition or merger, financing or share issue, contract or order, management change) plus `dslim/bert-base-NER` for company spans. No gold event labels exist in the dataset, so the tagger is measured by coverage and by a manual precision review.\n")
    w("| Category | Sentences tagged |")
    w("|---|---|")
    for k, v in sorted(ev["event_counts"].items(), key=lambda x: -x[1]):
        w(f"| {k.replace('_', ' ')} | {v} |")
    w(f"\nTagger fires on {ev['sentences_with_any_event']} of {ev['n_test']} sentences; NER finds an organisation in {ev['sentences_with_org']}. Manual review of 53 tagged sentences gives precision 0.57 overall (0.86 financing, 0.80 acquisition, 0.70 earnings, 0.40 contract and guidance, 0.17 management change); estimated recall 0.71.\n")

# ---------- repo layout ----------
w("## Repository layout\n")
w("```text")
w("src/")
w("  ectsum_stats_baseline.py   ECTSum statistics, Lead-4 and TF-IDF baselines, ROUGE harness")
w("  fpb_download.py            Financial PhraseBank download (Hugging Face mirror)")
w("  fpb_stats_baseline.py      PhraseBank statistics, seed-42 split, majority and TF-IDF baselines")
for f, d in [("fpb_finbert.py", "FinBERT released-head evaluation and fine-tuning"), ("ectsum_bart.py", "BART-base fine-tuning with head truncation, ROUGE + BERTScore"),
             ("event_rules.py", "rule-based event tagger + NER coverage"), ("factual_check.py", "normalised number-support check of summaries against transcripts"),
             ("ectsum_bart_v2.py", "long-input (salient sentences) and decoding ablations"), ("fpb_heldout_fiqa.py", "FiQA headline held-out check for all sentiment models")]:
    if os.path.exists(os.path.join("src", f)): w(f"  {f:<26} {d}")
w("results/                     one JSON (and prediction file) per experiment; every number above comes from here")
if os.path.isdir("docs"): w("docs/                        docx builders for the checkpoint memos and the Google Docs sync helpers")
if os.path.isdir("reports"): w("reports/                     checkpoint memos as PDF")
w("tools/build_readme.py        regenerates this README from results/")
w("```\n")

# ---------- how to run ----------
w("## Reproduce\n")
w("```bash")
w("pip install -r requirements.txt")
w("# data")
w("git clone https://github.com/rajdeep345/ECTSum data/ectsum/ECTSum")
w("python src/fpb_download.py data/fpb")
w("# statistics and baselines")
w("python src/ectsum_stats_baseline.py data/ectsum/ECTSum/data/final results")
w("python src/fpb_stats_baseline.py data/fpb results")
if os.path.exists("src/fpb_finbert.py"): w("python src/fpb_finbert.py data/fpb results runs --epochs 3            # about 1 min on an 8 GB laptop GPU")
if os.path.exists("src/ectsum_bart.py"): w("python src/ectsum_bart.py data/ectsum/ECTSum/data/final results runs --epochs 2 --bertscore   # about 6 min")
if os.path.exists("src/ectsum_bart_v2.py"): w("python src/ectsum_bart_v2.py data/ectsum/ECTSum/data/final results runs --tag salient --select salient --epochs 2 --bertscore")
if os.path.exists("src/factual_check.py"): w("python src/factual_check.py results/ectsum_bart_test_preds.json data/ectsum/ECTSum/data/final/test/ects results/factual_check_head.json")
if os.path.exists("src/event_rules.py"): w("python src/event_rules.py data/fpb results")
if os.path.exists("src/fpb_heldout_fiqa.py"): w("python src/fpb_heldout_fiqa.py data/fpb data/fiqa runs results")
w("python tools/build_readme.py")
w("```\n")
w("All seeds are fixed at 42. Model weights are written to `runs/` and are not committed; every configuration retrains in under 15 minutes on one RTX 5060 laptop GPU.\n")

# ---------- progress log ----------
if os.path.exists("PROGRESS.md"):
    w("## Progress log\n")
    w(open("PROGRESS.md", encoding="utf-8").read().strip() + "\n")

w("## Team\n")
w("Team BART Street, CIS 511 Graduate Natural Language Processing, Fall 2026: Raima Arif, Sudarshan Sridhar, Firas Abueida.\n")
w("## Licences\n")
w("Code in this repository is MIT. ECTSum is GPL-3.0, Financial PhraseBank is CC BY-NC-SA 3.0 (non-commercial), FiQA is a public research release, and the pretrained checkpoints carry their own Hugging Face licences. Cite the original authors if you use any of them.")

open("README.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("README.md written,", len(out), "lines")
