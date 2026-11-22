"""Download FinancialPhraseBank-v1.0.zip from the HF repo (takala/financial_phrasebank) and
parse Sentences_*Agree.txt (sentence@label, latin-1) into CSVs. Usage: python -E -P fpb_download.py OUT_DIR"""
import sys, os, csv, zipfile, shutil
from huggingface_hub import hf_hub_download
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
p = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
zp = os.path.join(out, "FinancialPhraseBank-v1.0.zip"); shutil.copy(p, zp)
with zipfile.ZipFile(zp) as z:
    names = [n for n in z.namelist() if n.endswith(".txt") and "Sentences_" in n]
    for n in names:
        rows = []
        for line in z.read(n).decode("latin-1").splitlines():
            line = line.strip()
            if not line: continue
            s, lab = line.rsplit("@", 1)
            rows.append((s.strip(), lab.strip()))
        cfg = "sentences_" + os.path.basename(n).replace("Sentences_", "").replace("Agree.txt", "agree").lower()
        with open(os.path.join(out, f"{cfg}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["sentence", "label"]); w.writerows(rows)
        print(cfg, len(rows), sorted(set(r[1] for r in rows)))
