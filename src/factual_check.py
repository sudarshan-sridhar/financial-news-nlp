"""Normalized number check of BART summaries against source transcripts (all 495 test docs).
Usage: python -E -P factual_check.py PRED_JSON ECTS_DIR OUT_JSON"""
import sys, json, re, os
preds=json.load(open(sys.argv[1])); root=sys.argv[2]
def norm_nums(t):
    t=t.lower().replace(",","")
    t=re.sub(r"\bq([1-4])\b","",t); t=re.sub(r"(\d)\s*(percent|%)",r"\1",t)
    t=re.sub(r"\$\s*(\d)",r"\1",t); t=re.sub(r"(\d+)\s*cents?",lambda m:str(int(m.group(1))/100),t)
    out=set()
    for x in re.findall(r"\d+(?:\.\d+)?",t):
        if len(x)>=2 or "." in x: out.add(x.rstrip("0").rstrip(".") if "." in x else x)
    return out
tot=0; docs_bad=0; nums=0; bad=0; per=[]
for p in preds:
    src=open(os.path.join(root,p["doc"]),encoding="utf-8",errors="replace").read()
    S=norm_nums(src); B=norm_nums(p["bart"]); miss=B-S
    tot+=1; nums+=len(B); bad+=len(miss)
    if miss: docs_bad+=1
    per.append({"doc":p["doc"],"n_numbers":len(B),"unsupported":sorted(miss)})
R={"n_docs":tot,"docs_with_unsupported_number":docs_bad,"share_docs":round(docs_bad/tot,3),"numbers_total":nums,"numbers_unsupported":bad,"share_numbers":round(bad/max(nums,1),3)}
json.dump({"summary":R,"per_doc":per},open(sys.argv[3],"w"),indent=1); print(R)
