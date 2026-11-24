"""Rule and dictionary event tagger + off-the-shelf NER over the Financial PhraseBank test split.
Usage: python -E -P event_rules.py DATA_DIR RESULTS_DIR"""
import sys, os, re, json, collections
import pandas as pd, torch
from transformers import pipeline

data_dir, res_dir = sys.argv[1], sys.argv[2]
df = pd.read_csv(os.path.join(data_dir, "sentences_75agree.csv"))
sp = json.load(open(os.path.join(data_dir, "split_seed42.json")))
test = df.loc[sp["test"]].reset_index(drop=True)

RULES = {
 "earnings_result": r"\b(net (profit|loss|income|sales)|operating (profit|loss|income|margin)|pre-?tax (profit|loss)|ebit(da)?|earnings per share|eps|revenues?|turnover|sales|profit|loss|margin|result)\b.*\b(rose|fell|increased|decreased|grew|declined|widened|narrowed|was|were|totaled|totalled|amounted|dropped|climbed|jumped|slipped|surged|improved|weakened|up|down|of|to|reported?s?)\b|\breport(ed|s)? (eps|a (net )?(profit|loss)|net sales|revenue)",
 "guidance": r"\b(expects?|expected|forecasts?|outlook|guidance|estimates?|targets?|aims? to|anticipates?|will be|is seen|sees)\b",
 "acquisition_merger": r"\b(acqui(re|res|red|sition)|merger|merge|takeover|buy(s|ing)? (a )?stake|divest(ment|ed)?|sell(s|ing)? (its )?(stake|unit|business|subsidiary|operations))\b",
 "financing_share_issue": r"\b(share issue|rights issue|new shares|bond|convertible|loan|credit facility|financing|dividend|buy-?back|repurchase|capital increase|listing|ipo)\b",
 "contract_order": r"\b(contract|order|agreement|deal|tender|supply|deliver(y|ies)?|signed|awarded|won|partnership|cooperation)\b",
 "management_change": r"\b(appointed|appoints?|resign(s|ed|ation)?|steps? down|named as|new (ceo|cfo|chairman|president|managing director)|ceo|chief executive|board of directors)\b",
}
MONEY = r"(EUR|USD|\$|€|mn|mln|million|billion|bn|m\b|%|percent|cents?)"
def tag(s):
    t = s.lower()
    return [k for k, rx in RULES.items() if re.search(rx, t)]
test["events"] = test["sentence"].apply(tag)
test["money"] = test["sentence"].str.contains(MONEY, regex=True, case=False)
cnt = collections.Counter(e for es in test["events"] for e in es)
n_any = int((test["events"].str.len() > 0).sum()); n_multi = int((test["events"].str.len() > 1).sum())

dev = 0 if torch.cuda.is_available() else -1
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple", device=dev)
outs = ner(test["sentence"].tolist(), batch_size=32)
test["orgs"] = [[e["word"] for e in o if e["entity_group"] == "ORG"] for o in outs]
n_org = int((test["orgs"].str.len() > 0).sum())
ent_cnt = collections.Counter(e["entity_group"] for o in outs for e in o)

by_label = {l: dict(collections.Counter(e for es in test.loc[test.label == l, "events"] for e in es)) for l in ["negative", "neutral", "positive"]}
R = {"n_test": len(test), "event_counts": dict(cnt), "sentences_with_any_event": n_any, "sentences_with_multiple_events": n_multi,
     "sentences_with_money_or_percent": int(test["money"].sum()), "ner_model": "dslim/bert-base-NER", "entity_counts": dict(ent_cnt),
     "sentences_with_org": n_org, "events_by_sentiment_label": by_label}
os.makedirs(res_dir, exist_ok=True)
json.dump(R, open(os.path.join(res_dir, "event_rules.json"), "w"), indent=2)
test[["sentence", "label", "events", "money", "orgs"]].to_json(os.path.join(res_dir, "event_rules_test.json"), orient="records", indent=1)
print(json.dumps(R, indent=1))
for i, r in test.head(8).iterrows(): print(r.label, r.events, r.orgs, "|", r.sentence[:120])
