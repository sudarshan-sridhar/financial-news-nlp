# Financial PhraseBank: statistics and baselines

Source: HF repo takala/financial_phrasebank -> data/FinancialPhraseBank-v1.0.zip (raw txt parsed; HF loading script unsupported in datasets 5.x)

## Label distribution
| subset | n | negative | neutral | positive |
|---|---|---|---|---|
| 50agree | 4846 | 604 (12.46%) | 2879 (59.41%) | 1363 (28.13%) |
| 66agree | 4217 | 514 (12.19%) | 2535 (60.11%) | 1168 (27.7%) |
| 75agree | 3453 | 420 (12.16%) | 2146 (62.15%) | 887 (25.69%) |
| allagree | 2264 | 303 (13.38%) | 1391 (61.44%) | 570 (25.18%) |

## 75agree sentence stats
- words/sentence: mean 22.76, median 21.0, max 81, min 2
- exact duplicate sentences: 5 (conflicting labels: 0); n after dedup: 3448

## Split (seed 42, stratified 80/10/10, after dedup)
| split | n | negative | neutral | positive |
|---|---|---|---|---|
| train | 2758 | 336 | 1713 | 709 |
| dev | 345 | 42 | 214 | 89 |
| test | 345 | 42 | 214 | 89 |

## Baselines (test split)
| model | accuracy | macro-F1 |
|---|---|---|
| majority (neutral) | 0.6203 | 0.2552 |
| TF-IDF(1-2g)+LogReg balanced | 0.8638 | 0.8069 |
| FinBERT zero-shot | skipped | transformers/torch not installed |

TF-IDF+LogReg per-class F1: negative 0.7164, neutral 0.9135, positive 0.7907

Confusion matrix (rows=true, cols=pred; negative/neutral/positive):
```
[24, 11, 7]
[0, 206, 8]
[1, 20, 68]
```
