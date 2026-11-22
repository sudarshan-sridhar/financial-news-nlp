# ECTSum statistics and extractive baselines

## Pairs per split

| split | n |
|---|---|
| train | 1681 |
| val | 249 |
| test | 495 |
| total | 2425 |

## Transcript word counts

| split | mean | median | min | max |
|---|---|---|---|---|
| train | 2860.3 | 2769 | 250 | 11672 |
| val | 2769.3 | 2729 | 772 | 8978 |
| test | 2818.1 | 2728 | 455 | 8439 |
| overall | 2842.4 | 2755 | 250 | 11672 |

## Summary word counts

| split | mean | median | min | max |
|---|---|---|---|---|
| train | 44.4 | 36 | 3 | 273 |
| val | 42.4 | 36 | 3 | 177 |
| test | 42.7 | 34 | 4 | 213 |
| overall | 43.8 | 35 | 3 | 273 |

## Transcripts over 1024 whitespace tokens

| split | n | share |
|---|---|---|
| train | 1638 | 97.4% |
| val | 245 | 98.4% |
| test | 484 | 97.8% |
| overall | 2367 | 97.6% |

## Mean bullet lines per summary

| split | bullets |
|---|---|
| train | 4.11 |
| val | 4.03 |
| test | 4.0 |
| overall | 4.08 |

## ROUGE F1 on test (n=495)

| system | R-1 | R-2 | R-L |
|---|---|---|---|
| lead4 | 9.36 | 1.6 | 6.57 |
| tfidf_top4 | 7.98 | 0.62 | 5.51 |

Wall time: 10.1 s
