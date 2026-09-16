# T02 cached decode

Status: pass

SHA: 4f2845da8a3565d8bbedec082fdbf2fc40319099
Baseline: T00-305bf8f9dc429f1af60e; tolerance: T00-tol-fbd0eb931ac1c7153403

BF16, batch=1, 128 output tokens, EOS disabled for timing; one warmup and five repetitions.
Generation comparisons use the frozen T00 margin <= 0.5 rule under identical prefixes; exact IDs are separately reported.

| Backend | Prompt | TTFT ms | Steady TPOT ms | Boundary TPOT ms | Tokens/s | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF | 1024 | 97.435 | 29.800 | 37.630 | 32.004 | 7.898 |
| HF | 4096 | 311.517 | 29.720 | 37.539 | 30.453 | 8.643 |
| vLLM | 1024 | 152.463 | 41.042 | 41.575 | 23.824 | 8.620 |
| vLLM | 4096 | 1592.784 | 41.113 | 42.010 | 18.747 | 16.040 |

Full-history Python KV and dense prefill; no serving, concurrency, graphs, paging, or long-context performance claim.
TTFT includes prefill and argmax; TPOT includes preparation, forward, argmax and device synchronization. Load/compile is outside warm measurements.
