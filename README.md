# semcache

A semantic caching layer for LLM apps - cache answers by meaning, not by
exact string, with the hit decision calibrated instead of guessed. Pure
TypeScript engine plus a demo that runs entirely in your browser: no server,
no API keys, no account.

**Live demo: https://aditya2375.github.io/semcache/**

The demo replays a labeled 280-query workload against a fictional helpdesk
backend (NimbusDB) and shows, for five hit policies side by side, the three
numbers that decide whether a semantic cache is safe to turn on: hit
precision, hit recall, and the wrong-hit rate - each with a 95% Wilson
confidence interval, because a rate without its interval is a marketing
number.

## The problem this solves

A semantic cache answers a new question from a cached one when they *mean*
the same thing. Get it wrong and the user gets a fluent answer to a question
they did not ask - at cache speed, with no regeneration to catch it. The
standard defense is a similarity threshold picked by vibes: 0.85? 0.9?
This project replaces the guess with a calibrated probability and derives
the operating point from costs:

```
serve cached  <=>  (1 - p) * W  <=  C      =>      p >= 1 - C/W
```

`p` is the calibrated probability that the nearest cached entry answers the
same question, `W` is what a wrong cached answer costs you, `C` is what
regenerating costs. At a wrong-answer cost multiple of 10, the operating
point is p* = 0.90. The tradeoff is a product decision made explicit, not a
constant hidden in a config file.

## What you can do in the demo

- **Console**: ask the helpdesk anything. Watch the decision card walk the
  lookup stages (guard, exact key, semantic candidates, policy), show the
  calibrated probability, and track cumulative spend against a no-cache
  baseline. The bundled workload's vectors are precomputed; a custom
  question downloads the pinned embedding model and runs it locally.
- **Workload**: replay all 280 labeled queries under a chosen policy, with
  a live event feed and a summary grid: precision, recall, wrong-hit rate,
  guard accuracy, cost saved, p50/p95 latency.
- **Thresholds**: one click sweeps 15 fixed thresholds and 8 calibrated
  operating points - 23 full replays in a couple of seconds - and plots
  precision-recall and wrong-hit curves on canvas.
- **Near-misses**: 24 hand-written trap questions ("How do I increase the
  connection limit?" is not "How many connections are allowed?"), each shown
  with what a naive threshold does (serves the wrong answer) versus what the
  calibrated policy does (declines and regenerates).
- **Method**: the serving rule, the calibration model, its held-out quality,
  and the known limits - in the page, not buried in this README.

## Measured results

Full replay of the 280-query labeled workload (60 exact repeats, 120
paraphrases, 84 first-asks, 16 uncacheable guards).
Reproduce with `node scripts/replay.mjs`.

| policy                    | hits | precision           | recall              | wrong-hit | cost saved |
|---------------------------|------|---------------------|---------------------|-----------|------------|
| fixed tau=0.75            | 199  | 24.6% [19.2-31.1]   | 27.2% [21.2-34.1]   | 53.6%     | 71.0%      |
| fixed tau=0.85            | 134  | 73.9% [65.9-80.6]   | 55.0% [47.7-62.1]   | 12.5%     | 47.7%      |
| fixed tau=0.92            | 93   | 97.8% [92.5-99.4]   | 50.6% [43.3-57.8]   | 0.7%      | 33.1%      |
| margin(tau=0.75, m=0.05)  | 150  | 79.3% [72.2-85.0]   | 66.1% [58.9-72.6]   | 11.1%     | 53.2%      |
| calibrated(p*=0.95)       | 60   | 100.0% [94.0-100.0] | 33.3% [26.9-40.5]   | 0.0%      | 21.0%      |

Brackets are 95% Wilson intervals. Honest wrinkles, visible rather than
hidden:

- **A hand-tuned fixed threshold is strong on this workload.** tau=0.92
  nearly matches the calibrated policy. The calibrated policy's claim is not
  that thresholds never work - it is that you cannot know 0.92 is the right
  number without fitting, and the right number moves when the model, domain,
  or cost ratio changes. Calibration turns a guessed constant into a fitted
  probability with a cost-derived operating point.
- **The margin feature learned a negative coefficient.** On this workload's
  hard negatives, the runner-up is usually also close, so a large gap over
  the runner-up correlates with the trap cluster, not with correctness.
  The Method tab says so plainly.
- **Recall is the price of safety.** At p*=0.95 the cache serves 33.3% of
  servable queries and saves 21.0% of baseline cost; loosen to p*=0.90 and
  recall rises to 41.1% with a 1.1% wrong-hit rate. That tradeoff is the
  product knob, and the Thresholds tab draws it.

## Calibration

The policy estimates p(correct) from four features - top similarity, margin
over the runner-up, log cache size, query length - with a logistic model
fitted on 86 labeled intent pairs and evaluated on 34 held-out pairs split
by intent group, so a paraphrase never leaks across folds. Held-out quality:
AUC 0.819, log-loss 0.542, Brier 0.176. Useful, not magic, and the demo says
so. Two engineering details worth stealing:

- **Degenerate features are frozen at fit time.** Cache size is constant in
  the fitting runs (the replay seeds all 60 canonical answers), so its
  coefficient is pinned to 0 with its column zeroed; float noise in a
  zero-variance feature would otherwise explode after standardization.
- **A missing runner-up is mean-imputed.** A cache holding one entry has no
  second candidate; the margin feature contributes zero (its training mean)
  instead of an extreme stand-in that a negative coefficient would turn
  into a veto. This bug is real and the console exposed it; there is a
  regression test.

## Engine

`src/core/`, pure TypeScript with zero dependencies:

- `cache.ts` - the semantic cache: cacheability guards, normalized exact
  keys, HNSW semantic candidates, TTL, LRU eviction with tombstoned graph
  rebuilds, and namespace + model-key isolation so entries written under one
  embedding model are invisible to another.
- `hnsw.ts` - the approximate index, from scratch (seeded, deterministic).
- `guards.ts` - lexical cacheability floor: time-relative, private-state,
  and imperative phrasings never enter the cache.
- `policy.ts` - fixed-threshold, margin, and calibrated policies; the
  cost-derived operating point.
- `calibrate.ts` - the logistic fit with group-disjoint folds and feature
  freezing.
- `workload.ts` / `replay.ts` - strict workload validation and the labeled
  replay runner shared by the CLI, the tests, and the browser.
- `stub.ts` - the simulated backend: answers composed from bundled NimbusDB
  documentation (tf-idf with IDF weighting), seeded latency and token
  counts. Every number it reports is labeled simulated.
- `precomputed.ts` - loader for the bundled vector artifact; texts outside
  the artifact are rejected so precomputed data can never masquerade as live
  model output.
- `metrics.ts` - Wilson intervals, AUC, log-loss, Brier, percentiles.

## Models and prices

- Embeddings: `Xenova/bge-small-en-v1.5` @ `ea104dac`, 384-dim, q8 ONNX,
  no query instruction prefix (query-to-query matching is symmetric; the
  bge instruction is for asymmetric query-to-passage retrieval).
- The demo's bundled vectors are precomputed with the exact pinned model and
  verified in CI (`scripts/verify-precomputed.mjs` fails below 0.995 cosine
  on regeneration; observed 0.999999).
- ONNX Runtime Web runs single-threaded from wasm files the repo ships
  itself (`docs/ort/`), because GitHub Pages cannot send the cross-origin
  isolation headers threaded wasm requires.
- Dollar figures use published gpt-4o-mini prices ($0.15 / $0.60 per 1M
  input / output tokens, openai.com/api/pricing, retrieved 2026-11-13) and
  are illustrative: the backend is simulated, so the dollars measure the
  policy, not a real invoice.

## Data and determinism

NimbusDB is fictional. The 60 canonical Q&A, 120 paraphrases, 24 near-miss
traps, and 16 guard queries are hand-authored; `scripts/gen-data.mjs`
assembles the 280-query workload deterministically (seed 20261113). No user
data appears anywhere. The replay is perfectly reproducible: same seed, same
pinned model, same numbers.

## Develop and verify

```sh
npm ci
npm run build                      # typecheck + compile
npm test                           # 58 unit + integration tests
node tests/browser-smoke.mjs       # 19 end-to-end checks in headless Chrome
node scripts/replay.mjs            # prints the policy table above
node scripts/precompute.mjs        # regenerate vectors + calibration fit
node scripts/verify-precomputed.mjs  # fails if the bundled artifact drifts
node scripts/build-web.mjs         # rebuild docs/ exactly as CI checks it
sh tests/check-docs.sh             # README claims must match the code
```

CI runs all of the above plus a docs/ rebuild diff check, dependency audit,
secret scanning, and semgrep rules banning innerHTML, eval, and
user-built fetch URLs.

## Security

No backend, no cookies, no analytics, no keys, no user data. The page ships
a strict CSP (`default-src 'self'`, model downloads allowed only from the
pinned Hugging Face hosts), renders all query and answer text through
`textContent` (never `innerHTML`), validates the workload and artifacts
strictly at load, and caps the console cache. See SECURITY_REVIEW.md for the
full item-by-item review.
